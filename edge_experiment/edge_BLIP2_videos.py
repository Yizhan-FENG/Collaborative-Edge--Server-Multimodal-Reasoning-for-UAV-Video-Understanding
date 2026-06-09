import torch
import transformers
import cv2
import numpy as np
import os
import re
import time
import psutil
import sys
import json
import datetime
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from ultralytics import YOLO
from sklearn.cluster import KMeans
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore") # 屏蔽大部分警告信息

# --- 配置 ---
os.chdir("/data3/health")

BLIP2_MODEL_PATH = "/data3/health/model/blip2-opt-6.7b"
YOLO_WORLD_MODEL_PATH = "yolov8l-world.pt"
YOLO_SEG_MODEL_PATH = "/data3/health/projects/BLIP2/yoloe-v8l-seg.pt"


VIDEO_BASE_DIR = "/data3/health/wyy/data/MSR-VTT/video/"  # 视频文件所在的共同目录
MSRVTT_JSON_PATH = "/data3/health/wyy/data/MSR-VTT/msrvtt_test_1k.json"  # 提供的JSON文件路径
# VIDEO_PATH = "/data3/health/data/无人机第一视角视频.mp4"

DEVICE_ID = "cuda:6"

# 定义结果保存路径
OUTPUT_BASE_DIR = "/data3/health/wyy/BLIP2/experiment/data/msrvtt" 
RESULTS_JSONL_PATH = os.path.join(OUTPUT_BASE_DIR, "videos_analysis_results.jsonl")
KEYFRAMES_SAVE_DIR = os.path.join(OUTPUT_BASE_DIR, "keyframes")

def save_results(record_time, video_duration, keyframe_count, raw_frame_count, analysis_report, frame_descriptions, rep_frames, rep_tstamps, yolo_log, video_path=None):
    """
    保存程序运行结果到JSONL文件和关键帧图片到文件夹。
    
    Args:
        record_time: 程序运行时间戳
        video_duration: 原始视频长度（秒）
        keyframe_count: 关键帧数量
        raw_frame_count: 原始抽帧数量
        analysis_report: 生成的视频总结文本
        frame_descriptions: 关键帧描述信息列表
        rep_frames: 关键帧PIL.Image对象列表
        rep_tstamps: 关键帧时间戳列表
        video_path: 原始视频路径，用于提取视频名
    """
    # 1. 确保输出目录存在
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # 2. 从视频路径提取基础名称，用于命名
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    # 为当前视频创建专属的关键帧保存目录
    video_keyframes_dir = os.path.join(KEYFRAMES_SAVE_DIR, video_basename)
    os.makedirs(video_keyframes_dir, exist_ok=True)
    
    # 3. 保存关键帧图片
    saved_frame_paths = []
    for idx, (frame, ts) in enumerate(zip(rep_frames, rep_tstamps)):
        # 生成文件名，例如: video0_keyframe_001_2.50s.jpg
        filename = f"keyframe_{idx:03d}_{ts:.2f}s.jpg"
        filepath = os.path.join(video_keyframes_dir, filename)
        frame.save(filepath)
        saved_frame_paths.append(filepath)  # 保存的是包含子文件夹的完整路径
        
    # 4. 构建要保存的JSON数据
    result_record = {
        "program_run_time": record_time,
        "video_source": video_path,
        "video_duration_seconds": video_duration,
        "raw_sampled_frames_count": raw_frame_count,
        "key_frames_count": keyframe_count,
        "video_summary": analysis_report,
        "yolo_detection_segmentation_log": yolo_log,
        "key_frames_info": frame_descriptions,  # 已包含timestamp和description
        "saved_keyframes_paths": saved_frame_paths
    }
    
    # 5. 以追加模式写入JSONL文件
    with open(RESULTS_JSONL_PATH, 'a', encoding='utf-8') as f:
        json_line = json.dumps(result_record, ensure_ascii=False, default=str)
        f.write(json_line + '\n')
    
    # print(f"\n[结果保存完成]")
    # print(f"  - 数据记录已保存至: {RESULTS_JSONL_PATH}")
    # print(f"  - 关键帧图片已保存至目录: {KEYFRAMES_SAVE_DIR}")
    # print(f"  - 本条目包含 {keyframe_count} 个关键帧")



def get_vram_gb(device):
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(device) / 1024 / 1024 / 1024
    return 0.0

def extract_features_for_clustering(video_path, model, processor, device, sample_rate=1):
    """抽帧函数"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        return np.array([]), [], [], 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count_total / fps if fps > 0 else 0.0  # 计算总时长


    frame_interval = int(fps * sample_rate)
    features, frames, timestamps = [], [], []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_count % frame_interval == 0:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
            with torch.no_grad():
                vis_out = model.vision_model(pixel_values=inputs.pixel_values)
                img_embeds = vis_out.last_hidden_state
                att_mask = torch.ones(img_embeds.size()[:-1], dtype=torch.long, device=img_embeds.device)
                q_tokens = model.query_tokens.expand(img_embeds.shape[0], -1, -1)
                q_out = model.qformer(query_embeds=q_tokens, encoder_hidden_states=img_embeds, encoder_attention_mask=att_mask)
                feats = q_out.last_hidden_state
            features.append(feats.mean(dim=1).squeeze().detach().cpu().numpy())
            frames.append(image)
            timestamps.append(frame_count / fps)
        frame_count += 1
    cap.release()
    return np.array(features), frames, timestamps, video_duration  # 返回的是: 视觉特征、关键帧图像、关键帧时间步、原始视频长度

def get_representative_frames(features, frames, timestamps):
    """
    聚类函数
    Args:
        features: 一次抽帧的视觉特征
        frames: 抽取的帧
        timestamps: 时间步
    """
    if len(features) < 2: 
        return frames, timestamps
    duration = timestamps[-1] if timestamps else 0
    K = min(len(features), max(2, int(duration / 4)))
    kmeans = KMeans(n_clusters=K, random_state=0, n_init='auto').fit(features)
    rep_frames_info = []
    for i in range(K):
        idx = np.where(kmeans.labels_ == i)[0]
        if len(idx) == 0: continue
        centroid = kmeans.cluster_centers_[i]
        dist = np.linalg.norm(features[idx] - centroid, axis=1)
        best_idx = idx[np.argmin(dist)]
        rep_frames_info.append({"frame": frames[best_idx], "ts": timestamps[best_idx]})
    rep_frames_info.sort(key=lambda x: x["ts"])
    return [x["frame"] for x in rep_frames_info], [x["ts"] for x in rep_frames_info]


def extract_video_features(frames, model, processor, device):
    """
    提取关键帧的视觉特征，并拼接所有关键帧的视觉特征
    Args:
        frames: 关键帧
        model: BLIP-2模型
        processor: 处理器
    """
    embeds_list = []
    with torch.no_grad():
        for frame in frames:
            pixel_values = processor(images=frame, return_tensors="pt").to(device, torch.float16).pixel_values
            img_embeds = model.vision_model(pixel_values).last_hidden_state
            att_mask = torch.ones(img_embeds.shape[:-1], dtype=torch.long, device=img_embeds.device)
            q_tokens = model.query_tokens.expand(img_embeds.shape[0], -1, -1)
            q_out = model.qformer(query_embeds=q_tokens, encoder_hidden_states=img_embeds, encoder_attention_mask=att_mask)
            embeds_list.append(model.language_projection(q_out.last_hidden_state.to(torch.float16)))
    vid_embeds = torch.cat(embeds_list, dim=1)
    vid_mask = torch.ones(vid_embeds.shape[:-1], dtype=torch.long, device=vid_embeds.device)
    return vid_embeds, vid_mask

def generate_visual_log(yolo_det, yolo_seg, frames, timestamps, targets, task_type):
    """
    基于YOLO模型，进行目标检测、图像分割任务，并生成日志
    """
    logs = []
    model = yolo_seg if task_type == 'segmentation' else yolo_det
    try:
        if targets: model.set_classes(targets)
    except: pass
    for i, frame in enumerate(frames):
        ts = timestamps[i]
        res = model(frame, conf=0.2, verbose=False)[0]
        frame_objs = []
        if res.boxes:
            for j, box in enumerate(res.boxes):
                cls_name = model.names[int(box.cls)]
                coords = box.xyxyn[0].tolist() 
                if task_type == 'detection':
                    confidence = box.conf.item()
                    info = f"{cls_name} found at [{coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}, {coords[3]:.2f}] conf:{confidence:.2f}"
                elif task_type == 'segmentation':
                    confidence = box.conf.item()
                    has_mask = hasattr(res, 'masks') and res.masks is not None
                    mask_info = "with mask" if has_mask else "no mask"
                    info = f"{cls_name} segmented at [{coords[0]:.2f}, {coords[1]:.2f}] {mask_info} conf:{confidence:.2f}"
                frame_objs.append(info)
        if frame_objs:
            log_str = "; ".join(frame_objs)
            logs.append(f"[Time: {ts:.1f}s] {log_str}")
        else:
            logs.append(f"[Time: {ts:.1f}s] No {task_type} targets found.")
    return "\n".join(logs)

def add_pos_encoding(video_features):
    batch, seq, hidden = video_features.shape
    dtype = video_features.dtype
    video_features = video_features.to(torch.float32)
    pos = torch.arange(seq, dtype=torch.float32, device=video_features.device).unsqueeze(1)
    div = torch.exp(torch.arange(0, hidden, 2, dtype=torch.float32, device=video_features.device) * -(np.log(10000.0)/hidden))
    pe = torch.zeros(seq, hidden, device=video_features.device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return (video_features + pe.unsqueeze(0)).to(dtype)

def describe_single_frame(frame, model, processor, device, prompt="Question: Describe this image in one sentence. Answer:"):
    """
    Args:
        frame: PIL.Image, 输入图片
        model: BLIP2模型
        processor: 处理器
        device: 运行设备
        prompt: 文本提示词
    Returns:
        str: 图片描述文本
    """
    # 直接使用官方API方式处理单帧
    inputs = processor(images=frame, text=prompt, return_tensors="pt").to(device, torch.float16)
    
    # 生成描述
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=50)
    
    # 解码结果
    description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    # 移除可能重复的prompt部分
    if description.startswith(prompt):
        description = description[len(prompt):].strip()
    
    return description


def process_single_video(video_path, blip_model, processor, yolo_det, yolo_seg, device):
    """
    处理单个视频的核心函数。
    由原来的 run_once 函数改造而来，接收视频路径和已加载的模型作为参数。
    """
    record_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # --- 视频预处理 ---
    print(f">>> 正在处理视频: {os.path.basename(video_path)}")
    t_prep_start = time.time()

    # 2. 原始抽帧 (Sample)
    feats, frames, tstamps, video_duration = extract_features_for_clustering(video_path, blip_model, processor, device, sample_rate=2)
    raw_frame_count = len(frames)

    # 3. 聚类筛选关键帧 (Cluster)
    rep_frames, rep_tstamps = get_representative_frames(feats, frames, tstamps)
    keyframe_count = len(rep_frames)

    # 4. YOLO模型的检测/分割任务
    user_prompt = "describe the video"
    targets = re.findall(r'\[(.*?)\]', user_prompt)
    task_type = 'segmentation' if 'segment' in user_prompt.lower() else 'detection'

    visual_log = generate_visual_log(yolo_det, yolo_seg, rep_frames, rep_tstamps, targets, task_type)

    frame_descriptions = []
    for idx, (frame, ts) in enumerate(zip(rep_frames, rep_tstamps)):
        description = describe_single_frame(
            frame, 
            blip_model, 
            processor, 
            device,
            prompt="Question: Describe this image in one sentence. Answer:"
        )
        frame_descriptions.append({
            "frame_index": idx,
            "timestamp": ts,
            "description": description
        })
        print(f"  关键帧 {idx} (时间: {ts:.2f}s): {description}")

    vid_embeds, vid_mask = extract_video_features(rep_frames, blip_model, processor, device)
    vid_embeds = add_pos_encoding(vid_embeds)
    vid_mask = vid_mask.to(torch.long)

    # 5. 文本提示词
    final_input = """Question: Summarize in one sentence what objects are in the video and what is happening. Answer:"""
    print(final_input)

    text_inputs = processor(text=final_input,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=1024).to(vid_embeds.device)
    text_embeds = blip_model.language_model.get_input_embeddings()(text_inputs.input_ids).to(torch.float16)

    # 6. 视觉前缀和文本embedding
    inputs_embeds = torch.cat([vid_embeds, text_embeds], dim=1)
    combined_attention_mask = torch.cat([vid_mask.to(torch.long), text_inputs.attention_mask], dim=1)

    # 7. 生成总结
    with torch.no_grad():
        generated_ids = blip_model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
            early_stopping=True
        )

    analysis_report = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    print("\n" + "="*50)
    print(f"视频分析报告 [{os.path.basename(video_path)}]:")
    print("="*50)
    print(analysis_report)
    print("="*50)
    print(f"程序运行时间: {record_time}")
    print(f"关键帧数量: {keyframe_count}")
    print(f"原始视频长度: {video_duration:.2f}秒")

    # 保存所有结果
    save_results(
        record_time=record_time,
        video_duration=video_duration,
        keyframe_count=keyframe_count,
        raw_frame_count=raw_frame_count,
        analysis_report=analysis_report,
        frame_descriptions=frame_descriptions,
        rep_frames=rep_frames,
        rep_tstamps=rep_tstamps,
        yolo_log=visual_log,
        video_path=video_path  # 使用传入的video_path
    )

if __name__ == "__main__":
    # 0. 加载模型 (只加载一次)
    print(">>> 正在加载模型...")
    t0 = time.time()
    device = torch.device(DEVICE_ID)
    
    processor = Blip2Processor.from_pretrained(BLIP2_MODEL_PATH)
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_MODEL_PATH, device_map={"": DEVICE_ID}, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    
    yolo_det = YOLO(YOLO_WORLD_MODEL_PATH)
    yolo_det.to(device)
    yolo_seg = YOLO(YOLO_SEG_MODEL_PATH)
    yolo_seg.to(device)
    
    t1 = time.time()
    print(f">>> 模型加载完成，耗时 {t1 - t0:.2f} 秒")

     # 1. 加载MSR-VTT JSON元数据文件
    try:
        with open(MSRVTT_JSON_PATH, 'r', encoding='utf-8') as f:
            video_metadata_list = json.load(f)
        print(f">>> 成功加载元数据文件，共 {len(video_metadata_list)} 个视频。")
    except FileNotFoundError:
        print(f"错误：未找到元数据文件 {MSRVTT_JSON_PATH}")
        print(f"请确保文档2 `msrvtt_test_1k.json` 已保存到该路径，或修改代码中的 `MSRVTT_JSON_PATH` 变量。")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误：JSON文件格式不正确 - {e}")
        sys.exit(1)

         # 2. 遍历并处理每个视频
    processed_count = 0
    for item in video_metadata_list:
        video_filename = item.get("video")  # 例如 "video7020.mp4"
        if not video_filename:
            print(f"警告：跳过一项，未找到 'video' 字段: {item}")
            continue
            
        video_full_path = os.path.join(VIDEO_BASE_DIR, video_filename)
        
        # 检查视频文件是否存在
        if not os.path.exists(video_full_path):
            print(f"警告：视频文件不存在，跳过。路径: {video_full_path}")
            continue
        
        print(f"\n{'#'*60}")
        print(f"开始处理视频 ({processed_count + 1}/{len(video_metadata_list)}): {video_filename}")
        print(f"{'#'*60}")
        
        try:
            process_single_video(video_full_path, blip_model, processor, yolo_det, yolo_seg, device)
            processed_count += 1
        except Exception as e:
            print(f"处理视频 {video_filename} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
            print(f"跳过此视频，继续处理下一个。")
    
    print(f"\n{'='*60}")
    print(f"批量处理完成！成功处理 {processed_count}/{len(video_metadata_list)} 个视频。")
    print(f"所有结果已保存至基础目录: {OUTPUT_BASE_DIR}")
    print(f"   - 分析日志: {RESULTS_JSONL_PATH}")
    print(f"   - 关键帧图片: {KEYFRAMES_SAVE_DIR}")
    print(f"{'='*60}")