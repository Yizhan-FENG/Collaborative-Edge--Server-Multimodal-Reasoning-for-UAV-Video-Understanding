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
import glob
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from ultralytics import YOLO
from sklearn.cluster import KMeans
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# --- 配置 ---
os.chdir("/data3/health")

LLM_MODEL_PATH = "/data3/health/model/Qwen3-VL-8B-Instruct"
USE_NEW_LLM = True
DEVICE_ID = "cuda:4"

# 1. 定义多视频的输入路径
# 结构化文本文件（包含所有视频的报告）
MULTI_VIDEO_REPORTS_PATH = "/data3/health/wyy/BLIP2/experiment/data/msrvtt/structured_reports/without_yolo_videos_analysis_results_human_readable_20260326_145436.txt"
# 关键帧图片的基础目录（每个视频有独立的子文件夹）
KEYFRAMES_BASE_DIR = "/data3/health/wyy/BLIP2/experiment/data/msrvtt/keyframes"
# 输出目录
OUTPUT_BASE_DIR = "/data3/health/wyy/BLIP2/experiment/Ablation_experiment/without_yolo_information/description/qwen3_vl_results"

# 确保输出目录存在
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)


def parse_multi_video_reports(reports_file_path):
    """
    从结构化文本文件中解析所有视频的报告
    返回：视频报告列表，每个元素是一个字典
    """
    with open(reports_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配从"报告 #数字"开始，到下一个"报告 #数字"或文件末尾之前的所有内容
    report_pattern = r'报告 #(\d+)[\s\S]*?(?=报告 #\d+|$)'
    reports = []
    
    for match in re.finditer(report_pattern, content):
        report_text = match.group(0).strip()  # 获取整个匹配的文本并去除首尾空白
        report_num = int(match.group(1))
        
        # 提取全局时序信息
        global_info_match = re.search(r'【通用视频全局时序信息】([\s\S]*?)【BLIP2全局语义摘要】', report_text)
        global_info = global_info_match.group(1).strip() if global_info_match else "无法提取全局时序信息"
        
        # 提取BLIP2全局语义摘要
        summary_match = re.search(r'【BLIP2全局语义摘要】([\s\S]*?)【关键帧绑定文本列表】', report_text)
        blip2_summary = summary_match.group(1).strip() if summary_match else "无法提取BLIP2摘要"
        
        # 提取关键帧绑定文本
        frames_text = []
        frame_pattern = r'帧ID:(\w+)\|时间戳:([\d.]+)s(.*?)(?=帧ID:|$)'
        for frame_match in re.finditer(frame_pattern, report_text, re.DOTALL):
            frame_id, timestamp, frame_content = frame_match.groups()
            frames_text.append({
                'frame_id': frame_id,
                'timestamp': timestamp,
                'content': frame_content.strip()
            })
        
        reports.append({
            'report_id': report_num,
            'global_info': global_info,
            'blip2_summary': blip2_summary,
            'frames_text': frames_text
        })
    
    print(f"✅ 成功解析 {len(reports)} 个视频报告")
    return reports


def find_video_folders(base_dir=KEYFRAMES_BASE_DIR):
    """
    查找base_dir下所有的视频文件夹，并按数字排序
    返回：排序后的视频文件夹路径列表
    """
    video_folders = []
    
    # 获取所有子文件夹
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith("video"):
            video_folders.append(item_path)
    
    # 按文件夹名称中的数字部分排序
    def extract_number(folder_path):
        folder_name = os.path.basename(folder_path)
        # 匹配"video"后面的数字部分
        match = re.search(r'video(\d+)', folder_name)
        if match:
            return int(match.group(1))
        return 0  # 如果没有数字，返回0
    
    video_folders.sort(key=extract_number)
    
    print(f"✅ 在 {base_dir} 下找到 {len(video_folders)} 个视频文件夹")
    for i, folder in enumerate(video_folders[:10]):  # 只显示前10个
        print(f"  {i+1}. {os.path.basename(folder)}")
    if len(video_folders) > 10:
        print(f"  ... 还有 {len(video_folders)-10} 个文件夹")
    
    return video_folders


def match_reports_to_folders(reports, video_folders):
    """
    将报告与视频文件夹进行匹配
    由于您的文件夹命名不是从0开始，我们按顺序匹配
    """
    matched_pairs = []
    
    # 检查数量是否匹配
    if len(reports) != len(video_folders):
        print(f"⚠ 警告：报告数量({len(reports)})与视频文件夹数量({len(video_folders)})不匹配")
        print(f"  将按照最小数量进行匹配: {min(len(reports), len(video_folders))}")
    
    # 按顺序匹配报告和文件夹
    min_count = min(len(reports), len(video_folders))
    for i in range(min_count):
        report = reports[i]
        folder_path = video_folders[i]
        folder_name = os.path.basename(folder_path)
        
        # 为报告添加视频ID（使用文件夹名称）
        report['video_id'] = folder_name
        report['folder_path'] = folder_path
        
        matched_pairs.append(report)
    
    return matched_pairs


def load_keyframe_images(folder_path):
    """
    从指定文件夹加载所有关键帧图片
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    
    # 对文件路径进行排序，确保每次输入的顺序一致
    image_paths.sort()
    
    if not image_paths:
        print(f"⚠ 警告：在路径 '{folder_path}' 下未找到任何图片文件")
        return []
    
    # 打开所有图片
    image_inputs = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            image_inputs.append(img)
        except Exception as e:
            print(f"⚠ 警告：无法打开图片 {img_path}，错误：{e}")
    
    return image_inputs


def process_single_video(report, model, processor, device, task_name="视频描述", jsonl_writer=None):
    """
    处理单个视频的核心函数
    如果提供了jsonl_writer，则将结果写入JSONL文件
    """
    folder_path = report['folder_path']
    video_id = report['video_id']
    report_id = report['report_id']
    
    print(f"\n{'='*60}")
    print(f"开始处理视频 {report_id}: {video_id}")
    print(f"关键帧文件夹: {folder_path}")
    print(f"{'='*60}")
    
    record_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. 读取关键帧图像
    image_inputs = load_keyframe_images(folder_path)
    
    if not image_inputs:
        print(f"❌ 错误：视频 {video_id} 没有可用的关键帧图片，跳过处理")
        return {
            'video_id': video_id,
            'report_id': report_id,
            'status': 'error',
            'error': '无关键帧图片',
            'process_time': record_time
        }
    
    print(f"✅ 从文件夹读取到 {len(image_inputs)} 张关键帧图片")
    
    # 2. 构建system prompt
    system_prompt = """You are a video captioning model. Your task is to generate a SINGLE, CONCISE English sentence that summarizes the main activity or event in the provided keyframe images.

【Core Rules for Video Captioning】
1.  **Output Format**: You must output ONLY a single English sentence. Do NOT output lists, bullet points, JSON, or multiple sentences.
2.  **Content Focus**: Describe the PRIMARY action, subject, and setting visible across the keyframes. Ignore minor details, timestamps, counts, or technical specifications (e.g., HUD, resolution, duration).
3.  **Style**: Use simple, declarative present-tense English, similar to common video caption datasets (e.g., "A person is doing X in a Y").
4.  **Length**: The sentence must be between 5 to 12 words. It should be a high-level summary, not a detailed play-by-play.
5.  **Visual Grounding**: The description must be directly supported by the content of the keyframe images. Do not hallucinate or infer actions not clearly shown.

【Example Outputs】
- Good: "A person is playing a video game on a laptop."
- Good: "Two people are sitting on a couch and talking."
- Bad: "The video starts with a person walking, then they sit down at 3.2s, and then they start typing..." (Too detailed, multiple events)
- Bad: "A man in a blue shirt uses a Dell laptop to play Grand Theft Auto V on a sunny day." (Too many unnecessary details)
- Bad: "Keyframes show a person possibly gaming." (Vague, not a full sentence)

Remember: Output ONLY the single caption sentence, nothing else.
"""

    # 3. 根据任务类型定义任务指令和输出格式
    task_configs = {
    "视频描述": {
        "instruction": "**Task: Video Captioning.** Look at the provided keyframe images. Generate ONE concise English sentence that describes the main activity or event happening in the video. Do not describe each keyframe separately. Do not mention time segments, durations, or technical details. Output ONLY the single sentence.",
        "output_format": "A person is riding a bicycle in a park."  # 这里直接放一个示例，告诉模型输出应该是这样的纯文本句子
    }
}
    
    # 4. 构建user prompt
    global_section = f"""【无人机视频全局时序信息】
                        {report['global_info']}
                        {report['blip2_summary']}"""
    
    # 构建关键帧视觉锚点与绑定文本部分
    frames_section = "【关键帧视觉锚点与绑定文本】\n"
    for i, frame_info in enumerate(report['frames_text']):
        if i < len(image_inputs):  # 确保不超出图片数量
            frames_section += f"""帧ID:{frame_info['frame_id']}|时间戳:{frame_info['timestamp']}s
{frame_info['content']}

"""
    
    # 完整的user prompt
    user_prompt = f"""{global_section}

{frames_section}
【执行任务指令】
{task_configs[task_name]['instruction']}

【输出格式要求】
{task_configs[task_name]['output_format']}"""
    
    # 5. 构建模型输入
    image_placeholders = [{"type": "image"} for _ in range(len(image_inputs))]
    image_placeholders.append({"type": "text", "text": user_prompt})
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": image_placeholders
        }
    ]
    
    text_inputs = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True,
    )
    
    inputs = processor(
        text=[text_inputs],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    # 6. 生成结果
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=1024,
            do_sample=False,
        )
    
     # 7. 解码输出
    response = processor.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # =========================================================
    # 8. 处理输出：【视频描述模式】= 纯文本，不再解析 JSON
    # =========================================================
    result_dict = {
        'video_id': video_id,
        'report_id': report_id,
        'process_time': record_time,
        'status': 'processing',
        'keyframe_count': len(image_inputs)
    }

    # ---- 清洗：去掉可能的 "system/user/assistant" 回声 ----
    text = response.strip()

    # 常见模型会把上一轮 role 回声出来，去掉它们
    for _bad in ["</s>", "<s>", "<|im_end|>", "<|im_start|>"]:
        text = text.replace(_bad, "")
    text = text.strip()

    # 如果模型还是带了 "Assistant:" / "Answer:" 前缀，截掉
    for prefix in ["Assistant:", "assistant:", "Answer:", "Caption:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # 取第一句（更稳）：遇到 ". " 或 "!" 或 "?" 截断
    for cut in [". ", "! ", "? "]:
        if cut in text:
            text = text.split(cut, 1)[0] + cut.strip()
            break

    # 兜底：如果仍然很长（模型又把旧 prompt 回声出来了），只保留前150字符保命
    if len(text) > 200:
        text = text[:150].rstrip() + "."

    # 基本判空
    if not text or len(text.split()) < 2:
        result_dict['status'] = 'error'
        result_dict['error'] = 'empty_or_too_short_output'
        result_dict['raw_response'] = response[:500]
    else:
        result_dict['status'] = 'success'
        # ★ 用你后面评测期望的字段名
        result_dict['full_video_summary'] = text   # 或者 result_dict['generated_caption'] = text
        # 保留一份原始便于 debug
        result_dict['raw_response'] = response[:500]

    # 写入 JSONL
    if jsonl_writer:
        jsonl_writer.write(json.dumps(result_dict, ensure_ascii=False) + '\n')

    print(f"✅ 视频 {report_id} ({video_id}) 处理完成")
    print(f"📝 描述: {text}")

    return result_dict


def main():
    """主函数：批量处理所有视频"""
    print("=" * 60)
    print("Qwen3-VL 多视频批量处理系统")
    print("=" * 60)
    
    # 1. 解析多视频报告
    print(">>> 正在解析多视频报告文件...")
    try:
        all_reports = parse_multi_video_reports(MULTI_VIDEO_REPORTS_PATH)
    except Exception as e:
        print(f"❌ 解析报告文件失败: {e}")
        return
    
    if not all_reports:
        print("❌ 未找到任何视频报告，程序退出")
        return
    
    # 2. 查找关键帧文件夹
    print(">>> 正在查找关键帧文件夹...")
    video_folders = find_video_folders(KEYFRAMES_BASE_DIR)
    
    if not video_folders:
        print(f"❌ 在 {KEYFRAMES_BASE_DIR} 下未找到任何视频文件夹，程序退出")
        return
    
    # 3. 匹配报告和文件夹（按顺序）
    matched_reports = match_reports_to_folders(all_reports, video_folders)
    
    if not matched_reports:
        print("❌ 未能匹配任何报告与文件夹，程序退出")
        return
    
    print(f"✅ 成功匹配 {len(matched_reports)} 个视频报告与文件夹")
    
    # 4. 加载模型（只需加载一次）
    print(">>> 正在加载Qwen3-VL模型...")
    t0 = time.time()
    device = torch.device(DEVICE_ID)
    
    processor = AutoProcessor.from_pretrained(LLM_MODEL_PATH)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        LLM_MODEL_PATH, device_map={"": DEVICE_ID}, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).eval()
    
    t1 = time.time()
    print(f"✅ 模型加载完成，耗时 {t1 - t0:.2f} 秒")
    
    # 5. 创建JSONL输出文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_filename = f"video_analysis_results_{timestamp}.jsonl"
    jsonl_path = os.path.join(OUTPUT_BASE_DIR, jsonl_filename)
    
    print(f"\n📁 创建JSONL输出文件: {jsonl_path}")
    
    # 6. 批量处理视频
    task_name = "视频描述"  # 固定为视频描述任务
    print(f">>> 开始批量处理 {len(matched_reports)} 个视频，任务类型: {task_name}")
    
    results = []
    successful_count = 0
    failed_count = 0
    
    # 打开JSONL文件进行写入
    with open(jsonl_path, 'w', encoding='utf-8') as jsonl_file:
        # 使用tqdm显示进度条
        for i, report in enumerate(tqdm(matched_reports, desc="处理进度")):
            # 处理单个视频，并将结果写入JSONL文件
            result = process_single_video(
                report=report,
                model=model,
                processor=processor,
                device=device,
                task_name=task_name,
                jsonl_writer=jsonl_file  # 传递文件写入器
            )
            
            if result:
                results.append(result)
                if result['status'] == 'success':
                    successful_count += 1
                else:
                    failed_count += 1
            else:
                failed_count += 1
            
            # 可选：添加延迟以避免GPU过载
            time.sleep(0.5)
            
            # 每处理10个视频刷新一次文件缓冲区
            if (i + 1) % 10 == 0:
                jsonl_file.flush()
    
    print(f"\n💾 所有结果已保存到JSONL文件: {jsonl_path}")
    
    # 7. 生成处理报告
    print("\n" + "=" * 60)
    print("批量处理完成！")
    print("=" * 60)
    print(f"总视频数: {len(matched_reports)}")
    print(f"成功处理: {successful_count}")
    print(f"处理失败: {failed_count}")
    print(f"JSONL文件: {jsonl_path}")
    
    # 保存批量处理摘要
    summary = {
        "process_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_videos": len(matched_reports),
        "successful": successful_count,
        "failed": failed_count,
        "task_type": task_name,
        "jsonl_file": jsonl_path,
        "file_format": "JSONL (每行一个完整的JSON对象)",
        "results_summary": {
            "success_count": successful_count,
            "error_count": failed_count
        }
    }
    
    summary_path = os.path.join(OUTPUT_BASE_DIR, f"batch_process_summary_{timestamp}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 处理摘要已保存到: {summary_path}")
    
    # 8. 验证JSONL文件
    print(f"\n🔍 验证JSONL文件内容...")
    line_count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
    
    print(f"✅ JSONL文件包含 {line_count} 行记录")
    
    # 打印失败的视频
    if failed_count > 0:
        print("\n❌ 处理失败的视频:")
        for result in results:
            if result['status'] == 'error':
                print(f"  视频{result['report_id']} ({result['video_id']}): {result.get('error', '未知错误')}")
    
    # 9. 使用说明
    print("\n📋 JSONL文件使用说明:")
    print("=" * 50)
    print("1. 查看文件内容:")
    print(f"   head -n 5 {jsonl_path}")
    print("\n2. 统计成功/失败数量:")
    print(f"   grep -c '\"status\":\"success\"' {jsonl_path}")
    print(f"   grep -c '\"status\":\"error\"' {jsonl_path}")
    print("\n3. 提取特定视频结果:")
    print(f"   grep '\"video_id\":\"video7020\"' {jsonl_path}")
    print("\n4. 转换为普通JSON数组:")
    print(f"   sed '1s/^/[/; $!s/$/,/; $s/$/]/' {jsonl_path} > output.json")
    print("=" * 50)


if __name__ == "__main__":
    main()