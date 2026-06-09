import torch
import numpy as np
import os
import av
import time
import datetime
import glob
import json
from tqdm import tqdm
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import warnings
warnings.filterwarnings("ignore")

print(av.__version__)

# --- 配置 ---
os.chdir("/data3/health")

LLM_MODEL_PATH = "/data3/health/model/Qwen3-VL-8B-Instruct"
DEVICE_ID = "cuda:4"

# MSR-VTT测试集视频路径
MSR_VTT_TEST_PATH = "/data3/health/wyy/data/summary/MSR-VTT/video"  # 请确认这是包含所有测试视频的目录
# 输出目录
OUTPUT_DIR = "/data3/health/wyy/BLIP2/experiment/Comparative_experiment/only-Qwen3/summary/msrvtt/Qwen3_video_descriptions"
# 修改输出文件命名
OUTPUT_JSONL_PATH = os.path.join(OUTPUT_DIR, f"msr_vtt_descriptions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_video(video_path, processor, model, device, task_config):
    """处理单个视频并生成描述"""
    
    # 1. 构建消息
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
    
    # 获取视频文件名（不含扩展名）用于输出
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 构建完整的user prompt
    user_prompt = f"""【Video Information】
Video file: {video_name}
Video path: {video_path}

【Execution Task Instruction】
{task_config['instruction']}

【Output Format Requirements】
{task_config['output_format']}"""
    
    # 构建消息列表
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": user_prompt}
            ]
        }
    ]
    
    try:
        # 第一步：应用聊天模板生成文本输入
        text_inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        
        # 第二步：处理视频和文本
        inputs = processor(
            text=[text_inputs],
            videos=[video_path],
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # 生成结果
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024,  # 增加token数量以容纳完整JSON
                do_sample=False,
            )
        
        # 解码输出
        response = processor.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return response, video_name
        
    except Exception as e:
        error_msg = f"Error processing video {video_path}: {str(e)}"
        print(f"❌ {error_msg}")
        # 返回错误信息
        error_response = {
            "task_type": "Video Temporal Description",
            "task_status": "invalid_input",
            "error_msg": error_msg,
            "video_file": video_name,
            "video_path": video_path
        }
        return json.dumps(error_response), video_name

def save_result(response_text, video_name, output_dir):
    """保存处理结果到JSON文件"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"video_description_{video_name}_{timestamp}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        # 尝试解析JSON响应
        response_json = json.loads(response_text.strip())
        
        # 添加元数据
        response_json["metadata"] = {
            "video_name": video_name,
            "processing_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_used": "Qwen3-VL-8B-Instruct"
        }
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(response_json, f, ensure_ascii=False, indent=2)
        
        return output_path, True
        
    except json.JSONDecodeError:
        # 如果响应不是有效的JSON，保存原始文本
        error_filename = f"error_{video_name}_{timestamp}.txt"
        error_path = os.path.join(output_dir, error_filename)
        
        with open(error_path, 'w', encoding='utf-8') as f:
            f.write(f"Video: {video_name}\n")
            f.write(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n--- Model Response (Invalid JSON) ---\n")
            f.write(response_text)
        
        return error_path, False

def process_all_videos():
    """处理MSR-VTT测试集中的所有视频"""
    
    device = torch.device(DEVICE_ID)
    
    # 1. 加载模型（只加载一次）
    print(">>> 正在加载Qwen3-VL模型...")
    t0 = time.time()
    
    processor = AutoProcessor.from_pretrained(LLM_MODEL_PATH)
    print(f"Tokenizer的最大模型长度 (model_max_length): {processor.tokenizer.model_max_length}")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        LLM_MODEL_PATH, 
        device_map={"": DEVICE_ID}, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    ).eval()
    
    model_load_time = time.time() - t0
    print(f"✅ 模型加载完成，耗时: {model_load_time:.2f}秒")
    
    # 2. 定义视频描述任务配置
    task_config = {
        "instruction": "**Task: Video Captioning.** Look at the provided keyframe images. Generate ONE concise English sentence that describes the main activity or event happening in the video. Do not describe each keyframe separately. Do not mention time segments, durations, or technical details. Output ONLY the single sentence.",
        "output_format": "A person is riding a bicycle in a park."  # 这里直接放一个示例，告诉模型输出应该是这样的纯文本句子
}

#     task_config = {
#         "instruction": "Execute task: Video temporal description. Requirements: 1. Analyze the entire video content to generate a comprehensive video description; 2. Describe in chronological segments, each time period should capture key events or scene changes; 3. Provide a concise but complete summary of the video content, focusing on visual elements, actions, and temporal progression.",
#         "output_format": """{
#     "task_type": "Video Temporal Description",
#     "task_status": "success",
#     "video_base_info": {
#         "video_file": "video file name",
#         "total_duration_estimated": "estimated video duration in seconds",
#         "scene_type": "main scene type detected",
#         "primary_subjects": "main subjects/objects in the video",
#         "temporal_progression": "overview of temporal progression"
#     },
#     "timeline_description": [
#         {
#             "time_period": "X.0s-Y.0s",
#             "key_events": "key events or scene changes in this period",
#             "primary_visual_elements": "main visual elements observed",
#             "notable_actions": "notable actions or movements"
#         }
#     ],
#     "full_video_summary": "Comprehensive video summary, capturing the essence of the entire video in 80-120 words",
#     "analysis_confidence": 0.00,
#     "remark": "None/Limitations in analysis/Visual quality notes"
# }"""
#     }
    
   # 3. 【关键修改】从JSON文件加载测试集视频列表
    test_set_json_path = "/data3/health/wyy/data/summary/MSR-VTT/msrvtt_test_1k.json"  # 请将此路径替换为实际JSON文件路径
    print(f"📂 正在加载测试集文件: {test_set_json_path}")
    
    try:
        with open(test_set_json_path, 'r', encoding='utf-8') as f:
            test_set_data = json.load(f)
        
        # 提取测试集视频文件名
        test_video_files = []
        for item in test_set_data:
            if "video" in item:
                video_filename = item["video"]
                video_path = os.path.join(MSR_VTT_TEST_PATH, video_filename)
                if os.path.exists(video_path):
                    test_video_files.append(video_path)
                else:
                    print(f"⚠️ 警告：视频文件不存在，已跳过: {video_path}")
        
        if not test_video_files:
            print(f"❌ 错误：在测试集JSON中未找到任何有效的视频文件路径。")
            print(f"请确认: 1) JSON文件路径正确; 2) 'video'字段存在; 3) 视频文件存在于 {MSR_VTT_TEST_PATH}")
            return
        
        test_video_files.sort()  # 按文件名排序
        print(f"📁 从测试集加载了 {len(test_video_files)} 个视频文件")
        
    except FileNotFoundError:
        print(f"❌ 错误：找不到测试集JSON文件: {test_set_json_path}")
        return
    except json.JSONDecodeError:
        print(f"❌ 错误：测试集JSON文件格式无效: {test_set_json_path}")
        return
    except Exception as e:
        print(f"❌ 加载测试集时发生未知错误: {e}")
        return
    
    # 4. 创建JSONL文件并写入处理元数据
    os.makedirs(os.path.dirname(OUTPUT_JSONL_PATH), exist_ok=True)
    
    # 5. 处理统计
    success_count = 0
    error_count = 0
    total_start_time = time.time()
    
    # 6. 打开JSONL文件准备写入
    with open(OUTPUT_JSONL_PATH, 'w', encoding='utf-8') as jsonl_file:
        # 首先写入处理元数据
        # metadata = {
        #     "metadata": {
        #         "processing_session": {
        #             "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        #             "total_videos": len(test_video_files),
        #             "model_used": "Qwen3-VL-8B-Instruct",
        #             "device": DEVICE_ID,
        #             "task_type": "Video Temporal Description",
        #             "output_format": "JSONL"
        #         }
        #     }
        # }
        # jsonl_file.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        
        # 7. 遍历处理所有视频
        print(f"\n>>> 开始批量处理 {len(test_video_files)} 个视频")
        print("=" * 60)
        
        for i, video_path in enumerate(tqdm(test_video_files, desc="处理进度", unit="视频"), 1):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # 在进度条下方显示当前处理的视频信息
            tqdm.write(f"\n📹 处理视频 {i}/{len(test_video_files)}: {video_name}")
            tqdm.write(f"📁 文件路径: {video_path}")
            
            # 处理单个视频
            video_start_time = time.time()
            response_text, processed_video_name = process_video(video_path, processor, model, device, task_config)
            video_process_time = time.time() - video_start_time
            
            # 构建结果记录
            result_record = {
                "index": i,
                "video_file": video_name,
                "video_path": video_path,
                "processing_time_seconds": video_process_time,
                "processing_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            try:
                # 尝试解析模型响应
                response_json = json.loads(response_text.strip())
                
                # 合并响应到结果记录
                result_record.update(response_json)
                result_record["parse_success"] = True
                
                # 检查任务状态
                if response_json.get("task_status") == "success":
                    print(f"✅ 处理成功")
                    if "full_video_summary" in response_json:
                        summary = response_json["full_video_summary"]
                        if len(summary) > 120:
                            summary = summary[:117] + "..."
                        print(f"📋 摘要预览: {summary}")
                    success_count += 1
                else:
                    print(f"⚠️ 任务状态: {response_json.get('task_status', 'N/A')}")
                    error_count += 1
                    
            except json.JSONDecodeError:
                # 响应不是有效的JSON
                result_record["parse_success"] = False
                result_record["raw_response"] = response_text
                result_record["error"] = "Invalid JSON response from model"
                print(f"❌ 响应不是有效JSON格式")
                error_count += 1
            except Exception as e:
                # 其他解析错误
                result_record["parse_success"] = False
                result_record["raw_response"] = response_text
                result_record["error"] = str(e)
                print(f"❌ 解析响应时出错: {e}")
                error_count += 1
            
            # 将结果记录写入JSONL文件
            jsonl_file.write(json.dumps(result_record, ensure_ascii=False) + '\n')
            jsonl_file.flush()  # 立即写入磁盘，避免数据丢失
            
            print(f"💾 结果已写入JSONL文件")
            print(f"⏱️ 本视频处理耗时: {video_process_time:.2f}秒")
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 8. 生成处理报告
    total_time = time.time() - total_start_time
    
    # 在JSONL文件末尾添加处理摘要
    with open(OUTPUT_JSONL_PATH, 'a', encoding='utf-8') as jsonl_file:
        summary_record = {
            "processing_summary": {
                "total_videos_processed": len(test_video_files),
                "successfully_processed": success_count,
                "failed_processing": error_count,
                "success_rate": f"{(success_count/len(test_video_files))*100:.1f}%" if test_video_files else "0%",
                "total_processing_time_seconds": total_time,
                "average_time_per_video_seconds": total_time/len(test_video_files) if test_video_files else 0,
                "model_load_time_seconds": model_load_time,
                "start_time": datetime.datetime.fromtimestamp(total_start_time).strftime('%Y-%m-%d %H:%M:%S'),
                "end_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "output_file": OUTPUT_JSONL_PATH
            }
        }
        jsonl_file.write(json.dumps(summary_record, ensure_ascii=False) + '\n')
    
    # 9. 打印最终报告
    print(f"\n{'='*60}")
    print("🎉 所有视频处理完成!")
    print('='*60)
    print(f"📊 处理统计:")
    print(f"   总视频数: {len(test_video_files)}")
    print(f"   成功处理: {success_count}")
    print(f"   处理失败: {error_count}")
    print(f"   成功率: {(success_count/len(test_video_files))*100:.1f}%" if test_video_files else "0%")
    print(f"   总耗时: {total_time:.2f}秒")
    print(f"   平均每个视频: {total_time/len(test_video_files):.2f}秒" if test_video_files else "0秒")
    print(f"💾 所有结果已保存到单个JSONL文件:")
    print(f"   {OUTPUT_JSONL_PATH}")
    print('='*60)
    # print("📋 JSONL文件结构说明:")
    # print("   第1行: 处理会话的元数据")
    # print(f"   第2-{len(test_video_files)+1}行: 每个视频的处理结果")
    # print(f"   第{len(test_video_files)+2}行: 处理摘要统计")
    # print('='*60)

if __name__ == "__main__":
    # 确认开始处理
    print("🚀 MSR-VTT测试集视频批量处理程序")
    print(f"📁 视频目录: {MSR_VTT_TEST_PATH}")
    print(f"💾 输出目录: {OUTPUT_DIR}")
    print(f"🤖 使用模型: {LLM_MODEL_PATH}")
    print(f"⚡ 使用设备: {DEVICE_ID}")
    
    process_all_videos()