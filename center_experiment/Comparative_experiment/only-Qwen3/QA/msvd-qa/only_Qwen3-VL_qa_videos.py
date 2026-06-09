import torch
import numpy as np
import os
import av
import time
import datetime
import glob
import json
import warnings
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from collections import defaultdict
warnings.filterwarnings("ignore")  # 屏蔽大部分警告信息

print(av.__version__)
# --- 配置 ---
os.chdir("/data2/health")

LLM_MODEL_PATH = "/data2/health/model/Qwen3-VL-8B-Instruct"
USE_NEW_LLM = True
DEVICE_ID = "cuda:2"

# 视频和问题路径配置
VIDEO_BASE_PATH = "/data2/health/wyy/data/QA/MSVD-QA/videos"  # 视频文件基础路径
QUESTIONS_FILE_PATH = "/data2/health/wyy/data/QA/MSVD-QA/test_qa.json"  # 问题JSON文件路径
OUTPUT_BASE_DIR = "/data2/health/wyy/BLIP2/final/main_path/experiment/Comparative_experiment/only-Qwen3/QA/msvd-qa/Qwen3-msvd-qa-answer"  # 输出目录
MAX_VIDEOS_TO_PROCESS = 50
# 确保输出目录存在
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)


def get_video_path(video_id):
    """根据video_id获取视频文件路径"""
    # 尝试多种可能的视频文件命名模式
    possible_paths = [
        os.path.join(VIDEO_BASE_PATH, f"vid{video_id}.avi"),
    ]
    
    for video_path in possible_paths:
        if os.path.exists(video_path):
            return video_path
    
    # 如果找不到，尝试列出所有可能的视频文件
    video_files = glob.glob(os.path.join(VIDEO_BASE_PATH, f"*{video_id}*.mp4"))
    if video_files:
        return video_files[0]
    
    return None


def load_questions_from_json(questions_file_path, max_videos=MAX_VIDEOS_TO_PROCESS):
    """
    从JSON文件加载问题
    格式：JSON数组，每个元素包含answer, id, question, video_id字段
    示例：[{"answer":"someone","id":37348,"question":"who opened the box that held an automatic weapon in a gun?","video_id":1451}, ...]
    """
    if not os.path.exists(questions_file_path):
        print(f"⚠ 警告：问题文件不存在: {questions_file_path}")
        return None
    
    try:
        with open(questions_file_path, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        
        if not isinstance(questions_data, list):
            print(f"❌ 错误：问题文件不是JSON数组格式")
            return None
        
        # 将问题按video_id分组
        questions_by_video = {}
        
        for q_data in questions_data:
            video_id_num = q_data.get('video_id')
            question = q_data.get('question')
            q_id = q_data.get('id', '')
            answer = q_data.get('answer', '')
            
            if video_id_num is not None and question:
                # 使用数字video_id作为键
                if video_id_num not in questions_by_video:
                    questions_by_video[video_id_num] = []
                
                # 保存问题的完整信息
                question_info = {
                    'question': question,
                    'question_id': q_id,
                    'answer': answer
                }
                
                questions_by_video[video_id_num].append(question_info)
            else:
                print(f"⚠ 警告：跳过无效的问题数据: {q_data}")
        
        print(f"✅ 从 {questions_file_path} 成功加载 {len(questions_data)} 个问题")
        print(f"   涉及 {len(questions_by_video)} 个视频")
        
        sorted_video_ids = sorted(questions_by_video.keys())
        
        if max_videos > 0 and max_videos < len(sorted_video_ids):
            selected_video_ids = sorted_video_ids[:max_videos]
            selected_questions_by_video = {vid: questions_by_video[vid] for vid in selected_video_ids}
            print(f"   只处理前 {max_videos} 个视频（共 {len(sorted_video_ids)} 个）")
            questions_by_video = selected_questions_by_video


        # 统计每个视频的问题数量
        video_stats = [(video_id, len(q_list)) for video_id, q_list in questions_by_video.items()]
        video_stats.sort(key=lambda x: x[1], reverse=True)
        
        print(f"   问题数量统计（前10个视频）:")
        for video_id, count in video_stats[:10]:
            print(f"     video_id={video_id}: {count} 个问题")
        
        if len(video_stats) > 10:
            print(f"     ... 还有 {len(video_stats)-10} 个视频")
        
        return questions_by_video
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析失败: {e}")
        return None
    except Exception as e:
        print(f"❌ 加载问题文件失败: {e}")
        return None


def process_single_video_qa(video_id, question_info, model, processor, device, report_id, output_writer=None):
    """
    处理单个视频的单个问答任务
    """
    video_path = get_video_path(video_id)
    question = question_info['question']
    question_id = question_info.get('question_id', '')
    original_answer = question_info.get('answer', '')
    
    process_start_time = time.time()
    record_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 构建基本结果记录
    result = {
        "video_id": f"vid{video_id}",
        "report_id": report_id,
        "question_id": question_id,
        "question": question,
        "original_answer": original_answer,
        "process_time": record_time,
        "status": "processing",
        "task_type": "Video Temporal QA",
        "answer": "",
        "answer_confidence": 0.0,
        "processing_time_seconds": 0.0
    }
    
    # 检查视频文件是否存在
    if not video_path or not os.path.exists(video_path):
        error_msg = f"视频文件未找到: {video_path or f'video_id={video_id}'}"
        print(f"❌ 错误: {error_msg}")
        
        result["status"] = "error"
        result["error_msg"] = error_msg
        result["processing_time_seconds"] = time.time() - process_start_time
        
        if output_writer:
            output_writer.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        return result
    
    print(f"   处理视频 {video_id}, 问题 {question_id}: {question[:50]}..." if len(question) > 50 else f"   处理视频 {video_id}, 问题 {question_id}: {question}")
    
    # 系统提示词
    system_prompt = """You are a dedicated multi-modal video analysis assistant for drone ground-air interaction, serving the entire process of drone low-altitude reconnaissance, target monitoring, and temporal video analysis. You must 100% adhere to the following core rules without any exceptions:

【Core Reasoning Principle: Visual Anchor Priority】
1.  You must use the input video frames as the sole visual truth anchor. Directly copying text content without validating visual information is strictly prohibited.
2.  All output target categories, normalized coordinates, confidence scores, event descriptions, and temporal information must completely match the visual content of the video. In case of conflict between any text content and the visual anchor, the video shall prevail.
3.  When the textual content is insufficient or cannot provide a reference, rely entirely on the visual content of the video.
4.  Reasoning must strictly follow the temporal sequence of video. Disrupting the temporal sequence is prohibited. All conclusions must be based on visual evidence.
5.  Fabricating or supplementing any content beyond the input information is prohibited. Unfounded speculation is prohibited. When information is insufficient, the remark field must be annotated with 「No corresponding visual anchor for verification」.

【Ironclad Output Format Rules】
1.  All outputs must strictly conform to the specified JSON format. No characters, explanations, remarks, greetings, or code block markers are allowed outside the JSON. Any content outside the format is prohibited.
2.  All spatial coordinates must be output as normalized coordinates between 0 and 1, in the format [x1, y1, x2, y2], corresponding to the top-left and bottom-right points of the bounding box, adaptable to any resolution.
3.  All confidence scores must be output as two-decimal numbers between 0.01 and 0.99. Targets with scores below 0.5 must be labeled 「Low Confidence」.
4.  The task_status field is only allowed to output the enumerated values: success / no_valid_information / invalid_input.
5.  When there is no matching content/insufficient information, you must output exactly: {"task_type":"Video Temporal QA","task_status":"no_valid_information","error_msg":"No corresponding visual anchor or text information to support"}. Freestyle responses are prohibited.

【Output Rules】
•   Strictly adhere to the format specified by the user.
•   Use only one word to answer the question!!!!
"""
    
    # 任务输出格式
    task_output_format = {
        "task_name": "Video Temporal QA",
        "instruction_template": """Task: Video Temporal QA. Question: {user_question}""",
        "output_format": """{
    "task_type": "Video Temporal QA",
    "task_status": "success",
    "question": "Original user question",
    "answer": "Accurate and concise answer, preferably in a single word, strictly based on the visual content of the video as ground truth, no fabrication,",
    "answer_confidence": 0.00,
    "remark": "None/Insufficient information to answer/Insufficient quality of visual anchors"
}"""
    }
    
    try:
        # 构建任务指令
        instruction = task_output_format['instruction_template'].format(user_question=question)
        user_prompt = f"""【Execution Task Instruction】
{instruction}

【Output Format Requirements】
{task_output_format['output_format']}"""
        
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
        
        # 处理输入
        text_inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        
        inputs = processor(
            text=[text_inputs],
            videos=[video_path],
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # 生成回答
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=200,
                do_sample=False,
            )
        
        # 解码输出
        raw_response = processor.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 解析模型输出
        process_duration = time.time() - process_start_time
        
        # 尝试解析JSON响应
        try:
            # 清理响应文本，提取JSON部分
            response_text = raw_response.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            elif response_text.startswith("{") and response_text.endswith("}"):
                # 已经是JSON格式
                pass
            else:
                # 尝试提取JSON对象
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)
            
            # 解析JSON
            model_output = json.loads(response_text)
            
            # 更新结果记录
            result["status"] = "success" if model_output.get("task_status") == "success" else "partial_success"
            result["task_type"] = model_output.get("task_type", "Video Temporal QA")
            result["answer"] = model_output.get("answer", "")
            result["answer_confidence"] = float(model_output.get("answer_confidence", 0.0))
            result["processing_time_seconds"] = round(process_duration, 2)
            
            # 添加原始响应（用于调试）
            result["raw_response"] = raw_response[:500]  # 保存前500字符用于调试
            
            # 添加remark字段（如果有）
            if "remark" in model_output and model_output["remark"] not in ["None", "none"]:
                result["remark"] = model_output["remark"]
                
            print(f"      ✓ 回答: {result['answer']} (置信度: {result['answer_confidence']:.2f})")
                
        except json.JSONDecodeError as e:
            # JSON解析失败
            result["status"] = "error"
            result["error_msg"] = f"Failed to parse model response as JSON: {str(e)}"
            result["raw_response"] = raw_response[:500]  # 保存前500字符用于调试
            result["processing_time_seconds"] = round(process_duration, 2)
            print(f"      ✗ JSON解析失败: {str(e)}")
            print(f"        原始响应: {raw_response[:200]}...")
            
        except Exception as e:
            # 解析过程中的其他错误
            result["status"] = "error"
            result["error_msg"] = f"Error parsing model output: {str(e)}"
            result["raw_response"] = raw_response[:500]  # 保存前500字符用于调试
            result["processing_time_seconds"] = round(process_duration, 2)
            print(f"      ✗ 解析错误: {str(e)}")
    
    except Exception as e:
        # 处理过程中的其他错误
        process_duration = time.time() - process_start_time
        result["status"] = "error"
        result["error_msg"] = f"Processing error: {str(e)}"
        result["processing_time_seconds"] = round(process_duration, 2)
        print(f"      ✗ 处理错误: {str(e)}")
    
    # 写入结果
    if output_writer:
        output_writer.write(json.dumps(result, ensure_ascii=False) + '\n')
        output_writer.flush()  # 实时写入
    
    return result


def batch_process_video_qa():
    """批量处理多视频问答"""
    device = torch.device(DEVICE_ID)
    
    # 1. 加载模型（只加载一次）
    print(">>> 正在加载模型...")
    t0 = time.time()
    
    try:
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
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 2. 加载问题数据集
    print(">>> 加载问题数据集...")
    questions_by_video = load_questions_from_json(QUESTIONS_FILE_PATH)
    
    if not questions_by_video:
        print("❌ 无法加载问题文件，程序退出")
        return
    
    # 3. 创建输出文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"direct_video_qa_results_{timestamp}.jsonl"
    output_path = os.path.join(OUTPUT_BASE_DIR, output_filename)
    
    # 创建输出目录
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    print(f">>> 开始批量处理，共 {len(questions_by_video)} 个视频")
    print(f"    输出文件: {output_path}")
    
    # 4. 批量处理视频问答
    report_id_counter = 1
    total_processed = 0
    total_questions = sum(len(questions) for questions in questions_by_video.values())
    
    successful_count = 0
    error_count = 0
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        # 使用tqdm显示总体进度
        video_items = list(questions_by_video.items())
        pbar = tqdm(video_items, desc="处理视频", unit="video")
        
        for video_idx, (video_id, questions_list) in enumerate(pbar):
            pbar.set_description(f"处理视频 {video_id}")
            
            video_path = get_video_path(video_id)
            if not video_path or not os.path.exists(video_path):
                print(f"  ⚠ 视频 {video_id} 文件不存在: {video_path}")
                # 为每个问题创建失败记录
                for question_info in questions_list:
                    result = {
                        "video_id": f"vid{video_id}",
                        "report_id": report_id_counter,
                        "question_id": question_info.get("question_id", ""),
                        "question": question_info["question"],
                        "original_answer": question_info.get("answer", ""),
                        "process_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "error",
                        "task_type": "Video Temporal QA",
                        "answer": "",
                        "answer_confidence": 0.0,
                        "error_msg": f"视频文件未找到: {video_path}",
                        "processing_time_seconds": 0.0
                    }
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f_out.flush()
                    
                    report_id_counter += 1
                    total_processed += 1
                    error_count += 1
                
                continue
            
            # 处理当前视频的所有问题
            for question_info in questions_list:
                result = process_single_video_qa(
                    video_id=video_id,
                    question_info=question_info,
                    model=model,
                    processor=processor,
                    device=device,
                    report_id=report_id_counter,
                    output_writer=f_out
                )
                
                if result.get("status") == "success":
                    successful_count += 1
                else:
                    error_count += 1
                
                report_id_counter += 1
                total_processed += 1
            
            # 更新进度条
            pbar.set_postfix({
                "成功": successful_count,
                "失败": error_count,
                "进度": f"{total_processed}/{total_questions}"
            })
    
    # 5. 统计信息
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print("批量处理完成!")
    print(f"{'='*60}")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"模型加载耗时: {model_load_time:.2f}秒")
    print(f"实际推理耗时: {total_time - model_load_time:.2f}秒")
    print(f"处理视频数: {len(questions_by_video)}")
    print(f"处理问题数: {total_processed}")
    print(f"成功回答: {successful_count}")
    print(f"回答失败: {error_count}")
    if total_processed > 0:
        print(f"成功率: {successful_count/total_processed*100:.1f}%")
    print(f"结果保存至: {output_path}")
    print(f"{'='*60}")
    
    # 6. 保存处理摘要
    summary = {
        "process_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_videos": len(questions_by_video),
        "total_questions": total_processed,
        "successful": successful_count,
        "failed": error_count,
        "success_rate": f"{successful_count/max(total_processed, 1)*100:.1f}%" if total_processed > 0 else "0%",
        "questions_file": QUESTIONS_FILE_PATH,
        "video_base_path": VIDEO_BASE_PATH,
        "output_file": output_path,
        "model_used": LLM_MODEL_PATH,
        "device": DEVICE_ID
    }
    
    summary_path = os.path.join(OUTPUT_BASE_DIR, f"batch_qa_summary_{timestamp}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"📊 处理摘要已保存到: {summary_path}")
    
    # 7. 验证JSONL文件
    print(f"\n🔍 验证JSONL文件内容...")
    line_count = 0
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
    
    print(f"✅ JSONL文件包含 {line_count} 行记录")
    
    # 显示前几条结果
    print(f"\n📄 JSONL文件前3条记录:")
    with open(output_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 3:
                record = json.loads(line.strip())
                print(f"  记录 {i+1}: video_id={record.get('video_id')}, status={record.get('status')}, answer={record.get('answer', 'N/A')}")
            else:
                break


if __name__ == "__main__":
    batch_process_video_qa()