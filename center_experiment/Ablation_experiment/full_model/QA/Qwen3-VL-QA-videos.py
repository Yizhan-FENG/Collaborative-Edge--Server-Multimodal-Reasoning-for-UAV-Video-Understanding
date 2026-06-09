import torch
import transformers
import os
import re
import time
import json
import datetime
import glob
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# --- 配置 ---
os.chdir("/data3/health")

# 模型路径
LLM_MODEL_PATH = "/data3/health/model/Qwen3-VL-8B-Instruct"
DEVICE_ID = "cuda:3"

# 多视频输入路径
# 结构化文本文件（包含所有视频的报告）
MULTI_VIDEO_REPORTS_PATH = "/data3/health/wyy/BLIP2/final/main_path/edge_output_videos/1-10video/structured_reports1/videos_analysis_results_human_readable_20260507_185741.txt"
# 关键帧图片的基础目录（每个视频有独立的子文件夹）
KEYFRAMES_BASE_DIR = "/data3/health/wyy/BLIP2/experiment/data/msvd-qa/keyframes"
# 问题文件路径（JSON格式，包含answer, id, question, video_id字段）
QUESTIONS_FILE_PATH = "/data3/health/wyy/data/QA/MSVD-QA/test_qa.json"
# 输出目录
OUTPUT_BASE_DIR = "/data3/health/wyy/BLIP2/final/main_path/server_output_videos/QA/msvd-qa"

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
        if os.path.isdir(item_path) and item.startswith("vid"):
            video_folders.append(item_path)
    
    # 按文件夹名称中的数字部分排序
    def extract_number(folder_path):
        folder_name = os.path.basename(folder_path)
        # 匹配"video"后面的数字部分
        match = re.search(r'vid(\d+)', folder_name)
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


def load_questions_from_json(questions_file_path):
    """
    从JSON文件加载问题
    格式：JSON数组，每个元素包含answer, id, question, video_id字段
    示例：[{"answer":"someone","id":37348,"question":"who opened the box that held an automatic weapon in a gun?","video_id":1451}, ...]
    """
    if not os.path.exists(questions_file_path):
        print(f"⚠ 警告：问题文件不存在: {questions_file_path}")
        print("  将使用默认问题")
        return None
    
    try:
        with open(questions_file_path, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        
        if not isinstance(questions_data, list):
            print(f"❌ 错误：问题文件不是JSON数组格式")
            return None
        
        # 将问题按video_id分组，并将数字video_id转换为字符串格式（如1451 -> "vid1451"）
        questions_by_video = {}
        
        for q_data in questions_data:
            video_id_num = q_data.get('video_id')
            question = q_data.get('question')
            q_id = q_data.get('id', '')
            
            if video_id_num is not None and question:
                # 将数字video_id转换为字符串格式，与文件夹名匹配
                video_id_str = f"vid{video_id_num}"
                
                if video_id_str not in questions_by_video:
                    questions_by_video[video_id_str] = []
                
                # 保存问题的完整信息
                question_info = {
                    'question': question,
                    'question_id': q_id
                }
                
                questions_by_video[video_id_str].append(question_info)
            else:
                print(f"⚠ 警告：跳过无效的问题数据: {q_data}")
        
        print(f"✅ 从 {questions_file_path} 成功加载 {len(questions_data)} 个问题")
        print(f"   涉及 {len(questions_by_video)} 个视频")
        
        # 统计每个视频的问题数量
        for video_id, q_list in list(questions_by_video.items())[:10]:  # 只显示前10个视频
            print(f"     {video_id}: {len(q_list)} 个问题")
        
        if len(questions_by_video) > 10:
            print(f"     ... 还有 {len(questions_by_video)-10} 个视频")
        
        return questions_by_video
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析失败: {e}")
        return None
    except Exception as e:
        print(f"❌ 加载问题文件失败: {e}")
        return None


def match_reports_to_folders(reports, video_folders):
    """
    将报告与视频文件夹进行匹配
    由于文件夹命名不是从0开始，我们按顺序匹配
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


def process_single_video_qa(report, question_info, model, processor, device, jsonl_writer=None):
    """
    处理单个视频的问答任务
    """
    folder_path = report['folder_path']
    video_id = report['video_id']
    report_id = report['report_id']
    question = question_info['question']
    question_id = question_info.get('question_id', '')
    
    print(f"\n{'='*60}")
    print(f"开始处理视频 {report_id}: {video_id}")
    print(f"问题ID: {question_id}")
    print(f"问题: {question}")
    print(f"关键帧文件夹: {folder_path}")
    print(f"{'='*60}")
    
    record_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. 读取关键帧图像
    image_inputs = load_keyframe_images(folder_path)
    
    if not image_inputs:
        print(f"❌ 错误：视频 {video_id} 没有可用的关键帧图片，跳过处理")
        error_result = {
            'video_id': video_id,
            'report_id': report_id,
            'question_id': question_id,
            'question': question,
            'process_time': record_time,
            'status': 'error',
            'error': '无关键帧图片',
            'task_status': 'no_valid_information',
            'error_msg': '无对应视觉锚点与文本信息支撑'
        }
        
        if jsonl_writer:
            jsonl_writer.write(json.dumps(error_result, ensure_ascii=False) + '\n')
        
        return error_result
    
    print(f"✅ 从文件夹读取到 {len(image_inputs)} 张关键帧图片")
    
    # 2. 构建system prompt (保持不变)
    system_prompt = """You are a dedicated multi-modal video analysis assistant for drone ground-air interaction, serving the entire process of drone low-altitude reconnaissance, target monitoring, and temporal video analysis. You must 100% adhere to the following core rules without any exceptions:

【Core Reasoning Principle: Visual Anchor Priority】
1.  You must use the input keyframe image as the sole visual truth anchor to verify, correct, and refine the accompanying BLIP2 structured text and YOLO pre-detection results. Directly copying text content without validating visual information is strictly prohibited.
2.  All output target categories, normalized coordinates, confidence scores, event descriptions, and temporal information must completely match the visual content of the corresponding timestamped keyframe. In case of conflict between text content and the visual anchor, the keyframe image shall prevail.
3.  When the textual content of key frames is insufficient or cannot provide a reference, rely entirely on the visual anchors of the key frames.
4.  Reasoning must strictly follow the ascending chronological order of keyframes. Disrupting the temporal sequence is prohibited. All conclusions must be annotated with the corresponding frame ID and timestamp to ensure traceability.
5.  Fabricating or supplementing any content beyond the input information is prohibited. Unfounded speculation is prohibited. When information is insufficient, the remark field must be annotated with 「No corresponding visual anchor for verification」.

【Ironclad Output Format Rules】
1.  All outputs must strictly conform to the specified JSON format. No characters, explanations, remarks, greetings, or code block markers are allowed outside the JSON. Any content outside the format is prohibited.
2.  All spatial coordinates must be output as normalized coordinates between 0 and 1, in the format [x1, y1, x2, y2], corresponding to the top-left and bottom-right points of the bounding box, adaptable to any resolution.
3.  All confidence scores must be output as two-decimal numbers between 0.01 and 0.99. Targets with scores below 0.5 must be labeled 「Low Confidence」.
4.  The task_status field is only allowed to output the enumerated values: success / no_valid_information / invalid_input.
5.  When there is no matching content/insufficient information, you must output exactly: {"task_type":"[current_task]","task_status":"no_valid_information","error_msg":"No corresponding visual anchor or text information to support"}. Freestyle responses are prohibited.

【Fixed Rules for Boundary Scenarios】
•   For blurred, occluded, or invalid keyframe images: Use the corresponding frame's remark information as reference. In the output, annotate with 「Visual anchor quality insufficient, result for reference only」.
•   Conflict between text and visual anchor: The visual anchor prevails. Annotate in the remark field with 「BLIP2 text inconsistent with visual anchor, corrected based on visual content」.
•   Unsupported command: Only output {"task_type":"unknown","task_status":"invalid_command","support_commands":["video description", "temporal QA", "target detection verification", "image segmentation verification", "event detection alert"]}.

【Output Rules】
•   Strictly adhere to the format specified by the user.
•   Generate the text for the video description in English!!
•   Use only one word to answer the question!!!!

"""

    task_config = {
        "task_name": "Temporal QA",
        "instruction_template": """Task: Video Temporal QA. Requirements: 1. Use the corresponding timestamped key frames as the visual ground truth, verify the text information, and answer the question accurately. 2. The answer must be annotated with supporting anchor frame IDs, timestamps, and visual evidence. 3. Do not fabricate content; questions without corresponding visual anchors must be marked as insufficient information. Question: {user_question}""",
        "output_format": """{
    "task_type": "Video Temporal QA",
    "task_status": "success",
    "question": "Original user question",
    "answer": "Accurate and concise answer, preferably in a single word, strictly based on the visual content of key frames as ground truth, no fabrication,",
    "answer_confidence": 0.00,
    "remark": "None/Insufficient information to answer/Insufficient quality of visual anchors"
}"""
    }

        
    # 使用用户输入的问题填充指令模板
    qa_instruction = task_config["instruction_template"].format(user_question=question)
    
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
{qa_instruction}

【输出格式要求】
{task_config['output_format']}"""
    
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
    
    try:
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
    except Exception as e:
        print(f"❌ 模型输入处理失败: {e}")
        error_result = {
            'video_id': video_id,
            'report_id': report_id,
            'question_id': question_id,
            'question': question,
            'process_time': record_time,
            'status': 'error',
            'error': f'模型输入处理失败: {str(e)}'
        }
        
        if jsonl_writer:
            jsonl_writer.write(json.dumps(error_result, ensure_ascii=False) + '\n')
        
        return error_result
    
    # 6. 生成结果
    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=1024,
                do_sample=False,
            )
        
        # 解码输出
        response = processor.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
    except Exception as e:
        print(f"❌ 模型推理失败: {e}")
        error_result = {
            'video_id': video_id,
            'report_id': report_id,
            'question_id': question_id,
            'question': question,
            'process_time': record_time,
            'status': 'error',
            'error': f'模型推理失败: {str(e)}'
        }
        
        if jsonl_writer:
            jsonl_writer.write(json.dumps(error_result, ensure_ascii=False) + '\n')
        
        return error_result
    
    # 7. 处理输出
    result_dict = {
        'video_id': video_id,
        'report_id': report_id,
        'question_id': question_id,
        'question': question,
        'process_time': record_time,
        'status': 'processing',
        'keyframe_count': len(image_inputs)
    }
    
    try:
        # 尝试解析JSON响应
        response_json = json.loads(response.strip())
        
        # 将模型响应合并到结果字典
        result_dict.update(response_json)
        result_dict['status'] = 'success'
        result_dict['raw_response'] = response  # 保存原始响应
        
        print(f"✅ 视频 {report_id} ({video_id}) 问答处理完成")
        print(f"   问题ID: {question_id}")
        print(f"   问题: {question[:50]}..." if len(question) > 50 else f"   问题: {question}")
        print(f"   答案: {response_json.get('answer', 'N/A')}")
        print(f"   状态: {response_json.get('task_status', 'N/A')}")
        
        # 写入JSONL文件
        if jsonl_writer:
            jsonl_writer.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
        
        return result_dict
    
    except json.JSONDecodeError as e:
        error_msg = f"JSON解码错误: {str(e)}"
        print(f"❌ 错误：视频 {video_id} 的输出不是有效的JSON格式")
        print(f"🤖 模型原始回答前200字符：{response[:200]}...")
        
        result_dict['status'] = 'error'
        result_dict['error'] = error_msg
        result_dict['raw_response'] = response[:500]  # 保存部分原始响应供调试
        result_dict['task_status'] = 'error'
        result_dict['error_msg'] = '模型输出格式错误'
        
        # 写入JSONL文件（即使是错误结果）
        if jsonl_writer:
            jsonl_writer.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
        
        return result_dict
    
    except Exception as e:
        error_msg = f"处理错误: {str(e)}"
        print(f"❌ 处理视频 {video_id} 时发生错误: {e}")
        
        result_dict['status'] = 'error'
        result_dict['error'] = error_msg
        result_dict['task_status'] = 'error'
        result_dict['error_msg'] = '处理过程中发生错误'
        
        # 写入JSONL文件（即使是错误结果）
        if jsonl_writer:
            jsonl_writer.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
        
        return result_dict


def main():
    """主函数：批量处理所有视频的问答任务"""
    print("=" * 60)
    print("Qwen3-VL 多视频批量问答系统 (JSON格式问题)")
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
    
    # 4. 加载问题文件
    print(">>> 正在加载问题文件...")
    questions_by_video = load_questions_from_json(QUESTIONS_FILE_PATH)
    
    if not questions_by_video:
        print("❌ 无法加载问题文件，程序退出")
        return
    
    # 过滤出有问题数据的视频
    videos_with_questions = [report for report in matched_reports if report['video_id'] in questions_by_video]
    
    if not videos_with_questions:
        print("❌ 没有找到匹配问题的视频，请检查问题文件中的video_id与视频文件夹名称")
        print("   视频文件夹名称示例: video1451, video7020 等")
        print("   问题文件中的video_id: 1451, 7020 等")
        return
    
    print(f"✅ 找到 {len(videos_with_questions)} 个有问题的视频")
    
    # 5. 加载模型（只需加载一次）
    print(">>> 正在加载Qwen3-VL模型...")
    t0 = time.time()
    device = torch.device(DEVICE_ID)
    
    try:
        processor = AutoProcessor.from_pretrained(LLM_MODEL_PATH)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            LLM_MODEL_PATH, device_map={"": DEVICE_ID}, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).eval()
        
        t1 = time.time()
        print(f"✅ 模型加载完成，耗时 {t1 - t0:.2f} 秒")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 6. 创建JSONL输出文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_filename = f"video_qa_results_{timestamp}.jsonl"
    jsonl_path = os.path.join(OUTPUT_BASE_DIR, jsonl_filename)
    
    print(f"\n📁 创建JSONL输出文件: {jsonl_path}")
    
    # 7. 批量处理视频问答
    max_videos_to_process = 50
    if len(videos_with_questions) > max_videos_to_process:
        videos_to_process = videos_with_questions[:max_videos_to_process]
        print(f">>> 开始批量处理前 {max_videos_to_process} 个视频（共 {len(videos_with_questions)} 个视频）")
    else:
        videos_to_process = videos_with_questions
        print(f">>> 开始批量处理 {len(videos_with_questions)} 个视频的问答任务")
    
    results = []
    successful_count = 0
    failed_count = 0
    total_questions = 0
    
    # 打开JSONL文件进行写入
    with open(jsonl_path, 'w', encoding='utf-8') as jsonl_file:
        # 使用tqdm显示进度条
        for i, report in enumerate(tqdm(videos_to_process, desc="处理进度")):
            video_id = report['video_id']
            
            # 获取该视频的问题列表
            video_questions = questions_by_video.get(video_id, [])
            
            # 处理该视频的每个问题
            for question_info in video_questions:
                total_questions += 1
                
                # 处理单个视频问答
                result = process_single_video_qa(
                    report=report,
                    question_info=question_info,
                    model=model,
                    processor=processor,
                    device=device,
                    jsonl_writer=jsonl_file
                )
                
                if result:
                    results.append(result)
                    if result.get('status') == 'success' and result.get('task_status') == 'success':
                        successful_count += 1
                    else:
                        failed_count += 1
                else:
                    failed_count += 1
            
            # 可选：添加延迟以避免GPU过载
            time.sleep(0.5)
            
            # 每处理5个视频刷新一次文件缓冲区
            if (i + 1) % 5 == 0:
                jsonl_file.flush()
    
    print(f"\n💾 所有结果已保存到JSONL文件: {jsonl_path}")
    
    # 8. 生成处理报告
    print("\n" + "=" * 60)
    print("批量问答处理完成！")
    print("=" * 60)
    print(f"总视频数: {len(videos_with_questions)}")
    print(f"总问题数: {total_questions}")
    print(f"成功回答: {successful_count}")
    print(f"回答失败: {failed_count}")
    print(f"JSONL文件: {jsonl_path}")
    
    # 保存批量处理摘要
    summary = {
        "process_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_videos": len(videos_with_questions),
        "total_questions": total_questions,
        "successful": successful_count,
        "failed": failed_count,
        "success_rate": f"{successful_count/max(total_questions, 1)*100:.1f}%" if total_questions > 0 else "0%",
        "questions_file": QUESTIONS_FILE_PATH,
        "jsonl_file": jsonl_path,
        "file_format": "JSONL (每行一个完整的问答结果)",
        "results_summary": {
            "success_count": successful_count,
            "error_count": failed_count
        }
    }
    
    summary_path = os.path.join(OUTPUT_BASE_DIR, f"batch_qa_summary_{timestamp}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 处理摘要已保存到: {summary_path}")
    
    # 9. 验证JSONL文件
    print(f"\n🔍 验证JSONL文件内容...")
    line_count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
    
    print(f"✅ JSONL文件包含 {line_count} 行记录")
    
    # # 打印失败的视频问答
    # if failed_count > 0:
    #     print("\n❌ 处理失败的问答:")
    #     error_count = 0
    #     for result in results:
    #         if result.get('status') != 'success' or result.get('task_status') != 'success':
    #             error_count += 1
    #             if error_count <= 10:  # 只显示前10个错误
    #                 error_msg = result.get('error', result.get('error_msg', '未知错误'))
    #                 video_info = f"视频{result.get('report_id', 'N/A')} ({result.get('video_id', 'N/A')})"
    #                 question_id = result.get('question_id', 'N/A')
    #                 print(f"  {video_info} 问题{question_id}: {error_msg}")
        
    #     if error_count > 10:
    #         print(f"  ... 还有 {error_count-10} 个错误未显示")
    
    # # 10. 使用说明
    # print("\n📋 JSONL文件使用说明:")
    # print("=" * 50)
    # print("1. 查看文件内容:")
    # print(f"   head -n 5 {jsonl_path}")
    # print("\n2. 统计成功/失败数量:")
    # print(f"   grep -c '\"status\":\"success\"' {jsonl_path}")
    # print(f"   grep -c '\"status\":\"error\"' {jsonl_path}")
    # print("\n3. 提取特定视频结果:")
    # print(f"   grep '\"video_id\":\"video1451\"' {jsonl_path}")
    # print("\n4. 转换为普通JSON数组:")
    # print(f"   sed '1s/^/[/; $!s/$/,/; $s/$/]/' {jsonl_path} > output.json")
    # print("\n5. 查看所有问题的答案:")
    # print(f"   grep -o '\"answer\":\"[^\"]*\"' {jsonl_path}")
    # print("=" * 50)


if __name__ == "__main__":
    main()