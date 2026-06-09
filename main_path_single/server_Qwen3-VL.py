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
warnings.filterwarnings("ignore") # 屏蔽大部分警告信息

# --- 配置 ---
os.chdir("/data3/health")

LLM_MODEL_PATH = "/data3/health/model/Qwen3-VL-8B-Instruct"
USE_NEW_LLM = True
EDGE_OUTPUT_PATH_TXT = "/data3/health/wyy/BLIP2/final/main_path/edge_output/structured_reports/video_analysis_results_human_readable_20260319_153708.txt"
FOLDER_PATH = "/data3/health/wyy/BLIP2/final/main_path/edge_output/keyframes/video0"
DEVICE_ID = "cuda:5"



def run_once(task_name):
    record_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device = torch.device(DEVICE_ID)
    
    # 1. 加载模型
    print(">>> 正在加载模型...")
    t0 = time.time()

    # 加载BLIP2的视觉部分
    processor = AutoProcessor.from_pretrained(LLM_MODEL_PATH)
    print(f"Tokenizer的最大模型长度 (model_max_length): {processor.tokenizer.model_max_length}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        LLM_MODEL_PATH, device_map={"": DEVICE_ID}, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).eval()

    # 2.读取BLIP2输出的结构化文本
    try:
        with open(EDGE_OUTPUT_PATH_TXT, 'r', encoding='utf-8') as f:
            blip2_content = f.read()
        
        # 提取全局时序信息
        global_info_match = re.search(r'【通用视频全局时序信息】(.*?)【BLIP2全局语义摘要】', blip2_content, re.DOTALL)
        global_info = global_info_match.group(1).strip() if global_info_match else "无法提取全局时序信息"
        
        # 提取BLIP2全局语义摘要
        summary_match = re.search(r'【BLIP2全局语义摘要】(.*?)【关键帧绑定文本列表】', blip2_content, re.DOTALL)
        blip2_summary = summary_match.group(1).strip() if summary_match else "无法提取BLIP2摘要"
        
        # 提取关键帧绑定文本
        frames_text = []
        frame_pattern = r'帧ID:(\w+)\|时间戳:([\d.]+)s(.*?)(?=帧ID:|$)'
        for match in re.finditer(frame_pattern, blip2_content, re.DOTALL):
            frame_id, timestamp, frame_content = match.groups()
            frames_text.append({
                'frame_id': frame_id,
                'timestamp': timestamp,
                'content': frame_content.strip()
            })

        # print("全局信息:", global_info)
        # print("全局语义摘要:", blip2_summary)
        # print("关键帧绑定文本:", frames_text)
        
    except Exception as e:
        print(f"错误：读取BLIP2结构化文本失败: {e}")
        return

    # 3.读取关键帧图像
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(FOLDER_PATH, ext)))

    # 对文件路径进行排序，确保每次输入的顺序一致
    image_paths.sort()

    if not image_paths:
        print(f"错误：在路径 '{FOLDER_PATH}' 下未找到任何图片文件。")
        return
    
    print(f">>> 从文件夹读取到 {len(image_paths)} 张图片。")

     # 打开所有图片
    image_inputs = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            image_inputs.append(img)
        except Exception as e:
            print(f"警告：无法打开图片 {img_path}，错误：{e}")
    
    if not image_inputs:
        print("错误：未能成功加载任何图片。")
        return
    
    # 4.构建system prompt
    system_prompt = """你是无人机地空交互专用多模态视频分析助手，服务于无人机低空侦察、目标监测、时序视频分析全流程，必须100%遵守以下核心规则，无任何例外：
【核心推理准则：视觉锚点优先】
1.  你必须以输入的关键帧图像为**唯一视觉真值锚点**，对配套的BLIP2结构化文本、YOLO预检测结果进行2；
2.  所有输出的目标类别、归一化坐标、置信度、事件描述、时序信息，必须与对应时间戳的关键帧视觉内容完全一致，文本内容与视觉锚点冲突时，以关键帧图像为准；
3.  必须严格按照关键帧的时间升序进行推理，禁止打乱时序，所有结论必须标注对应的帧ID、时间戳，保证可追溯；
4.  禁止编造、补充输入信息以外的任何内容，禁止无依据推测，信息不足时必须在备注字段标注「无对应视觉锚点验证」。

【输出格式铁则】
1.  所有输出必须严格符合指定的JSON格式，JSON外不得有任何字符、解释、备注、寒暄、代码块标记，禁止任何格式外的内容；
2.  所有空间坐标必须输出0-1之间的归一化坐标，格式为[x1,y1,x2,y2]，对应图像左上-右下点位，适配任意分辨率；
3.  所有置信度必须输出0-1之间的两位小数，范围0.01-0.99，低于0.5的目标必须标注「低置信度」；
4.  task_status字段仅允许输出枚举值：success/no_valid_information/invalid_input；
5.  无匹配内容/信息不足时，必须固定输出{"task_type":"[当前任务]","task_status":"no_valid_information","error_msg":"无对应视觉锚点与文本信息支撑"}，禁止自由发挥。

【边界场景固定规则】
- 关键帧图像模糊、遮挡、无效：以对应帧的备注信息为准，输出时标注「视觉锚点质量不足，结果仅供参考」；
- 文本与视觉锚点冲突：以视觉锚点为准，在备注字段标注「BLIP2文本与视觉锚点不一致，已按视觉内容修正」；
- 指令不支持：仅输出{"task_type":"unknown","task_status":"invalid_command","support_commands":["视频描述","时序问答","目标检测校验","图像分割校验","事件检测告警"]}

【输出规则】
- 严格按照user规定的格式；
- 用英文回答
"""


    # 5. 根据任务类型定义不同的任务指令和输出格式
    task_configs = {
        "视频描述": {
            "instruction": "执行任务：视频时序描述。要求：1. 以所有关键帧为视觉真值，校验并修正全局时序文本，生成标准化无人机侦察视频描述；2. 按时序分段描述，每个时间段必须标注对应锚点帧的ID与时间戳；3. 严格符合地空侦察报告规范，禁止冗余修饰。",
            "output_format": """{
    "task_type": "视频时序描述",
    "task_status": "success",
    "video_base_info": {
        "total_duration": "视频总时长",
        "start_time": "采集起始UTC时间",
        "scene_type": "场景类型",
        "total_keyframes": "关键帧总数",
        "total_targets": "全局目标总数"
    },
    "timeline_description": [
        {
            "time_period": "X.0s-Y.0s",
            "anchor_frame_id": "对应锚点帧ID",
            "scene_description": "该时间段的场景与目标描述，以锚点帧视觉内容为准",
            "target_change": "目标数量/状态变化",
            "is_corrected": "是否修正了BLIP2文本的误差",
            "correction_remark": "修正说明，无修正则填“无”"
        }
    ],
    "full_video_summary": "符合侦察规范的完整视频总结，不超过100字",
    "remark": "无/低置信度目标提示/视觉锚点质量提示"
}"""
        },
        "时序问答": {
            "instruction": """执行任务：视频时序问答。要求：1. 以对应时间戳的关键帧为视觉真值，校验文本信息后精准回答问题；2. 答案必须标注支撑结论的锚点帧ID、时间戳、视觉依据；3. 禁止编造内容，无对应视觉锚点的问题必须标注信息不足。问题：[视频中出现了多少辆车？]""",
            "output_format": """{
    "task_type": "视频时序问答",
    "task_status": "success",
    "question": "用户原始问题",
    "answer": "精准简洁的答案，严格以关键帧视觉内容为真值，无编造",
    "anchor_evidence": [
        {
            "frame_id": "支撑结论的锚点帧ID",
            "timestamp": "对应时间戳",
            "visual_evidence": "从锚点帧提取的视觉依据，如“帧F001中2.0s画面左侧车道有3辆汽车，坐标[0.49,0.59,0.53,0.67]”",
            "text_correction": "是否修正了BLIP2/YOLO文本的误差，无则填“无”"
        }
    ],
    "answer_confidence": 0.00,
    "remark": "无/信息不足无法回答/视觉锚点质量不足"
}"""
        },
    }

    # 6.构建user prompt 
    # 全局时序部分
    global_section = f"""【无人机视频全局时序信息】
                        {global_info}
                        {blip2_summary}"""
    # 构建关键帧视觉锚点与绑定文本部分
    frames_section = "【关键帧视觉锚点与绑定文本】\n"
    for i, frame_info in enumerate(frames_text):
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
    

    image_placeholders = [{"type": "image"} for _ in range(len(image_inputs))]
    image_placeholders.append({"type": "text", "text": user_prompt})

    # print("system prompt:", system_prompt)
    # print("user prompt:", user_prompt)


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
        add_generation_prompt=True,  # 添加模型开始生成回答的提示
    )
    
    inputs = processor(
        text=[text_inputs],  # 传入处理后的、带图像占位符的文本
        images=image_inputs,  # 传入图像列表，顺序与content中的占位符顺序对应
        padding=True,
        return_tensors="pt",
    ).to(device)

    # 开始生成
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=1024,
            do_sample=False,
        )

    # 解码输出
    response = processor.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
  # 新增：处理模型输出，保存JSON文件并选择性打印
    try:
        # 尝试解析JSON响应
        response_json = json.loads(response.strip())
        
        # 生成输出文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"video_analysis_{task_name}_{timestamp}.json"
        
        # 确定输出目录（使用与输入文件相同的目录）
        output_dir = os.path.dirname(EDGE_OUTPUT_PATH_TXT)
        output_path = os.path.join(output_dir, output_filename)
        
        # 保存完整JSON到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(response_json, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 完整输出已保存到: {output_path}")
        
        # 根据任务类型选择性打印
        if task_name == "视频描述":
            if "full_video_summary" in response_json:
                print("\n" + "="*50)
                print("📋 视频总结:")
                print("="*50)
                print(response_json["full_video_summary"])
                print("="*50)
            else:
                print("⚠ 警告：输出JSON中未找到'full_video_summary'字段")
                print("🤖 模型原始回答：", response)
        
        elif task_name == "时序问答":
            if "answer" in response_json:
                print("\n" + "="*50)
                print("❓ 问题回答:")
                print("="*50)
                print(f"问题: {response_json.get('question', '未知问题')}")
                print(f"回答: {response_json['answer']}")
                print(f"置信度: {response_json.get('answer_confidence', 'N/A')}")
                print("="*50)
            else:
                print("⚠ 警告：输出JSON中未找到'answer'字段")
                print("🤖 模型原始回答：", response)
    
    except json.JSONDecodeError as e:
        print("❌ 错误：模型输出不是有效的JSON格式")
        print("🤖 模型原始回答：", response)
    except Exception as e:
        print(f"❌ 处理输出时发生错误: {e}")
        print("🤖 模型原始回答：", response)
    
    print(f"模型加载与推理总耗时: {time.time() - t0:.2f}秒")


if __name__ == "__main__":
 # 修改：从控制台输入获取任务类型
    print("=" * 50)
    print("可用任务列表:")
    print("1. 视频描述")
    print("2. 时序问答")
    print("=" * 50)
    
 
    user_input = input("\n请选择要执行的任务（输入 1 或 2，或输入任务名称，输入 '退出' 结束程序）: ").strip()
    
    if user_input.lower() in ["退出", "exit", "quit", "q"]:
        print("程序结束")
    
    # 将用户输入转换为任务名称
    if user_input == "1":
        task_name = "视频描述"
    elif user_input == "2":
        task_name = "时序问答"
    elif user_input in ["视频描述", "时序问答"]:
        task_name = user_input
    else:
        print(f"无效输入: '{user_input}'，请重新输入")

    
    # 执行选定的任务
    print(f"\n开始执行任务: {task_name}")
    run_once(task_name=task_name)
        