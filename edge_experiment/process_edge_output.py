import json
import re
import os
from datetime import datetime
from typing import Dict, List, Any

def parse_yolo_log_to_dict(yolo_log: str) -> Dict[float, List[Dict]]:
    """
    将YOLO日志字符串解析为按时间戳索引的结构化字典
    返回格式: {时间戳: [{"object": 目标类型, "bbox": [坐标], "conf": 置信度}, ...]}
    """
    frame_dict = {}
    
    lines = yolo_log.strip().split('\n')
    
    for line in lines:
        if not line.strip():
            continue
            
        time_match = re.search(r'\[Time:\s*([\d.]+)s\]', line)
        if not time_match:
            continue
            
        timestamp = float(time_match.group(1))
        content = line[time_match.end():].strip()
        
        detections = []
        # 分割多个检测结果
        detection_parts = [part.strip() for part in content.split(';') if part.strip()]
        
        for part in detection_parts:
            # 匹配目标类型和坐标
            match = re.match(r'(\w+)\s+found at\s+\[([\d.,\s]+)\]', part)
            if match:
                obj_type = match.group(1)
                bbox_str = match.group(2)
                bbox = [float(x.strip()) for x in bbox_str.split(',')]
                
                # 尝试提取置信度（如果存在）
                conf_match = re.search(r'conf:([\d.]+)', part)
                confidence = float(conf_match.group(1)) if conf_match else 0.0
                
                detections.append({
                    "object": obj_type,
                    "bbox": bbox,
                    "confidence": confidence
                })
            else:
                # 如果没有匹配到标准格式，保存原始字符串
                detections.append({
                    "object": "unknown",
                    "raw_text": part
                })
        
        frame_dict[timestamp] = detections
    
    return frame_dict

def format_yolo_detection(yolo_detections: List[Dict]) -> str:
    """
    格式化YOLO检测结果为字符串
    """
    if not yolo_detections:
        return "无检测结果"
    
    formatted = []
    for det in yolo_detections:
        if "raw_text" in det:
            formatted.append(det["raw_text"])
        else:
            bbox_str = ','.join([f"{coord:.2f}" for coord in det.get("bbox", [])])
            conf = det.get("confidence", 0)
            if conf > 0:
                formatted.append(f"{det['object']}[{bbox_str}]conf:{conf:.2f}")
            else:
                formatted.append(f"{det['object']}[{bbox_str}]")
    
    return '; '.join(formatted)

def create_structured_report(json_data: Dict, video_type: str = "无人机视频") -> Dict[str, Any]:
    """
    从单个JSON数据创建结构化报告字典
    """
    # 解析YOLO日志
    yolo_log = json_data.get('yolo_detection_segmentation_log', '')
    yolo_frame_dict = parse_yolo_log_to_dict(yolo_log)
    
    # 构建关键帧信息列表
    key_frames_list = []
    key_frames_info = json_data.get('key_frames_info', [])
    
    for i, frame_info in enumerate(key_frames_info):
        frame_idx = i + 1
        timestamp = frame_info.get('timestamp', 0)
        description = frame_info.get('description', '')
        
        # 获取该时间戳对应的YOLO检测结果
        yolo_detections = yolo_frame_dict.get(timestamp, [])
        
        key_frame_data = {
            "frame_id": f"F{frame_idx:03d}",
            "timestamp": timestamp,
            "timestamp_str": f"{timestamp:.1f}s",
            "blip2_description": description if description else "无描述",
            "yolo_detections": yolo_detections,  # 结构化数据
            "yolo_detections_formatted": format_yolo_detection(yolo_detections),  # 格式化字符串
        }
        
        key_frames_list.append(key_frame_data)
    
    # 构建完整报告结构
    program_time = json_data.get('program_run_time', '')
    if program_time:
        try:
            dt = datetime.strptime(program_time, "%Y-%m-%d %H:%M:%S")
            utc_time = dt.strftime("%Y-%m-%d %H:%M:%S.000")
        except:
            utc_time = f"{program_time}.000"
    else:
        utc_time = "需手动补充采集时间"
    
    structured_report = {
        "metadata": {
            "video_type": video_type,
            "program_run_time": program_time,
            "report_generation_time": datetime.now().isoformat(),
            "source_file": json_data.get('video_source', '')
        },
        "global_temporal_info": {
            "video_duration_seconds": json_data.get('video_duration_seconds', 0),
            "video_duration_formatted": f"{json_data.get('video_duration_seconds', 0):.1f}s",
            "utc_start_time": utc_time, 
            "total_key_frames": json_data.get('key_frames_count', 0),
            "raw_sampled_frames": json_data.get('raw_sampled_frames_count', 0)
        },
        "blip2_global_summary": {
            "summary": json_data.get('video_summary', ''),
            "summary_zh": json_data.get('video_summary', '')  # 这里可以添加翻译后的中文摘要
        },
        "key_frames_binding_list": key_frames_list,
        "saved_resources": {
            "keyframes_paths": json_data.get('saved_keyframes_paths', []),
            "original_yolo_log": yolo_log
        }
    }
    
    return structured_report

def generate_human_readable_report(structured_data: Dict) -> str:
    """
    从结构化数据生成人类可读的报告文本（用于参考）
    """
    report_lines = []
    
    # 全局时序信息
    global_info = structured_data["global_temporal_info"]
    report_lines.append(f"【{structured_data['metadata']['video_type']}全局时序信息】")
    report_lines.append(f"视频总时长：{global_info['video_duration_formatted']}")
    report_lines.append(f"视频采集UTC起始时间：{global_info['utc_start_time']}")
    report_lines.append(f"总关键帧数量：{global_info['total_key_frames']}")
    report_lines.append("")
    
    # BLIP2全局语义摘要
    blip_summary = structured_data["blip2_global_summary"]
    report_lines.append("【BLIP2全局语义摘要】")
    report_lines.append(blip_summary["summary_zh"] if blip_summary["summary_zh"] else blip_summary["summary"])
    report_lines.append("")
    
    # 关键帧绑定文本列表
    report_lines.append("【关键帧绑定文本列表】")
    
    for i, frame in enumerate(structured_data["key_frames_binding_list"]):
        report_lines.append(f"帧ID:{frame['frame_id']}|时间戳:{frame['timestamp_str']}")
        report_lines.append(f"• BLIP2帧语义：{frame['blip2_description']}")
        report_lines.append(f"• YOLO预检测结果：{frame['yolo_detections_formatted'] }")
        
        if i < len(structured_data["key_frames_binding_list"]) - 1:
            report_lines.append("")
            report_lines.append("")
    
    return "\n".join(report_lines)

def process_jsonl_to_structured_jsonl(input_jsonl_path: str, output_dir: str = "./structured_reports"):
    """
    处理JSONL文件，生成结构化的JSONL报告
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件名
    input_basename = os.path.basename(input_jsonl_path)
    input_name = os.path.splitext(input_basename)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 结构化JSONL输出路径
    jsonl_output_path = os.path.join(output_dir, f"{input_name}_structured_{timestamp}.jsonl")
    # 人类可读文本报告路径
    text_output_path = os.path.join(output_dir, f"{input_name}_human_readable_{timestamp}.txt")
    
    structured_reports = []
    text_reports = []
    
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                json_data = json.loads(line)
                
                # 判断视频类型
                video_source = json_data.get('video_source', '')
                if '无人机' in video_source or 'drone' in video_source.lower():
                    video_type = "无人机视频"
                elif '监控' in video_source or 'surveillance' in video_source.lower():
                    video_type = "监控视频"
                else:
                    video_type = "通用视频"
                
                print(f"处理第{line_num}行: {os.path.basename(video_source) if video_source else '未知视频'}")
                
                # 创建结构化报告
                structured_report = create_structured_report(json_data, video_type)
                structured_reports.append(structured_report)
                
                # 生成人类可读报告（可选）
                human_report = generate_human_readable_report(structured_report)
                text_reports.append(human_report)
                
            except json.JSONDecodeError as e:
                print(f"第{line_num}行JSON解析错误: {e}")
                continue
            except Exception as e:
                print(f"处理第{line_num}行时发生错误: {e}")
                continue
    
    # 保存结构化JSONL文件
    if structured_reports:
        with open(jsonl_output_path, 'w', encoding='utf-8') as f:
            for report in structured_reports:
                json_line = json.dumps(report, ensure_ascii=False, indent=2)
                f.write(json_line + '\n')
        print(f"\n结构化JSONL报告已保存至: {jsonl_output_path}")
        print(f"共包含 {len(structured_reports)} 条记录")
    
    # 保存人类可读文本报告（可选）
    if text_reports:
        with open(text_output_path, 'w', encoding='utf-8') as f:
            for i, report in enumerate(text_reports):
                f.write(f"报告 #{i+1}")
                f.write("\n" + "="*50 + "\n")
                f.write(report)
                if i < len(text_reports) - 1:
                    f.write("\n\n" + "="*50 + "\n\n")
        print(f"人类可读文本报告已保存至: {text_output_path}")
    
    return structured_reports, jsonl_output_path

def main():
    """
    主函数：处理JSONL文件并生成结构化报告
    """
    # 配置参数
    input_jsonl_path = "/data3/health/wyy/BLIP2/experiment/data/msrvtt/videos_analysis_results.jsonl"  # 您的输入JSONL文件路径
    output_directory = "/data3/health/wyy/BLIP2/experiment/data/msrvtt/structured_reports"  # 输出目录
    
    print("开始处理JSONL文件并生成结构化报告...")
    print(f"输入文件: {input_jsonl_path}")
    print(f"输出目录: {output_directory}")
    print("-" * 50)
    
    try:
        structured_reports, output_path = process_jsonl_to_structured_jsonl(input_jsonl_path, output_directory)
        
        if structured_reports:
            print(f"\n处理完成！")
            print(f"结构化报告字段包括:")
            print(f"  - metadata: 元数据信息")
            print(f"  - global_temporal_info: 全局时序信息")
            print(f"  - blip2_global_summary: BLIP2全局摘要")
            print(f"  - key_frames_binding_list: 关键帧绑定列表（包含详细检测信息）")
            print(f"  - saved_resources: 保存的资源路径")
            
            # 显示第一条记录的示例
            print(f"\n第一条记录的结构示例:")
            sample = structured_reports[0] if structured_reports else {}
            print(json.dumps({
                "global_temporal_info": sample.get("global_temporal_info", {}),
                "key_frames_count": len(sample.get("key_frames_binding_list", []))
            }, ensure_ascii=False, indent=2))
            
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_jsonl_path}")
        print("请检查文件路径是否正确")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    main()