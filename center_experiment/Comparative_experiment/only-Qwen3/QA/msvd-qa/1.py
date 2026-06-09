import json

def aggregate_qa_files(file1_path, file2_path, output_file_path):
    """
    将文件1和文件2通过 question_id 进行匹配和聚合，生成文件3格式的新文件。
    
    :param file1_path: test_qa.json 的路径 (标准JSON数组格式)
    :param file2_path: direct_video_qa_results... 的路径 (每行一个JSON对象的 JSONL 格式)
    :param output_file_path: 生成的文件3 (qa_comparison_minimal.json) 路径
    """
    # 1. 读取文件1 (建立 question_id -> gt_answer 的映射)
    file1_data = {}
    with open(file1_path, 'r', encoding='utf-8') as f:
        try:
            # 尝试标准 JSON 数组解析
            questions_list = json.load(f)
            for item in questions_list:
                q_id = str(item.get('id'))
                file1_data[q_id] = item.get('answer')
        except json.JSONDecodeError:
            # 若不是标准数组则按行读取
            f.seek(0)
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    q_id = str(item.get('id'))
                    file1_data[q_id] = item.get('answer')

    # 2. 读取文件2 并进行匹配聚合
    aggregated_results = []
    with open(file2_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            file2_item = json.loads(line)
            q_id = str(file2_item.get('question_id'))
            
            # 只有当文件1和文件2都有该条 question_id 时才提取聚合
            if q_id in file1_data:
                # 获取模型预测答案
                model_answer = file2_item.get('answer')
                # 从文件1中获取真实答案 (Ground Truth)
                gt_answer = file1_data[q_id]
                
                # 获取并标准化视频ID (例如将 1451 转换为 "vid1451")
                raw_video_id = file2_item.get('video_id')
                if isinstance(raw_video_id, int) or (isinstance(raw_video_id, str) and raw_video_id.isdigit()):
                    video_id_gt = f"vid{raw_video_id}"
                else:
                    video_id_gt = str(raw_video_id)
                
                # 构造符合文件3格式的单行字典
                comparison_item = {
                    "question_id": q_id,
                    "video_id_gt": video_id_gt,
                    "model_answer": model_answer,
                    "gt_answer": gt_answer
                }
                aggregated_results.append(comparison_item)

    # 3. 将聚合结果以每行一个JSON对象(JSONL)的格式写入新文件3
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in aggregated_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"数据聚合完成！成功匹配并生成了 {len(aggregated_results)} 条数据，已保存至: {output_file_path}")

# ==========================================
# 使用示例：
# ==========================================
if __name__ == "__main__":
    # 请根据您的本地实际情况替换文件名或路径
    file1 = "/data3/health/wyy/data/QA/MSVD-QA/test_qa.json"
    file2 = "/data3/health/wyy/BLIP2/experiment/center_experiment/Ablation_experiment/full_model/QA/qwen3_vl_results/video_qa_results_20260608_164438.jsonl"
    file3_output = "/data3/health/wyy/BLIP2/experiment/center_experiment/Ablation_experiment/full_model/QA/qa_comparison_results/qa_comparison_minimal_new.json"
    
    aggregate_qa_files(file1, file2, file3_output)