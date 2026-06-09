import json
import torch
from tqdm import tqdm
from bert_score import score, model2layers
import os

# 可选：指定使用的GPU，如果您的环境有多个GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 定义模型路径和层数映射 (与文档1保持一致的设置)
model_path = "/data3/health/model/roberta-large"  # 请确保此路径在您的服务器上有效
model2layers[model_path] = 12  # 假设是12层的RoBERTa-large模型

def calculate_bertscore_for_answers(cands, refs, batch_size=64):
    """
    计算两组文本之间BERTScore P、R、F1的函数
    Args:
        cands: 候选文本列表 (通常为模型的answer)
        refs: 参考文本列表 (通常为original_answer)
        batch_size: 批处理大小
    Returns:
        所有样本的P、R、F1分数列表 (tensor)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    all_p_scores = []
    all_r_scores = []
    all_f1_scores = []

    for i in tqdm(range(0, len(cands), batch_size), desc="计算BERTScore"):
        c_batch = cands[i:i+batch_size]
        r_batch = refs[i:i+batch_size]

        # 调用bert_score库进行计算
        # 注意：这里 cands 是模型的答案，refs 是原始答案
        P, R, F1 = score(
            cands=c_batch,
            refs=r_batch,
            lang="en",  # 因为您的问题和答案都是英文
            model_type=model_path,
            device=device,
            verbose=False  # 关闭库自带的进度条，避免与tqdm冲突
        )
        all_p_scores.extend(P.tolist())
        all_r_scores.extend(R.tolist())
        all_f1_scores.extend(F1.tolist())

    return (torch.tensor(all_p_scores),  # 返回P值的tensor
            torch.tensor(all_r_scores),  # 返回R值的tensor
            torch.tensor(all_f1_scores))  # 返回F1值的tensor

def calculate_exact_match(cands, refs):
    """
    计算完全匹配度
    Args:
        cands: 候选文本列表 (通常为模型的answer)
        refs: 参考文本列表 (通常为original_answer)
    Returns:
        每个样本的匹配结果列表 (True/False)
    """
    exact_matches = []
    for cand, ref in zip(cands, refs):
        # 去除首尾空格，并转换为小写进行比较（更严格的匹配可以去掉.lower()）
        exact_matches.append(cand.strip().lower() == ref.strip().lower())
    return exact_matches

def load_jsonl(file_path):
    """加载JSONL文件（每行一个JSON对象）"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"警告：JSONL文件第{line_num}行解析错误: {e}")
                continue
    return data

def evaluate_answers(jsonl_path, batch_size=64):
    """
    评估jsonl文件中original_answer与answer的相似度
    Args:
        jsonl_path: 您的模型输出JSONL文件路径
        batch_size: 批处理大小
    """
    print("开始加载数据...")
    # 加载数据
    try:
        entries = load_jsonl(jsonl_path)
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        return None, None, None, None, None
    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        return None, None, None, None, None

    if not entries:
        print("错误：未加载到任何有效数据。")
        return None, None, None, None, None

    print(f"成功加载 {len(entries)} 条数据。")

    # 提取original_answer和answer字段
    original_answers = []
    model_answers = []
    valid_entries = []  # 保存有效的数据条目，用于后续结果关联
    error_count = 0

    for idx, item in enumerate(entries):
        if not isinstance(item, dict):
            print(f"警告：跳过第{idx+1}行，不是字典格式。")
            error_count += 1
            continue

        original_ans = item.get('original_answer')
        model_ans = item.get('answer')

        # 检查必要字段是否存在
        if original_ans is None or model_ans is None:
            print(f"警告：跳过第{idx+1}行，缺少'original_answer'或'answer'字段。数据: {item}")
            error_count += 1
            continue

        # 确保两个字段都是字符串类型
        if not isinstance(original_ans, str) or not isinstance(model_ans, str):
            print(f"警告：跳过第{idx+1}行，'original_answer'或'answer'字段不是字符串类型。")
            error_count += 1
            continue

        original_answers.append(original_ans)
        model_answers.append(model_ans)
        valid_entries.append(item)

    if error_count > 0:
        print(f"警告：共跳过 {error_count} 条无效数据。")

    if not original_answers:
        print("错误：没有有效的待比较数据。")
        return None, None, None, None, None

    print(f"即将对 {len(original_answers)} 对答案进行比较...")

    # 1. 计算完全匹配度
    print("\n计算完全匹配度...")
    exact_match_results = calculate_exact_match(model_answers, original_answers)
    exact_match_tensor = torch.tensor([1.0 if match else 0.0 for match in exact_match_results])

    # 2. 计算BERTScore P, R, F1
    print(f"使用本地BERT模型路径: {model_path}")
    print(f"批处理大小: {batch_size}")
    bertscore_p_scores, bertscore_r_scores, bertscore_f1_scores = calculate_bertscore_for_answers(model_answers, original_answers, batch_size)

    # 输出详细结果表格
    print("\n" + "="*150)
    print(f"{'序号':<5} {'Video ID':<12} {'Original Answer':<25} {'Model Answer':<25} {'完全匹配':<10} {'BERTScore P':<12} {'BERTScore R':<12} {'BERTScore F1':<12}")
    print("-"*150)

    for idx, item in enumerate(valid_entries):
        vid = item.get('video_id', 'N/A')
        orig_ans = original_answers[idx]
        mod_ans = model_answers[idx]
        exact_match = "是" if exact_match_results[idx] else "否"
        bertscore_p = bertscore_p_scores[idx].item()
        bertscore_r = bertscore_r_scores[idx].item()
        bertscore_f1 = bertscore_f1_scores[idx].item()

        # 对长文本进行截断显示
        orig_display = (orig_ans[:22] + '...') if len(orig_ans) > 25 else orig_ans
        mod_display = (mod_ans[:22] + '...') if len(mod_ans) > 25 else mod_ans
        print(f"{idx+1:<5} {str(vid):<12} {orig_display:<25} {mod_display:<25} {exact_match:<10} {bertscore_p:.4f}  {bertscore_r:.4f}  {bertscore_f1:.4f}")

    print("-"*150)

    # 输出统计摘要
    print("\n" + "="*60)
    print("评估结果统计摘要")
    print("="*60)
    print(f"总计评估样本数: {len(valid_entries)}")

    # 完全匹配度统计
    exact_match_count = sum(exact_match_results)
    exact_match_rate = exact_match_count / len(exact_match_results)
    print(f"\n1. 完全匹配度 (Exact Match):")
    print(f"   匹配数量: {exact_match_count}")
    print(f"   匹配率: {exact_match_rate:.4f} ({exact_match_rate*100:.2f}%)")

    # BERTScore 统计
    print(f"\n2. BERTScore 精确率 (Precision) 统计:")
    print(f"   平均值: {bertscore_p_scores.mean():.4f}")
    print(f"   中位数: {bertscore_p_scores.median():.4f}")
    print(f"   最高分: {bertscore_p_scores.max():.4f}")
    print(f"   最低分: {bertscore_p_scores.min():.4f}")
    print(f"   标准差: {bertscore_p_scores.std():.4f}")

    print(f"\n3. BERTScore 召回率 (Recall) 统计:")
    print(f"   平均值: {bertscore_r_scores.mean():.4f}")
    print(f"   中位数: {bertscore_r_scores.median():.4f}")
    print(f"   最高分: {bertscore_r_scores.max():.4f}")
    print(f"   最低分: {bertscore_r_scores.min():.4f}")
    print(f"   标准差: {bertscore_r_scores.std():.4f}")

    print(f"\n4. BERTScore F1 统计:")
    print(f"   平均值: {bertscore_f1_scores.mean():.4f}")
    print(f"   中位数: {bertscore_f1_scores.median():.4f}")
    print(f"   最高分: {bertscore_f1_scores.max():.4f}")
    print(f"   最低分: {bertscore_f1_scores.min():.4f}")
    print(f"   标准差: {bertscore_f1_scores.std():.4f}")

    # 保存结果到文件
    save_results(valid_entries, original_answers, model_answers, 
                 exact_match_results, 
                 bertscore_p_scores.tolist(), 
                 bertscore_r_scores.tolist(), 
                 bertscore_f1_scores.tolist())

    return (bertscore_p_scores, bertscore_r_scores, bertscore_f1_scores, 
            exact_match_results, valid_entries)

def save_results(entries, original_answers, model_answers, exact_matches, 
                 bertscore_p, bertscore_r, bertscore_f1, 
                 output_path="answer_evaluation_results.json"):
    """将评估结果保存到JSON文件"""
    results = []
    for i, item in enumerate(entries):
        result = {
            "video_id": item.get('video_id'),
            "question_id": item.get('question_id'),
            "question": item.get('question'),
            "original_answer": original_answers[i],
            "model_answer": model_answers[i],
            "exact_match": bool(exact_matches[i]),  # 将numpy.bool_转换为Python bool
            "bertscore_precision": bertscore_p[i],
            "bertscore_recall": bertscore_r[i],
            "bertscore_f1": bertscore_f1[i],
            "task_type": item.get('task_type'),
            "status": item.get('status')
        }
        results.append(result)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n详细结果已保存到文件: {output_path}")

# 使用示例
if __name__ == "__main__":
    # 例如: path_to_your_jsonl = "/path/to/your/model_output.jsonl"
    path_to_your_jsonl = "/data3/health/wyy/BLIP2/final/main_path/experiment/Comparative_experiment/only-Qwen3/QA/msvd-qa/Qwen3-msvd-qa-answer/direct_video_qa_results_20260409_170732.jsonl"  # 用您的文件路径替换

    # 调用评估函数
    evaluate_answers(path_to_your_jsonl, batch_size=32)