"""
evaluate_answer_pairs_no_args.py
功能：针对包含 `model_answer` 和 `gt_answer` 字段的JSONL数据，计算完全匹配率和BERTScore。
特点：无需命令行参数，直接在脚本中配置路径和参数。
"""

import json
import torch
from tqdm import tqdm
from bert_score import score, model2layers
import torch
import os
from typing import List, Dict, Tuple, Any
import numpy as np

# ==================== 配置区域 ====================
# 请根据您的实际情况修改以下配置

# 1. 数据文件路径
DATA_PATH = "/data3/health/wyy/BLIP2/final/main_path/experiment/Ablation_experiment/without_summary_information/QA/qa_comparison_results/qa_comparison_minimal.jsonl"  # 替换为您的JSONL文件路径
OUTPUT_PATH = "/data3/health/wyy/BLIP2/final/main_path/experiment/Ablation_experiment/without_summary_information/QA/qa_comparison_results/evaluation_results.json"  # 结果输出文件路径
DETAILED_RESULTS_PATH = "/data3/health/wyy/BLIP2/final/val/msvd-qa/detailed_results.jsonl"  # 新增：每个样本的详细结果

# 2. BERT模型配置
MODEL_PATH = "/data3/health/model/roberta-large"  # 本地BERT模型路径
model2layers[MODEL_PATH] = 12  # 模型层数，roberta-large通常是12层

# 3. 计算参数
BATCH_SIZE = 64  # 批处理大小，根据GPU内存调整

# 4. CUDA设备设置（可选）
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 指定GPU设备
# ==================== 配置结束 ====================

# -------------------- 工具函数 --------------------

def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件（每行一个JSON对象）。"""
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

def calculate_exact_match(candidate: str, reference: str) -> int:
    """
    计算完全匹配。
    返回1表示完全匹配，0表示不匹配。
    """
    return 1 if candidate.strip().lower() == reference.strip().lower() else 0

def calculate_bertscore_batch(
    candidates: List[str], 
    references: List[str], 
    batch_size: int = BATCH_SIZE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算一个批次数据的BERTScore。
    返回: (精确度P, 召回率R, F1值)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    P, R, F1 = score(
        cands=candidates,
        refs=references,
        lang="en",  # 假设您的答案是英文
        model_type=MODEL_PATH,
        device=device,
        batch_size=batch_size,
        verbose=False  # 禁用bert_score自带的进度条
    )
    return P, R, F1

def calculate_statistics(tensor: torch.Tensor) -> Dict[str, float]:
    """计算张量的统计信息"""
    if tensor.numel() == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "max": 0.0,
            "min": 0.0,
            "std": 0.0,
            "q1": 0.0,
            "q3": 0.0
        }
    
    return {
        "mean": float(tensor.mean().item()),
        "median": float(tensor.median().item()),
        "max": float(tensor.max().item()),
        "min": float(tensor.min().item()),
        "std": float(tensor.std().item()),
        "q1": float(tensor.kthvalue(max(1, len(tensor) // 4))[0].item()),  # 第一四分位数
        "q3": float(tensor.kthvalue(min(len(tensor), 3 * len(tensor) // 4))[0].item())  # 第三四分位数
    }

# -------------------- 核心评估函数 --------------------

def evaluate_answer_pairs():
    """
    主评估函数：计算完全匹配率和BERTScore。
    
    返回:
        包含详细统计结果的字典。
    """
    print(f"正在加载数据: {DATA_PATH}")
    entries = load_jsonl(DATA_PATH)
    
    if not entries:
        print("错误：未加载到任何有效数据。")
        return {}
    
    print(f"成功加载 {len(entries)} 条记录。")
    
    # 检查必要字段
    required_fields = ['model_answer', 'gt_answer']
    valid_entries = []
    invalid_count = 0
    
    for i, entry in enumerate(entries):
        if not all(field in entry for field in required_fields):
            print(f"警告：跳过第{i+1}条记录，缺少必要字段: {entry}")
            invalid_count += 1
            continue
        
        # 确保答案是字符串类型
        if not isinstance(entry['model_answer'], str) or not isinstance(entry['gt_answer'], str):
            print(f"警告：第{i+1}条记录的答案字段不是字符串类型，已转换为字符串。")
            entry['model_answer'] = str(entry['model_answer'])
            entry['gt_answer'] = str(entry['gt_answer'])
        
        valid_entries.append(entry)
    
    if invalid_count > 0:
        print(f"跳过 {invalid_count} 条无效记录。")
    
    if not valid_entries:
        print("错误：没有包含有效问答对的记录。")
        return {}
    
    print(f"开始评估 {len(valid_entries)} 条有效记录...")
    print(f"使用BERT模型: {MODEL_PATH}")
    print(f"批处理大小: {BATCH_SIZE}")
    
    # 准备数据列表
    model_answers = []
    gt_answers = []
    entry_ids = []  # 用于标识每条记录
    
    for i, entry in enumerate(valid_entries):
        model_answers.append(entry['model_answer'])
        gt_answers.append(entry['gt_answer'])
        # 尝试使用 question_id 或 video_id_gt 作为标识，否则使用索引
        entry_id = entry.get('question_id', entry.get('video_id_gt', f"entry_{i}"))
        entry_ids.append(entry_id)
    
    # 存储每条记录的结果
    all_results = []
    total_exact_match = 0
    
    # 批次计算BERTScore
    all_f1_scores = []
    all_precision_scores = []
    all_recall_scores = []
    
    print("开始计算BERTScore（这可能需要一些时间）...")
    for i in tqdm(range(0, len(valid_entries), BATCH_SIZE), desc="计算BERTScore"):
        batch_model = model_answers[i:i+BATCH_SIZE]
        batch_gt = gt_answers[i:i+BATCH_SIZE]
        batch_ids = entry_ids[i:i+BATCH_SIZE]
        
        # 计算当前批次的BERTScore
        P_batch, R_batch, F1_batch = calculate_bertscore_batch(batch_model, batch_gt, len(batch_model))
        
        # 处理当前批次中的每条记录
        for j in range(len(batch_model)):
            idx = i + j
            model_ans = batch_model[j]
            gt_ans = batch_gt[j]
            entry_id = batch_ids[j]
            
            # 计算完全匹配
            em_score = calculate_exact_match(model_ans, gt_ans)
            total_exact_match += em_score
            
            # 获取BERTScore
            f1_score = float(F1_batch[j].item())
            precision_score = float(P_batch[j].item())
            recall_score = float(R_batch[j].item())
            
            all_f1_scores.append(f1_score)
            all_precision_scores.append(precision_score)
            all_recall_scores.append(recall_score)
            
            # 存储当前记录的结果
            result_entry = {
                "entry_id": entry_id,
                "model_answer": model_ans,
                "gt_answer": gt_ans,
                "exact_match": em_score,
                "bertscore_precision": precision_score,
                "bertscore_recall": recall_score,
                "bertscore_f1": f1_score
            }
            
            # 添加原始记录中的其他字段（如果存在）
            for key, value in valid_entries[idx].items():
                if key not in result_entry:
                    result_entry[key] = value
            
            all_results.append(result_entry)
    
    # -------------------- 统计与输出 --------------------
    print("\n" + "="*100)
    print("评估结果摘要")
    print("="*100)
    
    # 转换为Tensor以便计算统计信息
    f1_tensor = torch.tensor(all_f1_scores)
    precision_tensor = torch.tensor(all_precision_scores)
    recall_tensor = torch.tensor(all_recall_scores)
    
    # 计算完全匹配率
    exact_match_rate = total_exact_match / len(valid_entries)
    
    print(f"评估记录总数: {len(valid_entries)}")
    print(f"\n1. 完全匹配率 (Exact Match Rate):")
    print(f"   匹配数: {total_exact_match} / {len(valid_entries)}")
    print(f"   比率: {exact_match_rate:.4f} ({exact_match_rate*100:.2f}%)")
    
    # 计算各指标的统计信息
    f1_stats = calculate_statistics(f1_tensor)
    precision_stats = calculate_statistics(precision_tensor)
    recall_stats = calculate_statistics(recall_tensor)
    
    print(f"\n2. BERTScore F1 统计:")
    print(f"   平均值: {f1_stats['mean']:.4f}")
    print(f"   中位数: {f1_stats['median']:.4f}")
    print(f"   最高分: {f1_stats['max']:.4f}")
    print(f"   最低分: {f1_stats['min']:.4f}")
    print(f"   标准差: {f1_stats['std']:.4f}")
    print(f"   第一四分位数(Q1): {f1_stats['q1']:.4f}")
    print(f"   第三四分位数(Q3): {f1_stats['q3']:.4f}")
    
    print(f"\n3. BERTScore 精确率 (Precision) 统计:")
    print(f"   平均值: {precision_stats['mean']:.4f}")
    print(f"   中位数: {precision_stats['median']:.4f}")
    print(f"   最高分: {precision_stats['max']:.4f}")
    print(f"   最低分: {precision_stats['min']:.4f}")
    print(f"   标准差: {precision_stats['std']:.4f}")
    print(f"   第一四分位数(Q1): {precision_stats['q1']:.4f}")
    print(f"   第三四分位数(Q3): {precision_stats['q3']:.4f}")
    
    print(f"\n4. BERTScore 召回率 (Recall) 统计:")
    print(f"   平均值: {recall_stats['mean']:.4f}")
    print(f"   中位数: {recall_stats['median']:.4f}")
    print(f"   最高分: {recall_stats['max']:.4f}")
    print(f"   最低分: {recall_stats['min']:.4f}")
    print(f"   标准差: {recall_stats['std']:.4f}")
    print(f"   第一四分位数(Q1): {recall_stats['q1']:.4f}")
    print(f"   第三四分位数(Q3): {recall_stats['q3']:.4f}")
    
    # 输出部分示例记录
    print(f"\n5. 前5条记录的详细结果:")
    print("-"*120)
    print(f"{'序号':<5} {'ID':<12} {'模型答案':<20} {'标准答案':<20} {'完全匹配':<10} {'P':<8} {'R':<8} {'F1':<8}")
    print("-"*120)
    
    for idx, result in enumerate(all_results[:5]):
        model_display = (result['model_answer'][:18] + '..') if len(result['model_answer']) > 20 else result['model_answer']
        gt_display = (result['gt_answer'][:18] + '..') if len(result['gt_answer']) > 20 else result['gt_answer']
        
        exact_match_str = "是" if result['exact_match'] == 1 else "否"
        print(f"{idx+1:<5} {result['entry_id']:<12} {model_display:<20} {gt_display:<20} {exact_match_str:<10} "
              f"{result['bertscore_precision']:.4f}  {result['bertscore_recall']:.4f}  {result['bertscore_f1']:.4f}")
    
    # 找出一些高分和低分的例子
    if len(all_results) >= 5:
        sorted_by_f1 = sorted(all_results, key=lambda x: x['bertscore_f1'], reverse=True)
        
        print(f"\n6. BERTScore 最高和最低的示例:")
        print("-"*100)
        print("最高分示例 (Top 3):")
        for i in range(min(3, len(sorted_by_f1))):
            result = sorted_by_f1[i]
            print(f"  F1={result['bertscore_f1']:.4f} (P={result['bertscore_precision']:.4f}, R={result['bertscore_recall']:.4f}): "
                  f"模型='{result['model_answer']}', 标准='{result['gt_answer']}'")
        
        print("\n最低分示例 (Bottom 3):")
        for i in range(min(3, len(sorted_by_f1))):
            idx = -1 - i
            result = sorted_by_f1[idx]
            print(f"  F1={result['bertscore_f1']:.4f} (P={result['bertscore_precision']:.4f}, R={result['bertscore_recall']:.4f}): "
                  f"模型='{result['model_answer']}', 标准='{result['gt_answer']}'")
    
    # 计算各分数区间的分布
    print(f"\n7. BERTScore F1 分数区间分布:")
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i in range(len(bins)-1):
        count = sum(1 for score in all_f1_scores if bins[i] <= score < bins[i+1])
        percentage = count / len(all_f1_scores) * 100
        print(f"   [{bins[i]:.1f}-{bins[i+1]:.1f}): {count} 条 ({percentage:.1f}%)")
    
    # 保存结果到文件
    final_results = {
        "summary": {
            "total_entries": len(valid_entries),
            "exact_match_count": total_exact_match,
            "exact_match_rate": float(exact_match_rate),
            "bertscore_f1": f1_stats,
            "bertscore_precision": precision_stats,
            "bertscore_recall": recall_stats
        },
        "detailed_results": all_results
    }
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    # 同时保存每个样本的详细结果为JSONL格式
    with open(DETAILED_RESULTS_PATH, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\n详细结果已保存到: {OUTPUT_PATH}")
    print(f"每个样本的详细结果已保存到: {DETAILED_RESULTS_PATH}")
    print("="*100)
    
    return final_results

# -------------------- 主程序入口 --------------------
if __name__ == "__main__":
    # 直接运行评估函数
    results = evaluate_answer_pairs()