import json
import os
import torch
from bert_score import model2layers, score
from tqdm import tqdm

# 1. 模仿文件3，指定 GPU 显卡和本地模型路径
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
model_path = "/data3/health/model/roberta-large"  # 你的本地 BERT 模型路径
model2layers[model_path] = 12  # 映射层数


def load_file1_jsonlines(file1_path):
    """读取文件1 (JSON Lines 格式)"""
    data_dict = {}
    with open(file1_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                v_id = item.get("video_id")
                summary = item.get("full_video_summary")
                if v_id and summary:
                    data_dict[v_id] = summary
            except json.JSONDecodeError:
                continue
    return data_dict


def load_file2_jsonarray(file2_path):
    """读取文件2 (标准 JSON 数组格式)"""
    with open(file2_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def compute_and_save_bertscore(file1_path, file2_path, output_path, batch_size=64):
    # 2. 加载并对齐数据
    print("正在加载并对齐数据...")
    summary_map = load_file1_jsonlines(file1_path)
    msrvtt_data = load_file2_jsonarray(file2_path)

    cands = []  # 预测值 (full_video_summary)
    refs = []  # 参考值 (caption)
    paired_entries = []  # 用于记录元数据，方便后续保存

    for item in msrvtt_data:
        v_id = item.get("video_id")
        caption = item.get("caption")

        # 检查文件1中是否存在对应的 video_id
        if v_id in summary_map and caption:
            summary = summary_map[v_id]
            cands.append(summary)
            refs.append(caption)
            paired_entries.append((v_id, caption, summary))

    print(f"成功对齐数据共: {len(cands)} 条")
    if len(cands) == 0:
        print("没有找到匹配的 video_id，请检查文件内容！")
        return

    # 3. 模仿文件3，分批处理并计算 BERTScore
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前使用设备: {device}")

    all_p_scores = []
    all_r_scores = []
    all_f1_scores = []

    for i in tqdm(range(0, len(cands), batch_size), desc="计算 BERTScore"):
        c_batch = cands[i : i + batch_size]
        r_batch = refs[i : i + batch_size]

        # 调用 bert_score 库
        P, R, F1 = score(
            cands=c_batch,
            refs=r_batch,
            lang="en",
            model_type=model_path,
            device=device,
        )

        all_p_scores.extend(P.tolist())
        all_r_scores.extend(R.tolist())
        all_f1_scores.extend(F1.tolist())

    # 4. 打印整体统计指标
    f1_tensor = torch.tensor(all_f1_scores)
    print("\n" + "=" * 15 + " BERTScore 统计结果 " + "=" * 15)
    print(f"数据集整体平均 F1: {f1_tensor.mean().item():.4f}")
    print(f"  最高分: {f1_tensor.max().item():.4f}")
    print(f"  最低分: {f1_tensor.min().item():.4f}")
    print(f"  标准差: {f1_tensor.std().item():.4f}")

    # 5. 模仿文件3的 save_results 逻辑保存到新的 json 文件
    results = []
    for (vid, caption, summary), p, r, f1 in zip(
        paired_entries, all_p_scores, all_r_scores, all_f1_scores
    ):
        results.append(
            {
                "video_id": vid,
                "msrvtt_caption": caption,
                "full_video_summary": summary,
                "bertscore_p": float(p),
                "bertscore_r": float(r),
                "bertscore_f1": float(f1),
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"结果已成功保存到: {output_path}")


if __name__ == "__main__":
    # 请根据你本地的实际路径配置文件名
    file1 = "/data3/health/wyy/BLIP2/experiment/data/msrvtt/extracted_video_summaries.jsonl" 
    file2 = "/data3/health/wyy/data/summary/MSR-VTT/msrvtt_test_1k.json" # 原始msrvtt数据集中的标签文件，请自行下载
    output = "/data3/health/wyy/BLIP2/experiment/data/msrvtt/video_summary_bertscore_results.json"

    compute_and_save_bertscore(file1, file2, output, batch_size=64)