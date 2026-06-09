import json
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


def calculate_video_metrics(summary_file_path, captions_file_path):
    # 1. 读取文件2 (标准 JSON 数组)，获取参考标签并建立映射
    print("正在读取参考标签 (extracted_captions.json)...")
    with open(captions_file_path, "r", encoding="utf-8") as f:
        captions_data = json.load(f)

    # 建立映射字典: {video_id: ["caption1", "caption2", ...]}
    ref_dict = {item["video_id"]: item["captions"] for item in captions_data}

    # 2. 读取文件1 (JSON Lines 格式)，逐行解析预测摘要
    print("正在读取预测摘要 (video_analysis_results_20260606_113250.json)...")
    gts = {}  # 真值 Ground Truths
    res = {}  # 预测结果 Results

    with open(summary_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                v_id = item.get("video_id")
                summary = item.get("full_video_summary")

                # 如果两个文件里都存在该 video_id，且 summary 不为空，则进行对齐
                if v_id and summary and v_id in ref_dict:
                    # pycocoevalcap 要求格式必须是：{video_id: [句子字符串列表]}
                    gts[v_id] = ref_dict[v_id]
                    res[v_id] = [summary]
            except json.JSONDecodeError:
                continue

    print(f"成功对齐并准备评测的视频数量: {len(res)} 条")

    if len(res) == 0:
        print("未找到匹配的 video_id，请检查两个文件的数据是否对应！")
        return

    # 3. 计算四大类指标
    print("\n开始计算各项指标（这可能需要一些时间）...")

    print("-> 正在计算 BLEU-4...")
    bleu_scorer = Bleu(4)
    bleu_score, _ = bleu_scorer.compute_score(gts, res)

    print("-> 正在计算 METEOR (调用 Java 子进程)...")
    meteor_scorer = Meteor()
    meteor_score, _ = meteor_scorer.compute_score(gts, res)

    print("-> 正在计算 ROUGE-L...")
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(gts, res)

    print("-> 正在计算 CIDEr...")
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)

    # 4. 打印最终结果
    print("\n" + "=" * 25 + " 视频评测最终结果 " + "=" * 25)
    print(f"BLEU-1:  {bleu_score[0]:.4f}")
    print(f"BLEU-2:  {bleu_score[1]:.4f}")
    print(f"BLEU-3:  {bleu_score[2]:.4f}")
    print(f"BLEU-4:  {bleu_score[3]:.4f}")
    print(f"METEOR:  {meteor_score:.4f}")
    print(f"ROUGE-L: {rouge_score:.4f}")
    print(f"CIDEr:   {cider_score:.4f}")
    print("=" * 68)


if __name__ == "__main__":
    # 配置你本地的实际文件路径
    file1_path = "/data3/health/wyy/BLIP2/experiment/data/msrvtt/extracted_video_summaries.jsonl"
    file2_path = "/data3/health/wyy/data/summary/MSR-VTT/extracted_captions.json"

    calculate_video_metrics(file1_path, file2_path)