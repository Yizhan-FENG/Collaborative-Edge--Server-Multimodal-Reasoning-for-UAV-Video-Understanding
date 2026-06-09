import json
import os
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm

# ==========================================
# 一定输入各保存结果qa_comparison_minimal.jsonl文件
# ==========================================

LLM_MODEL_PATH = "/data3/health/model/Qwen3-VL-8B-Instruct"
DEVICE_ID = "cuda:5"

def main():
    input_file = "/data3/health/wyy/BLIP2/experiment/Ablation_experiment/without_yolo_information/QA/qa_comparison_results/qa_comparison_minimal.jsonl"
    output_file = "/data3/health/wyy/BLIP2/experiment/Ablation_experiment/without_yolo_information/QA/qa_comparison_results/qa_judge_results_qwen3vl_text.jsonl"
    
    if not os.path.exists(input_file):
        print(f"❌ 找不到输入文件：{input_file}，请检查路径。")
        return

    # 2. 直接用原本的组件加载模型（不需要另外下载模型）
    print(">>> 正在加载 Qwen3-VL-8B-Instruct 提取其纯文本能力...")
    processor = AutoProcessor.from_pretrained(LLM_MODEL_PATH)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        LLM_MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map={"": DEVICE_ID}
    ).eval()
    print("   模型加载成功！将直接以纯文本 LLM 模式运行裁判任务。")

    # 读取问答对
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    total = 0
    em_correct = 0       # 传统字面匹配正确数
    llm_correct = 0      # Qwen3-VL 文本裁判认定的正确数
    refused_count = 0    # 模型主动安全拒绝的样本数
    
    output_f = open(output_file, 'w', encoding='utf-8')
    
    print(">>> 开始让 Qwen3-VL 裁判进行语义等价性判卷...")
    for line in tqdm(lines):
        data = json.loads(line)
        total += 1
        
        pred = data.get('model_answer', '').strip()
        ref = data.get('gt_answer', '').strip()
        
        # 基础字面匹配统计
        if pred.lower() == ref.lower():
            em_correct += 1
            
        # 拦截机制：跳过之前主动安全拒绝的样本（包含 visual anchor 等字样）
        if "visual anchor" in pred.lower() or "no corresponding" in pred.lower() or "unknown" in pred.lower():
            refused_count += 1
            data['judge_result'] = "Refused"
            output_f.write(json.dumps(data, ensure_ascii=False) + "\n")
            continue
            
        # 3. 裁判 Prompt 设计
        system_prompt = (
            "You are a strict and professional academic evaluator for Video QA datasets. "
            "Your job is to judge if the Model Prediction means the same thing, "
            "or is a completely acceptable substitute for the Ground Truth answer under video context."
        )
        
        user_prompt = f"""Compare the Prediction with the Ground Truth (GT) answer. 
Consider synonyms, hypernyms (general terms), and common video actions.

[Examples of Acceptable Matches]:
- 'person' vs 'someone' or 'man' -> Yes (Reasonable generalization)
- 'gun' or 'rifle' vs 'weapon' -> Yes (Weapon category match)
- 'chopping' vs 'cut' or 'chop' -> Yes (Same action, different tense)
- 'violin' vs 'fiddle' -> Yes (Synonyms)

[Current Pair]:
Ground Truth: {ref}
Model Prediction: {pred}

Question: Does the Model Prediction match the semantic meaning of the Ground Truth? 
Answer with exactly one word: 'Yes' or 'No'. Do not include any explanations or punctuation."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 4. 纯文本推理流程（不传入任何 Image/Video）
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # 仅处理文本输入
        model_inputs = processor(text=[text], return_tensors="pt").to(DEVICE_ID)
        
        with torch.no_grad():
            # 严格限制最多生成2个Token，关闭采样保证答案确定
            generated_ids = model.generate(**model_inputs, max_new_tokens=2, do_sample=False)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
            response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            response = response.strip().lower().replace(".", "")
            
        # 5. 判定裁判结果
        judge = "No"
        if "yes" in response:
            llm_correct += 1
            judge = "Yes"
            
        # 将结果写入日志
        data['judge_result'] = judge
        data['raw_judge_response'] = response
        output_f.write(json.dumps(data, ensure_ascii=False) + "\n")

    output_f.close()

    # ==========================================
    # 6. 打印学术测评报告
    # ==========================================
    print("\n" + "="*60)
    print("🤖 MSVD-QA 测试集 - 方案 B (Qwen3-VL 纯文本裁判) 测评报告")
    print("="*60)
    print(f" 🔹 总计评估问答对数量 (Total QAs)   : {total}")
    print(f" 🔹 模型高确定性安全拒绝数 (Refused)  : {refused_count} 组 (占比: {refused_count/total*100:.2f}%)")
    print("-" * 60)
    print(f" ❌ 传统严格字面准确率 (Exact Match)  : {em_correct/total*100:.2f}%")
    print(f" 🎯 方案B 大模型裁判准确率 (LLM-Judge) : {llm_correct/(total - refused_count)*100:.2f}% (排除拒绝样本)")
    print(f" 🎯 方案B 考虑安全拒绝的广义准确率      : {llm_correct/total*100:.2f}% (包含拒绝样本)")
    print("="*60)
    print(f"💾 详细判卷日志已保存至: {output_file}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()