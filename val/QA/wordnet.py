import json
import os
import nltk
from nltk.corpus import wordnet

# ==========================================
# 一定输入各保存结果qa_comparison_minimal.jsonl文件
# ==========================================


WORDNET_PATH = '/data3/health/wyy/BLIP2/experiment/data/wordnet/'
nltk.data.path.append(WORDNET_PATH)

def get_synonyms(word):
    """
    通过 WordNet 获取一个英文单词的所有同义词
    """
    synonyms = set()
    try:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().lower().replace('_', ' '))
    except Exception:
        pass
    return synonyms

def is_synonym_or_equivalent(pred, ref):
    """
    核心评测逻辑：判断预测答案(pred)与标准答案(ref)是否在语义上等价或包含
    """
    pred = str(pred).strip().lower().replace(".", "").replace(",", "")
    ref = str(ref).strip().lower().replace(".", "").replace(",", "")
    
    # 规则 1：基础清理后字面完全一致
    if pred == ref:
        return True
        
    # 规则 2：泛化人称代词与具体身份的蕴含兼容
    person_words = {'person', 'someone', 'somebody', 'anybody', 'anyone', 'people', 'human'}
    man_woman_words = {'man', 'woman', 'boy', 'girl', 'kid', 'child', 'guy', 'lady'}
    if pred in person_words and ref in man_woman_words:
        return True
    if ref in person_words and pred in man_woman_words:
        return True

    # 规则 3：WordNet 严格词林同义词泛化匹配
    pred_synonyms = get_synonyms(pred)
    if ref in pred_synonyms:
        return True
        
    # 规则 4：子串包络或核心词包含关系补漏
    if len(pred) > 1 and len(ref) > 1:
        if pred in ref or ref in pred:
            return True
            
    return False

# ==========================================
# 2. 评测主流程 (完美适配 JSONL 格式)
# ==========================================
if __name__ == "__main__":
    # 指向代码2生成的全新最小化比对文件 (JSONL 格式)
    eval_file_path = "/data3/health/wyy/BLIP2/experiment/Ablation_experiment/without_yolo_information/QA/qa_comparison_results/qa_comparison_minimal.jsonl" 
    
    total = 0
    em_correct = 0
    scheme_a_correct = 0
    refused_count = 0
    
    if not os.path.exists(eval_file_path):
        print(f"❌ 未找到待评测文件: {eval_file_path}，请检查路径。")
        exit(1)

    print(f"🚀 开始逐行读取 JSONL 文件 [{eval_file_path}] 并进行评估...")
    
    with open(eval_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"⚠️ 第 {line_num} 行 JSON 解析失败，已跳过。内容: {line[:30]}...")
                continue
            
            # 成功读取到一行有效的 JSON 对象，total 加 1
            total += 1
            
            # 【关键修正】适配新 JSONL 格式的 Key
            pred = data.get('model_answer', '')
            ref = data.get('gt_answer', '')
            
            # 【鲁棒性优化】强转为字符串，防止由于 None 类型导致 lower() 崩溃
            pred_str = str(pred).lower()
            
            # 统计模型遵循高安全性机制，主动拒绝回答的样本
            if "visual anchor" in pred_str or "no corresponding" in pred_str or "unknown" in pred_str or pred is None:
                refused_count += 1
                continue
            
            # 1. 传统 Exact Match (EM) 评估
            if str(pred).strip().lower() == str(ref).strip().lower():
                em_correct += 1
                
            # 2. 方案 A 同义词与蕴含宽松评估
            if is_synonym_or_equivalent(pred, ref):
                scheme_a_correct += 1

    # ==========================================
    # 3. 输出多维评测报告 (带安全分母检查)
    # ==========================================
    print("\n" + "="*60)
    print("📊 MSVD-QA 测试集 - 方案 A 宽松语义多维测评报告 (JSONL 适配版)")
    print("="*60)
    print(f" 🔹 总计有效读取问答对数量 (Total QAs) : {total}")
    
    # 计算拒绝率时进行安全的分母检查
    refused_rate = (refused_count / total * 100) if total > 0 else 0.0
    print(f" 🔹 模型高确定性安全拒绝数 (Refused)  : {refused_count} 组 (占比: {refused_rate:.2f}%)")
    print("-" * 60)
    
    valid_total = total - refused_count
    if total > 0 and valid_total > 0:
        print(f" ❌ 传统严格字面 EM 准确率 (Strict EM) : {em_correct / valid_total * 100:.2f}%  ({em_correct}/{valid_total})")
        print(f" ✅ 方案 A 宽松语义准确率 (Scheme A EM): {scheme_a_correct / valid_total * 100:.2f}%  ({scheme_a_correct}/{valid_total})")
        print(f" 📈 相对传统指标净绝对提升 (Gain)      : +{(scheme_a_correct - em_correct) / valid_total * 100:.2f}%")
    else:
        print(" ⚠️ 警告：没有有效的可评估样本！")
        print(f"    - 当前读取到的有效数据行数为: {total}")
        print(f"    - 请确保当前运行路径下存在非空的 {eval_file_path}")
    print("="*60 + "\n")