实验中使用的模型及版本如下：
    blip2-opt-6.7b
    yoloe-v8l-seg.pt
    Qwen3-VL-8B-Instruct
请自行下载并存好路径，在后续代码中自行修改模型路径。

<B>注意！代码中的所有路径均需按照实际情况进行修改！</B>

文件解释：
data：包含msrvtt数据集和msvd-qa数据集的<u>处理后状态</u>，以及nltk_data库中的wordnet单词库。<B>（不包括msrvtt数据集和msvd-qa数据集的原始视频及标签文件）</B>
      msrvtt和msvd-qa文件中，keyframe文件夹存储边缘端实验后，筛选出的视频关键帧图像；strctured_reports文件夹存储使用脚本处理后的结构化BLIP2文本和YOLO检测日志文本；videos_analysis_results.jsonl为边缘端输出的原始记录。

edge_experiment: 边缘端实验，用于筛选视频关键帧图像，生成blip2语义文本和YOLO检测日志。
                 edge_BLIP2_videos.py 边缘端实验流程代码，运行结束后将在data的对应数据集文件夹中生成keyframe文件夹和videos_analysis_results.jsonl原始记录。
                 process_edge_output.py 处理原始记录文件，运行后生成structured_reports文件夹。

center_experiment: 中心端实验，包括对比实验和消融实验。
    Ablation_experiment：消融实验文件夹，包括full_model, only_keyframe, without_summary_information, without_yolo_information四种消融配置。
        每种配置中包含两个任务的内容。description中，.py代码用于中心端Qwen3-VL-8B-Instruct模型的描述任务推理，qwen3_vl_results文件夹存储模型推理的元信息Jsonl文件和结果Jsonl文件；QA中，py代码用于中心端Qwen3-VL-8B-Instruct模型的QA任务推理，qwen3_vl_results文件夹存储模型推理的元信息Jsonl文件和结果Jsonl文件，qa_comparison_results文件夹用于存储提取关键答案后的jsonl文件，便于后续量化评估。
    Comparative_experiment：对比实验文件夹，包括full_model和only_Qwen3. full_model同消融实验中的full_model；only_Qwen3存储直接用Qwen3-VL-8B-Instruct模型处理原始视频的结果。

val: 评估代码。包含description和QA两个任务的量化评估代码。
    description中，BERTScore.py用于计算描述答案与标签中caption的BERTScore F1；four.py用于计算描述答案与标签中caption的BLEU, METEOR, OUGE-L, CIDEr四个指标。
    QA中，BERTScore.py用于计算问答答案与标签中的BERTScore F1；LLM_test.py大模型对问答答案与标签的相似程度的评估（使用Qwen3-8B-Instruct）；wordnet.py是wordnet库对问答答案与标签的相似程度的评估。

main_path_single: 单个视频的单次推理示例。