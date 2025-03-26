# 运行采用qwen-math官方仓库，这里只用eval
filename = "outputs/Qwen/Qwen2.5-Math-72B-Instruct/math_eval/aqua/test_qwen25-math-cot_-1_seed0_t0.0_s0_e-1.jsonl"
import jsonlines
from utils import aqua_utils

from reasoners.benchmark import AQuAEvaluator

evaluator = AQuAEvaluator(
    output_extractor=aqua_utils.retrieve_chat_aqua_answer,
    answer_extractor=lambda x: aqua_utils.retrieve_answer_from_dataset(x["answer"]),
    sample_prompt_type="rap",
    dataset_path="data/AQuA/test.json",
)


def eval_acc(output, answer):
    return output == answer


dataset = list(evaluator.full_dataset)
total = 0
right = 0
with jsonlines.open(filename, mode="r") as reader:
    # 遍历文件中的每一行，将其作为字典对象读取
    for i, obj in enumerate(reader):
        total += 1
        try:
            pred = evaluator.output_extractor(obj["code"][0], dataset[i]["options"])
            answer = evaluator.answer_extractor(dataset[i])
            right += eval_acc(pred, answer)

        except:
            pass
print(right / total)
