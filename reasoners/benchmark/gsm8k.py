import datasets
import json
import random
import copy
from reasoners.base import Evaluator
import string
import pandas as pd

class GSM8KEvaluator(Evaluator):
    def __init__(
        self,
        init_prompt=None,
        disable_log=False,
        disable_tqdm=False,
        file_path="data/Gsm8k",
        sample_prompt_type="l2m",
        split="test",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.init_prompt = init_prompt

        self.input_processor = lambda x: x["question"]
        self.split = split
        self.full_dataset = datasets.load_dataset(file_path, split=split)
        self._dataset_name = "gsm8k"
        self.disable_log = disable_log
        self.disable_tqdm = disable_tqdm
        self.sample_prompt_type = sample_prompt_type

    def sample_prompt(self, shuffle_prompt=True, num_shot=4):
        sample_prompt_type = self.sample_prompt_type
        if sample_prompt_type == "l2m":
            prompt = {}
            if shuffle_prompt:
                decomp_examples = random.sample(
                    self.init_prompt["decomposition_pool"], num_shot
                )
                solv_examples = random.sample(
                    self.init_prompt["solving_pool"], num_shot
                )
            else:
                decomp_examples = self.init_prompt["decomposition_pool"][:num_shot]
                solv_examples = self.init_prompt["solving_pool"][:num_shot]
            prompt["decomposition"] = (
                "".join(decomp_examples) + self.init_prompt["composition_prefix"]
            )
            prompt["overall"] = (
                "".join(decomp_examples) + self.init_prompt["overall_prefix"]
            )
            prompt["solving"] = (
                "".join(solv_examples) + self.init_prompt["solving_prefix"]
            )

        elif sample_prompt_type == "cot":
            prompt = {}
            if shuffle_prompt:
                examples = random.sample(self.init_prompt["cot_pool"], num_shot)
            else:
                examples = self.init_prompt["cot_pool"][:num_shot]
            prompt["cot"] = "".join(examples) + self.init_prompt["prefix"]

        elif sample_prompt_type == "rap":
            ret = copy.deepcopy(self.init_prompt)
            if num_shot == 0:
                ret["interactive_examples"] = []
            else:
                ret["interactive_examples"], ret["useful_examples"] = zip(
                    *random.sample(
                        list(zip(ret["interactive_examples"], ret["useful_examples"])),
                        k=num_shot,
                    )
                )
            return ret

        elif sample_prompt_type == "grace":
            return None

        else:
            raise NotImplementedError
        return prompt

    def eval_output(self, answer, output):
        if output is None:
            return False
        try:
            output = int(output)
            answer = int(answer)
            return output == answer
        except ValueError:
            pass
        try:
            output = float(output)
            answer = float(answer)
            return output == answer
        except ValueError:
            pass
        return output == answer


import jsonlines


class SvampEvaluator(GSM8KEvaluator):
    def __init__(self, file_path, init_prompt=None, **kwargs):
        super(GSM8KEvaluator, self).__init__(**kwargs)
        self._dataset_name = "svamp"
        self.full_dataset = []
        with jsonlines.open(file_path, "r") as file:
            for t in file:
                for key in list(t.keys()):
                    if key.lower() not in t.keys():
                        t[key.lower()] = t[key]
                if t["body"][-1] != ".":
                    t["body"] += " ."
                self.full_dataset.append(
                    {
                        "finnal_question": t["question"],
                        "question": t["body"] + " " + t["question"],
                        "answer": t["answer"],
                        "equation": t["equation"],
                    }
                )
        self.split = "test" if "test" in file_path else "train"
        self.sample_prompt_type = "rap"
        self.init_prompt = init_prompt


def load_weak12k(filename):
    datas = []
    with open(filename, "r", encoding="utf-8") as load_f:
        load_dict = json.load(load_f)
    for t in load_dict:
        question = t["original_text"][:-1] + "?"
        datas.append(
            {
                "question": question.replace("，", ",")
                .replace("。", ".")
                .replace("．", ",")
                .replace("？", "?"),
                "answer": t["answer"],
            }
        )
        datas[-1]["finnal_question"] = (
            datas[-1]["question"].replace(",", ".").split(".")[-1]
        )
    return datas


class Weak12kEvaluator(GSM8KEvaluator):
    def __init__(
        self, filename, init_prompt=None, sample_prompt_type="rap", **kwargs
    ) -> None:
        super(GSM8KEvaluator, self).__init__(**kwargs)
        self.sample_prompt_type = sample_prompt_type
        self._dataset_name = "weak12k"
        self.full_dataset = load_weak12k(filename)
        self.init_prompt = init_prompt


class CMathEvaluator(GSM8KEvaluator):
    def __init__(
        self, filename, init_prompt=None, sample_prompt_type="rap", **kwargs
    ) -> None:
        super(GSM8KEvaluator, self).__init__(**kwargs)
        self.sample_prompt_type = sample_prompt_type

        self.init_prompt = init_prompt
        self._dataset_name = filename.replace("\\", "/").split("/")[-2]
        self.full_dataset = []
        with jsonlines.open(filename, "r") as file:
            for t in file:
                t["answer"] = eval(t["answer"].replace("%", "/100"))
                self.full_dataset.append(t)


class MathEvaluator(GSM8KEvaluator):
    def __init__(
        self, filename, init_prompt=None, sample_prompt_type="rap", **kwargs
    ) -> None:
        super(GSM8KEvaluator, self).__init__(**kwargs)
        self.sample_prompt_type = sample_prompt_type

        self.init_prompt = init_prompt
        self._dataset_name = filename.replace("\\", "/").split("/")[-2]
        self.full_dataset = []
        with jsonlines.open(filename, "r") as file:
            for t in file:
                if "problem" in t.keys() and "question" not in t.keys():
                    t["question"] = t["problem"]
                self.full_dataset.append(t)


class GaokaoEvaluator(GSM8KEvaluator):
    def __init__(
        self, filename, init_prompt=None, sample_prompt_type="rap", **kwargs
    ) -> None:
        super(GSM8KEvaluator, self).__init__(**kwargs)
        self.sample_prompt_type = sample_prompt_type

        self.init_prompt = init_prompt
        self._dataset_name = filename.replace("\\", "/").split("/")[-2]
        self.full_dataset = []
        with jsonlines.open(filename, "r") as file:
            for t in file:
                t["question"] += "\n选项是:%s" % str(t["options"])
                self.full_dataset.append(t)


class MMLUEvaluator(GSM8KEvaluator):
    def __init__(
        self, filename, init_prompt=None, sample_prompt_type="rap", **kwargs
    ) -> None:
        super(GSM8KEvaluator, self).__init__(**kwargs)
        self.sample_prompt_type = sample_prompt_type

        self.init_prompt = init_prompt
        self._dataset_name = filename.replace("\\", "/").split("/")[-2]
        self.full_dataset = []
        with jsonlines.open(filename, "r") as file:
            for t in file:
                options = []
                for i in range(len(t["choices"])):
                    options.append(string.ascii_uppercase[i] + "." + t["choices"][i])
                t = {
                    "question": t["question"] + "\nOptions:\n" + "\n".join(options),
                    "options": options,
                    "answer": string.ascii_uppercase[t["answer"]],
                }
                self.full_dataset.append(t)
                
class Game24Evaluator(GSM8KEvaluator):
    def __init__(
        self, filename, init_prompt=None, sample_prompt_type="rap", **kwargs
    ) -> None:
        super(GSM8KEvaluator, self).__init__(**kwargs)
        self.sample_prompt_type = sample_prompt_type

        self.init_prompt = init_prompt
        self._dataset_name = filename.replace("\\", "/").split("/")[-2]
        self.full_dataset = pd.read_csv(filename).values[:,1]


