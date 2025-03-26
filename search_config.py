import io
import re
from typing import TypedDict, Optional
import numpy as np

from world_model import GSM8kState, GSM8kAction, GSM8kPromptDict
from reasoners import SearchConfig
from copy import deepcopy


class GSM8kUsefulPrompt(TypedDict):
    input: str
    question_prefix: str
    subquestion_prefix: str
    new_subquestion_prefix: str
    useful_prefix: str


class GSM8kConfig(SearchConfig):
    def __init__(
        self,
        base_model=None,
        useful_prompt=None,
        n_actions=4,
        batch_size=1,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        reward_alpha=0.5,
        reward_confidence_default=0.8,
        depth_limit=5,
        force_terminating_on_depth_limit=True,
        force_overall_prompt_on_overall_question=True,
        force_overall_question_on_overall_prompt=True,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.useful_prompt = useful_prompt
        self.example = ""
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.n_actions = n_actions
        self.force_terminating_on_depth_limit = force_terminating_on_depth_limit
        self.depth_limit = depth_limit
        self.reward_alpha = reward_alpha
        self.reward_confidence_default = reward_confidence_default
        self.force_overall_prompt_on_overall_question = (
            force_overall_prompt_on_overall_question
        )
        self.force_overall_question_on_overall_prompt = (
            force_overall_question_on_overall_prompt
        )
        self.overall_question: Optional[str] = None
        self.prompt_examples = ""
        self.n_shots = 0
        self.question_prefix = "Question"
        self.answer_prefix = "Answer"

    def update_prompt_examples(self, example: str) -> None:
        with io.StringIO() as f:
            f.write(self.prompt["instruction"] + "\n\n")
            for idx, example in enumerate(self.prompt["interactive_examples"]):
                f.write(example.format(idx=idx + 1) + "\n\n")
            self.n_shots = len(self.prompt["interactive_examples"])
            self.prompt_examples = f.getvalue()

    def update_example(self, example: str, prompt: GSM8kPromptDict = None) -> None:
        super().update_example(example, prompt=prompt)

        assert prompt is not None
        self.prompt = prompt
        self.update_prompt_examples(example)
        if (
            self.force_overall_prompt_on_overall_question
            or self.force_overall_question_on_overall_prompt
        ):
            # print(self.example)
            # self.overall_question = re.match('.*((Calculate|calculate|how|How|what|What|Find|find|True or false).*)$',
            #                                  self.example, flags=re.DOTALL)[1]
            self.overall_question = re.match(
                ".*((([A-Z].* (calculate|how|what|find|true or false))|((Calculate|How|What|Find|True or false))).*)$",
                self.example,
                flags=re.DOTALL,
            )[1]

    def get_actions_model_input(
        self,
        state: GSM8kState,
    ):
        with io.StringIO() as f:
            f.write(self.prompt_examples)
            f.write(
                self.prompt["question_prefix"].format(
                    idx=self.n_shots + 1, question=self.example
                )
                + "\n"
            )
            for idx, t in enumerate(state):
                q, a = t[:2]
                f.write(
                    self.prompt["subquestion_prefix"].format(
                        idx=self.n_shots + 1, sub_idx=idx + 1
                    )
                    + " "
                    + q
                    + "\n"
                )
                f.write(
                    self.prompt["answer_prefix"].format(
                        idx=self.n_shots + 1, sub_idx=idx + 1
                    )
                    + " "
                    + a
                    + "\n"
                )
            f.write(
                self.prompt["subquestion_prefix"].format(
                    idx=self.n_shots + 1, sub_idx=len(state) + 1
                )
            )
            if (
                at_depth_limit := self.force_terminating_on_depth_limit
                and len(state) + 1 >= self.depth_limit
            ):
                f.write(" " + self.prompt["overall_question_prefix"])
            model_input = f.getvalue()
        return model_input, at_depth_limit

    def get_actions_inputs(
        self,
        state: GSM8kState,
    ) -> list[GSM8kAction]:
        model_input, at_depth_limit = self.get_actions_model_input(state)
        n_actions = 1 if at_depth_limit else self.n_actions
        n_samples = 0
        for idx in range(0, n_actions, self.batch_size):
            n_samples += min(n_actions - idx, self.batch_size)

        return n_samples, model_input, at_depth_limit

    def get_actions_output(self, outputs, at_depth_limit):
        outputs = [output.strip() for output in outputs]
        if at_depth_limit:
            outputs = [
                self.prompt["overall_question_prefix"] + " " + output
                for output in outputs
            ]
        if self.force_overall_question_on_overall_prompt:
            for i, output in enumerate(outputs):
                if self.prompt["overall_question_prefix"] in output:
                    outputs[i] = (
                        self.prompt["overall_question_prefix"]
                        + " "
                        + self.overall_question
                    )
        if self.force_overall_prompt_on_overall_question:
            for i, output in enumerate(outputs):
                if self.overall_question.lower() == output.lower():
                    outputs[i] = (
                        self.prompt["overall_question_prefix"]
                        + " "
                        + self.overall_question
                    )

        # set does not guarantee order, but dict does guarantee
        # we cannot use set here because torch.distributed in LLaMA requires the same order across all processes
        outputs = list(dict.fromkeys(outputs))
        return outputs

    def get_actions(
        self,
        state: GSM8kState,
    ) -> list[GSM8kAction]:
        with io.StringIO() as f:
            f.write(self.prompt_examples)
            f.write(
                self.prompt["question_prefix"].format(
                    idx=self.n_shots + 1, question=self.example
                )
                + "\n"
            )
            for idx, (q, a, _) in enumerate(state):
                f.write(
                    self.prompt["subquestion_prefix"].format(
                        idx=self.n_shots + 1, sub_idx=idx + 1
                    )
                    + " "
                    + q
                    + "\n"
                )
                f.write(
                    self.prompt["answer_prefix"].format(
                        idx=self.n_shots + 1, sub_idx=idx + 1
                    )
                    + " "
                    + a
                    + "\n"
                )
            f.write(
                self.prompt["subquestion_prefix"].format(
                    idx=self.n_shots + 1, sub_idx=len(state) + 1
                )
            )
            if (
                at_depth_limit := self.force_terminating_on_depth_limit
                and len(state) + 1 >= self.depth_limit
            ):
                f.write(" " + self.prompt["overall_question_prefix"])
            model_input = f.getvalue()

        # print(model_input)
        # input(">")

        n_actions = 1 if at_depth_limit else self.n_actions
        temperature = 0 if at_depth_limit else self.temperature
        outputs = []
        for idx in range(0, n_actions, self.batch_size):
            n_samples = min(n_actions - idx, self.batch_size)
            outputs += self.base_model.generate(
                [model_input] * n_samples,
                hide_input=True,
                do_sample=True,
                temperature=temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                eos_token_id="\n",
            ).text

        outputs = [output.strip() for output in outputs]
        if at_depth_limit:
            outputs = [
                self.prompt["overall_question_prefix"] + " " + output
                for output in outputs
            ]
        if self.force_overall_question_on_overall_prompt:
            for i, output in enumerate(outputs):
                if self.prompt["overall_question_prefix"] in output:
                    outputs[i] = (
                        self.prompt["overall_question_prefix"]
                        + " "
                        + self.overall_question
                    )
        if self.force_overall_prompt_on_overall_question:
            for i, output in enumerate(outputs):
                if self.overall_question.lower() == output.lower():
                    outputs[i] = (
                        self.prompt["overall_question_prefix"]
                        + " "
                        + self.overall_question
                    )

        # set does not guarantee order, but dict does guarantee
        # we cannot use set here because torch.distributed in LLaMA requires the same order across all processes
        outputs = list(dict.fromkeys(outputs))
        return outputs

    def get_fast_reward_input(
        self, state: GSM8kState, action: GSM8kAction
    ) -> tuple[float, dict]:
        with io.StringIO() as f:
            f.write(self.useful_prompt["input"])
            f.write(self.useful_prompt["question_prefix"] + self.example + "\n")
            for idx, t in enumerate(state):
                q = t[0]
                f.write(
                    self.useful_prompt["subquestion_prefix"].format(idx + 1)
                    + " "
                    + q
                    + "\n"
                )
            f.write(
                self.useful_prompt["new_subquestion_prefix"].format(len(state) + 1)
                + " "
                + action
                + "\n"
            )
            f.write(self.useful_prompt["useful_prefix"])
            model_input = f.getvalue()
        return model_input

    def get_fast_reward_output(self, logits):
        probs = np.exp(logits) / np.sum(np.exp(logits) + 1e-9)
        useful_prob = probs[0]
        fast_reward, _ = self.calculate_reward(useful_prob)
        return fast_reward, {"r_useful": useful_prob}

    def fast_reward(self, state: GSM8kState, action: GSM8kAction) -> tuple[float, dict]:
        with io.StringIO() as f:
            f.write(self.useful_prompt["input"])
            f.write(self.useful_prompt["question_prefix"] + self.example + "\n")
            for idx, (q, _, _) in enumerate(state):
                f.write(
                    self.useful_prompt["subquestion_prefix"].format(idx + 1)
                    + " "
                    + q
                    + "\n"
                )
            f.write(
                self.useful_prompt["new_subquestion_prefix"].format(len(state) + 1)
                + " "
                + action
                + "\n"
            )
            f.write(self.useful_prompt["useful_prefix"])
            model_input = f.getvalue()

        logits = self.base_model.get_next_token_logits(model_input, ["Yes", "No"])[0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        useful_prob = probs[0]
        fast_reward, _ = self.calculate_reward(useful_prob)
        return fast_reward, {"r_useful": useful_prob}

    def calculate_reward(self, r_useful, r_conf=None):
        if r_conf is None:
            r_conf = self.reward_confidence_default
        return r_useful**self.reward_alpha * r_conf ** (1 - self.reward_alpha), {
            "r_useful": r_useful,
            "r_conf": r_conf,
        }

    def reward(
        self, state, action, r_useful: float = None, confidence: float = 1
    ) -> tuple[float, dict]:
        # return confidence, {'r_conf': confidence}
        assert r_useful is not None, (
            "useful_reward is required to calculate reward in this search config, consider passing it in fast_reward"
        )
        assert confidence is not None, (
            "confidence is required to calculate reward in this search config, consider passing it in world model's step"
        )
        return self.calculate_reward(r_useful, confidence)

    def get_finnal_question(self, question=""):
        if question == "":
            return "Now we can answer the question "
        return "Now we can answer the question " + question.split(".")[-1]


class CodeGSM8kConfig(GSM8kConfig):
    def update_prompt_examples(self, example) -> None:
        self.prompt_examples = []
        for idx, example in enumerate(self.prompt["interactive_examples"]):
            for i in range(len(example)):
                if "{idx}" in example[i]["content"]:
                    example[i]["content"] = example[i]["content"].format(idx=idx + 1)
            self.prompt_examples.extend(example)
        self.n_shots = len(self.prompt["interactive_examples"])

    def get_actions_model_input(
        self,
        state: GSM8kState,
    ):
        model_input = deepcopy(self.prompt_examples)
        model_input.append(
            {
                "role": "user",
                "content": self.prompt["question_prefix"].format(
                    idx=self.n_shots + 1, question=self.example
                ),
            }
        )
        model_input.append(self.prompt["get_var_prompt"])
        model_input.append(
            {
                "role": "assistant",
                "content": "```python"
                + self.prompt["known_variables_generate"]
                + "\n```",
            }
        )
        idx = 0
        for t in state:
            q, a = t[:2]
            model_input.append(
                {
                    "role": "user",
                    "content": self.prompt["subquestion_prefix"].format(
                        idx=self.n_shots + 1, sub_idx=idx + 1
                    ),
                }
            )
            model_input.append({"role": "assistant", "content": q})
            model_input.append(self.prompt["answer_prefix"])
            model_input.append({"role": "assistant", "content": a})
            idx += 1
        model_input.append(
            {
                "role": "user",
                "content": self.prompt["subquestion_prefix"].format(
                    idx=self.n_shots + 1, sub_idx=idx + 1
                ),
            }
        )
        if (
            at_depth_limit := self.force_terminating_on_depth_limit
            and len(state) + 1 >= self.depth_limit
        ):
            model_input.append(
                {"role": "assistant", "content": self.prompt["overall_question_prefix"]}
            )
        return model_input, at_depth_limit


class ChatGSM8kConfig(GSM8kConfig):
    def update_prompt_examples(self, example) -> None:
        self.prompt_examples = []
        for idx, example in enumerate(self.prompt["interactive_examples"]):
            self.prompt_examples.extend(example)
        self.n_shots = len(self.prompt["interactive_examples"])

    def get_actions_model_input(
        self,
        state: GSM8kState,
    ):
        model_input = deepcopy(self.prompt_examples)
        model_input.append(self.prompt["instruction"])
        model_input.append({"role": "user", "content": self.example})

        reasoning_path, idx = self.prompt["answer_prefix"], 1
        for t in state:
            q, a = t[:2]
            reasoning_path += self.prompt["question_prefix"] % idx + q + "\n"
            reasoning_path += a + "\n\n"
            idx += 1

        if (
            at_depth_limit := self.force_terminating_on_depth_limit
            and len(state) + 1 >= self.depth_limit
        ):
            reasoning_path += (
                self.prompt["question_prefix"] % idx
                + self.prompt["overall_question_prefix"]
            )
        else:
            reasoning_path += self.prompt["question_prefix"] % idx
        model_input.append({"role": "assistant", "content": reasoning_path})
        return model_input, at_depth_limit

    def calculate_reward(self, r_useful, r_conf=None):
        if r_conf is None:
            r_conf = self.reward_confidence_default
        return r_useful + r_conf, {"r_useful": r_useful, "r_conf": r_conf}


class ChatCodeGSM8kConfig(ChatGSM8kConfig):
    def get_actions_model_input(
        self,
        state: GSM8kState,
    ):
        model_input = deepcopy(self.prompt_examples)
        model_input.append(self.prompt["instruction"])
        model_input.append({"role": "user", "content": self.example})

        reasoning_path, idx = self.prompt["inital_variable"], 2
        for t in state:
            q, a = t[:2]
            reasoning_path += self.prompt["question_prefix"] % idx + q + "\n"
            reasoning_path += a + "\n\n"
            idx += 1

        if (
            at_depth_limit := self.force_terminating_on_depth_limit
            and len(state) + 1 >= self.depth_limit
        ):
            reasoning_path += (
                self.prompt["question_prefix"] % idx
                + self.prompt["overall_question_prefix"]
            )
        else:
            reasoning_path += self.prompt["question_prefix"] % idx
        model_input.append({"role": "assistant", "content": reasoning_path})
        return model_input, at_depth_limit


class Weak12KConfig(ChatGSM8kConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_prefix = "问题"
        self.answer_prefix = "答案"
        self.force_overall_prompt_on_overall_question = False
        self.force_overall_question_on_overall_prompt = False

    def get_finnal_question(self, question=""):
        if question == "":
            return "现在我们可以回答这个问题:"
        question = question.replace(",", ".").split(".")[-1]
        return "现在我们可以回答这个问题:" + question


class CodeWeak12KConfig(CodeGSM8kConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_prefix = "问题"
        self.answer_prefix = "答案"
        self.force_overall_prompt_on_overall_question = False
        self.force_overall_question_on_overall_prompt = False

    def get_finnal_question(self, question=""):
        if question == "":
            return "现在我们可以回答这个问题:"
        question = question.replace(",", ".").split(".")[-1]
        return "现在我们可以回答这个问题:" + question


class DeepMctsConfig:
    def __init__(self, partial_order=[False] * 5, use_plan_set=False, **kwargs):
        super().__init__(**kwargs)
        self.partial_order = partial_order
        self.use_plan_set = use_plan_set

    def update_example(self, example, prompt) -> None:
        super(GSM8kConfig, self).update_example(example, prompt)
        # self.prompt["code_actions"] = []
        # 发现plan集没有用，决定去掉
        self.prompt["actions"] = (
            self.prompt["understand_actions"] + self.prompt["reflect_actions"]
        )  # +self.prompt["plan_actions"]
        if self.use_plan_set:
            self.prompt["actions"] += self.prompt["plan_actions"]

    def pop_action(self, action, candidates: list):
        if action not in candidates:
            return candidates
        actions = deepcopy(candidates)
        actions.pop(actions.index(action))
        return actions

    def get_actions(self, state):
        # 必须规则，第一个必须是understa，最后一个必须是summary
        if len(state) + 1 >= self.depth_limit:
            return self.prompt["summar_prompt"]
        elif len(state) == 0:
            return self.prompt["understand_actions"]
        if isinstance(self.partial_order, list) and any(self.partial_order):
            """

            必须的规则是，最后一个必须是summary，第一个必须是understand
            非必须规则 
            1.不能有两次连续的动作
            2.至少出现一次reflect
            3.最大迭代一半后只有反思和代码
            4.不能有连续两次代码
            5.如果执行到一半，还没执行过代码，强制执行一次
            partial_order是一个列表，形如[True,True,False,False，False],分别对应上面的4条规则是否被启动
            """
            actions = [t.sub_question for t in state]
            last_action = actions[-1]
            candidates = deepcopy(self.prompt["actions"])
            # 规则5 如果执行到一半，还没执行过代码，强制执行一次
            if (
                len(state) == self.depth_limit // 2 + 1
                and not any(
                    [action in self.prompt["code_actions"] for action in actions]
                )
                and self.partial_order[4]
            ):
                return self.prompt["code_actions"]
            # 规则3 最大迭代一半后只有反思和代码
            if len(state) >= self.depth_limit // 2 and self.partial_order[2]:
                candidates = deepcopy(self.prompt["reflect_actions"])
            else:
                candidates = deepcopy(self.prompt["actions"])
            # 规则4，不能有连续两次代码
            if (
                last_action not in self.prompt["code_actions"]
                or not self.partial_order[3]
            ):
                candidates += self.prompt["code_actions"]
            # 规则2，至少出现一次reflect
            if (
                any([action in self.prompt["reflect_actions"] for action in actions])
                or not self.partial_order[1]
            ):
                candidates += self.prompt["summar_prompt"]
            elif len(state) + 1 >= self.depth_limit - 1:
                candidates = self.prompt["reflect_actions"]
            # 规则1 不能有两次连续的动作
            if self.partial_order[0]:
                return self.pop_action(last_action, candidates)
            return candidates

        else:
            return (
                self.prompt["code_actions"]
                + self.prompt["actions"]
                + self.prompt["summar_prompt"]
            )

    def get_fast_reward_input(self, state, action) -> tuple[float, dict]:
        model_input = []
        model_input.append(deepcopy(self.prompt["instruction"]))
        model_input.append({"role": "user", "content": self.example})

        reasoning_path, idx = "", 1
        for t in state:
            q, a = t[:2]
            reasoning_path += self.prompt["question_prefix"] % (q) + "\n"
            reasoning_path += a + "\n\n"
            idx += 1
        reasoning_path += self.prompt["question_prefix"] % (action)
        model_input.append({"role": "assistant", "content": reasoning_path})
        return model_input


class DeepWeak12KConfig(DeepMctsConfig, Weak12KConfig):
    pass


class DeepGSM8kConfig(DeepMctsConfig, ChatGSM8kConfig):
    pass
