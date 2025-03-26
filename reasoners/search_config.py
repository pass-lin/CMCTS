import io
import re
from typing import TypedDict, Optional
import numpy as np

from world_model import GSM8kState, GSM8kAction, GSM8kPromptDict
from reasoners import SearchConfig, LanguageModel


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

    def update_example(self, example: str, prompt: GSM8kPromptDict = None) -> None:
        super().update_example(example, prompt=prompt)

        assert prompt is not None
        self.prompt = prompt
        with io.StringIO() as f:
            f.write(self.prompt["instruction"] + "\n\n")
            for idx, example in enumerate(self.prompt["interactive_examples"]):
                f.write(example.format(idx=idx + 1) + "\n\n")
            self.n_shots = len(self.prompt["interactive_examples"])
            self.prompt_examples = f.getvalue()

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

    def get_actions_inputs(
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

        # print(model_input)
        # input(">")

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
        probs = np.exp(logits) / np.sum(np.exp(logits))
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


class Math23kConfig(GSM8kConfig):
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


class AQuAUsefulPrompt(TypedDict):
    input: str
    question_prefix: str
    subquestion_prefix: str
    new_subquestion_prefix: str
    useful_prefix: str


from world_model import MATHState, MATHAction, MATHPromptDict


class MATHConfig(GSM8kConfig):
    def __init__(
        self,
        base_model: LanguageModel,
        useful_prompt: dict,
        n_actions=4,
        batch_size=1,
        temperature=0.8,
        reward_alpha=0,
        reward_confidence_default=0.8,
        depth_limit=5,
        force_terminating_on_depth_limit=True,
        force_overall_prompt_on_overall_question=True,
        force_overall_question_on_overall_prompt=True,
    ) -> None:
        super(GSM8kConfig, self).__init__()
        self.base_model = base_model
        self.useful_prompt: AQuAUsefulPrompt = useful_prompt
        self.example = ""
        self.batch_size = batch_size
        self.question_prefix = "Question"
        self.answer_prefix = "Answer"
        self.temperature = temperature
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

    def get_actions_inputs(
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
            for idx, (q, a, *_) in enumerate(state):
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
        n_samples = 0
        for idx in range(0, n_actions, self.batch_size):
            n_samples += min(n_actions - idx, self.batch_size)

        return n_samples, model_input, at_depth_limit

    def update_example(self, example: str, prompt: MATHPromptDict = None) -> None:
        super().update_example(example, prompt=prompt)
        assert prompt is not None
        self.prompt = prompt
        with io.StringIO() as f:
            f.write(self.prompt["instruction"] + "\n\n")
            for idx, example in enumerate(self.prompt["interactive_examples"]):
                f.write(example.format(idx=idx + 1) + "\n\n")
            self.n_shots = len(self.prompt["interactive_examples"])
            self.prompt_examples = f.getvalue()

        if (
            self.force_overall_prompt_on_overall_question
            or self.force_overall_question_on_overall_prompt
        ):
            self.overall_question = re.match(
                ".*((([A-Z].* (calculate|how|what|find|true or false))|((Calculate|How|What|Find|True or false))).*)$",
                self.example,
                flags=re.DOTALL,
            )
            if self.overall_question is not None:
                self.overall_question = self.overall_question[1]
            else:
                raise ValueError("Cannot find overall question in example")

    def get_actions(
        self,
        state: MATHState,
    ) -> list[MATHAction]:
        with io.StringIO() as f:
            f.write(self.prompt_examples)
            f.write(
                self.prompt["question_prefix"].format(
                    idx=self.n_shots + 1, question=self.example
                )
                + "\n"
            )
            for idx, (q, a, *_) in enumerate(state):
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

    """return 0, {}
        return 1, {'r_useful': 1} #TODO"""

    def get_fast_reward_input(
        self, state: GSM8kState, action: GSM8kAction
    ) -> tuple[float, dict]:
        with io.StringIO() as f:
            f.write(self.useful_prompt["input"])
            f.write(self.useful_prompt["question_prefix"] + self.example + "\n")
            for idx, (q, *_) in enumerate(state):
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

    def fast_reward(self, state, action) -> tuple[float, dict]:
        with io.StringIO() as f:
            f.write(self.useful_prompt["input"])
            f.write(self.useful_prompt["question_prefix"] + self.example + "\n")
            for idx, (q, *_) in enumerate(state):
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
        self,
        state: MATHState,
        action: MATHAction,
        r_useful: float = None,
        confidence: float = None,
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
            return "Now we can answer the question with an option from A to E:"
        return (
            "Now we can answer the question with an option from A to E: "
            + question.split("?")[0].replace(",", ".").split(".")[-1]
            + "?"
        )


from examples.RAP.prontoqa import prompts
from examples.RAP.prontoqa.prompts import finish
from examples.RAP.prontoqa.prompts import next_step
from examples.RAP.prontoqa.prompts import valid_rap

prompts.finish = finish
prompts.next_step = next_step
prompts.valid_rap = valid_rap


from examples.RAP.prontoqa.dataset import ProntoQAExample
from reasoners import SearchConfig, LanguageModel
from world_model import ProntoQAState, ProntoQAAction


def format_examples(sampled_data):
    formatted_examples = ""
    for i, entry in enumerate(sampled_data, 1):
        facts = f"Determine whether Query's conclusion is correct based on the following Facts.\nFacts {i}: {entry['Facts']}\n"
        query = f"Query {i}: {entry['Query']}\n"
        claims_and_next = ""

        for j, (claim, next_step) in enumerate(
            zip(entry["claims"], entry["next_steps"]), 1
        ):
            claims_and_next += f"Claim {i}.{j}: {claim}\nNext {i}.{j}: {next_step}\n"

        formatted_examples += facts + query + claims_and_next + "\n"

    return formatted_examples


class ProntoQAConfig(GSM8kConfig):
    def __init__(self, base_model: LanguageModel, temperature=0.8, n_candidates=4):
        super(GSM8kConfig, self).__init__()
        self.base_model = base_model
        self.temperature = temperature
        self.n_candidates = n_candidates
        self.example: ProntoQAExample = self.example
        self.reward_confidence_default = 1
        self.reward_alpha = 0
        self.history = ""
        self.index = 1
        self.question_prefix = "Next"
        self.answer_prefix = "Claim"

    def get_actions_inputs(self, state, action):
        *base_facts, init_state = self.example.test_example.question.split(". ")
        input_prompt = self.history
        if self.history == "":
            input_prompt = "Determine whether Query's conclusion is correct based on the following Facts.\n"
            input_prompt += prompts.next_step.FACTS_FORMAT.format(
                len(self.prompt) + 1, ". ".join(base_facts)
            )
            input_prompt += prompts.next_step.QUERY_FORMAT.format(
                len(self.prompt) + 1, self.example.test_example.query
            )
        else:
            input_prompt += action + "\n"
        input_prompt += prompts.next_step.CLAIM_FORMAT.format(
            len(self.prompt) + 1, self.index, state
        )
        input_prompt += prompts.next_step.NEXT_STEP_PREFIX.format(
            len(self.prompt) + 1, self.index
        )
        self.history = input_prompt
        self.index += 1
        input_prompt = format_examples(self.prompt) + input_prompt
        return self.n_candidates, input_prompt, None

    def get_actions_output(self, outputs, at_depth_limit):
        outputs = [output.strip() for output in outputs]
        outputs = list(dict.fromkeys(outputs))
        return outputs

    def get_fast_reward_input(
        self,
        state: ProntoQAState,
        action: ProntoQAAction,
    ) -> tuple[float, dict]:
        *base_facts, init_state = self.example.test_example.question.split(". ")
        input_prompt = ""
        match action:
            case "Finish.":
                input_prompt += prompts.finish.EXAMPLES
                input_prompt += prompts.finish.TARGET_FORMAT.format(
                    self.example.test_example.query
                )
                input_prompt += prompts.finish.CLAIM_FORMAT.format(state)
                input_prompt += prompts.finish.OUTPUT_PREFIX
            case _:
                input_prompt = (
                    prompts.valid_rap.TEMPLATE.replace("[[STATE]]", state.body)
                    .replace("[[ACTION]]", action)
                    .replace("[[QUERY]]", self.example.test_example.query)
                    .replace("[[FACTS]]", ". ".join(base_facts) + ".")
                )
        return input_prompt

    def update_example(self, example, prompt) -> None:
        super(GSM8kConfig, self).update_example(example, prompt=prompt)

    def get_finnal_question(self, question=""):
        return "Finish."


from utils import strategyQA_utils


class strategyQAUsefulPrompt(TypedDict):
    input: str
    question_prefix: str
    subquestion_prefix: str
    new_subquestion_prefix: str
    useful_prefix: str


from world_model import StrategyQAState, StrategyQAPrompt


class StrategyQAConfig(GSM8kConfig):
    def __init__(
        self,
        base_model: LanguageModel,
        prompt: dict,
        useful_prompt: dict,
        decompose_prompt: str,
        n_actions=4,
        batch_size=2,
        temperature=0.8,
        eos_token_id="\n",
        reward_alpha=0.5,
        reward_confidence_default=0.8,
        depth_limit=5,
        force_terminating_on_depth_limit=True,
        force_overall_prompt_on_overall_question=True,
        force_overall_question_on_overall_prompt=True,
    ) -> None:
        super(GSM8kConfig, self).__init__()
        self.base_model = base_model
        self.example = ""
        self.prompt: StrategyQAPrompt = prompt
        self.useful_prompt: strategyQAUsefulPrompt = useful_prompt
        self.decompose_prompt = decompose_prompt
        self.batch_size = batch_size
        self.temperature = temperature
        self.eos_token_id = eos_token_id
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
        self.subquestion_conf = {"Yes": 1.0, "Maybe": 0.5, "No": 0.1}
        self.question_prefix = "Question"
        self.answer_prefix = "Answer"

    def update_example(self, example: str, prompt=None) -> None:
        super(GSM8kConfig, self).update_example(example)
        if (
            self.force_overall_prompt_on_overall_question
            or self.force_overall_question_on_overall_prompt
        ):
            # self.overall_question = re.match('.*((Calculate|calculate|how|How|what|What|Find|find|True or false).*)$',
            #                                  self.example)[1]
            self.overall_question = self.example

    def get_actions_inputs(self, state: StrategyQAState):
        with io.StringIO() as f:
            if len(state) == 0:
                f.write(
                    self.decompose_prompt
                    + "\n\nQ: "
                    + self.overall_question
                    + '\nA: To answer the question "'
                    + self.overall_question
                    + '", we need to know:'
                )
            else:
                f.write(self.prompt["input"])
                f.write(self.prompt["question_prefix"] + self.example + "\n")
                for idx, t in enumerate(state):
                    q, a = t[:2]
                    f.write(
                        self.prompt["subquestion_prefix"].format(idx + 1)
                        + " "
                        + q
                        + "\n"
                    )
                    f.write(
                        self.prompt["answer_prefix"].format(idx + 1) + " " + a + "\n"
                    )
                f.write(self.prompt["subquestion_prefix"].format(len(state) + 1))
            if (
                at_depth_limit := self.force_terminating_on_depth_limit
                and len(state) + 1 >= self.depth_limit
            ):
                f.write(" " + self.prompt["overall_question_prefix"])

            model_input = f.getvalue()

        # print(model_input)
        # input(">")

        n_actions = 1 if at_depth_limit else self.n_actions
        n_samples = 0
        n_actions = self.n_actions
        temperature = self.temperature
        for idx in range(0, n_actions, self.batch_size):
            n_samples += min(n_actions - idx, self.batch_size)
        self.state = state
        return n_samples, model_input, at_depth_limit

    def get_actions_output(self, outputs, at_depth_limit):
        outputs = [output.strip() for output in outputs]
        if len(self.state) == 0:
            for i, output in enumerate(outputs):
                # print(f"sub-question output: {output}")
                subqs_list = strategyQA_utils.extract_subquestions(output[:-1])
                # print('\n<<<< sub-questions list >>>>\n{}'.format(subqs_list))
                q1 = subqs_list[0]
                if q1[0] != '"':
                    q1 = '"' + q1
                if q1[-1] != '"':
                    q1 = q1 + '"'
                # print('\n<<<< Q1 >>>>\n{}'.format(subq_format))
                outputs[i] = q1[1:-1]
        # print(f"====\nsub-question: {outputs}\n====")
        ### similar to is_terminal function in world
        if at_depth_limit:
            outputs = [
                self.prompt["overall_question_prefix"] + " " + self.overall_question
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
                last_sub_words = set(output.lower().split(" "))
                overall_ques_words = set(self.overall_question.lower().split(" "))
                new_words = last_sub_words - overall_ques_words
                if len(new_words) <= 1:
                    outputs[i] = (
                        self.prompt["overall_question_prefix"]
                        + " "
                        + self.overall_question
                    )
        outputs = list(dict.fromkeys(outputs))
        return outputs

    def get_fast_reward_input(self, state, action) -> tuple[float, dict]:
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
            model_input = f.getvalue().replace("Now we can answer the question: ", "")
        return model_input


from world_model import BWState, BWAction


class BWConfig(SearchConfig):
    def __init__(
        self,
        base_model: LanguageModel,
        prompt: dict,
        batch_size=2,
        reward_alpha=0.5,
        goal_reward_default=0.0,
        goal_reached_reward=100,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.batch_size = batch_size
        self.reward_alpha = reward_alpha
        self.goal_reward_default = goal_reward_default
        self.goal_reached_reward = goal_reached_reward
        import reasoners.benchmark.bw_utils as utils

        self.utils = utils

    def get_actions(self, state: BWState) -> list[BWAction]:
        blocks_state = state.blocks_state
        return self.utils.generate_all_actions(blocks_state)

    def get_fast_reward_input(self, state: BWState, action: BWAction):
        if state.buffered_action == "":
            # if no action buffered
            current_blocks_state = state.blocks_state
        else:
            # if action buffered
            current_blocks_state = state.last_blocks_state
        previous_action = (
            state.buffered_action + "\n" if state.buffered_action != "" else ""
        )

        icl_template = self.prompt["icl_list"][state.step_idx // 2]
        # every two step, we will deduct the icl prompt
        # so that the distribution of step length is more reasonable
        inputs_intuition = (
            icl_template.replace("<init_state>", current_blocks_state)
            .replace("<goals>", self.utils.extract_goals(self.example, return_raw=True))
            .replace("<action>", previous_action)
        )
        self_eval_prompt = (
            self.prompt["self-eval"]
            .replace("<init_state>", current_blocks_state)
            .replace("<goals>", self.utils.extract_goals(self.example, return_raw=True))
            .replace("<action>", action)
        )
        return [inputs_intuition, self_eval_prompt]

    def get_fast_reward_output(self, loglikelihood):
        intuition, self_eval = loglikelihood[:]
        return self.calculate_reward(intuition[0], self_eval[0]), {
            "intuition": intuition,
            "self_eval": self_eval,
        }

    def fast_reward(self, state: BWState, action: BWAction) -> tuple[float, dict]:
        pass

    def calculate_reward(self, intuition, self_eval, goal_reached=None):
        # to provide a unified interface for reward and fast_reward
        if goal_reached is None:
            goal_reward = self.goal_reward_default
        elif goal_reached[0]:
            goal_reward = self.goal_reached_reward
        else:
            goal_reward = goal_reached[1]
        return (intuition + self_eval) * self.reward_alpha + goal_reward * (
            1 - self.reward_alpha
        )

    def reward(
        self,
        state: BWState,
        action: BWAction,
        intuition: float = None,
        self_eval: float = None,
        goal_reached: tuple[bool, float] = None,
    ) -> float:
        assert intuition is not None, (
            "intuition is required to calculate reward in this search config, consider passing it in fast_reward"
        )
        assert self_eval is not None, (
            "self_eval is required to calculate reward in this search config, consider passing it in fast_reward"
        )
        assert goal_reached is not None, (
            "goal_reached is required to calculate reward in this search config, consider passing it in world model's step"
        )
        return (
            self.calculate_reward(intuition, self_eval, goal_reached),
            {"intuition": intuition, "goal_reached": goal_reached},
        )

    def update_example(self, example, prompt=None) -> None:
        super().update_example(example, prompt=prompt)
