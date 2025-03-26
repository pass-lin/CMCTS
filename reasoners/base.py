from typing import (
    Generic,
    TypeVar,
    Union,
    NamedTuple,
    Protocol,
    Optional,
    runtime_checkable,
    Tuple,
)
from abc import ABC, abstractmethod
import faiss
import numpy as np
from transformers import StoppingCriteriaList
from datetime import datetime
import os
import sys
import pickle
from tqdm import tqdm
import torch

import gc
import jsonlines

State = TypeVar("State")
Action = TypeVar("Action")
Example = TypeVar("Example")
Trace = tuple[list[State], list[Action]]


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class GenerateOutput(NamedTuple):
    text: list[str]
    log_prob: Optional[list[np.ndarray]] = None


class LanguageModel(ABC):
    @abstractmethod
    def generate(
        self,
        inputs: list[str],
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        eos_token_id: Union[None, str, int, list[str, int]] = None,
        hide_input: bool = True,
        output_log_probs: bool = False,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        **kwargs,
    ) -> GenerateOutput:
        """Generate text from a list of prompts.

        :param inputs: List of prompts.
        :param max_length: Maximum length of the total output (input + generated).
        :param max_new_tokens: Maximum length of generated tokens. Override max_length.
        :param do_sample: If False, do greedy decoding.
        :param temperature: Temperature for sampling.
        :param top_k: Top-k for sampling.
        :param top_p: Top-p for sampling.
        :param num_return_sequences:
        :param eos_token_id: Token id for end of sentence. Passed *str* will be translated into token_id.
                             Passed *list* will be treated as multiple possible tokens ending the generation.
        :param hide_input: If set true, decode only the generated part.
        :param output_log_probs: If set true, also output the log_probs of each generated token
        :param stopping_criteria:
        """
        ...

    @abstractmethod
    def get_next_token_logits(
        self,
        prompt: Union[str, list[str]],
        candidates: Union[list[str], list[list[str]]],
        postprocess: Optional[str] = None,
        **kwargs,
    ) -> list[np.ndarray]:
        """TODO: doc

        :param prompt:
        :param candidates:
        :param postprocess: optional, can be 'log_softmax' or 'softmax'. Apply the corresponding function to logits before returning
        :return:
        """
        ...

    @abstractmethod
    def get_loglikelihood(
        self, prefix: str, contents: list[str], **kwargs
    ) -> np.ndarray:
        """Get the log likelihood of the contents given the prefix.

        :param prefix: The prefix to be excluded from the log likelihood.
        :param contents: The contents to evaluate (must include the prefix).
        """
        ...


class WorldModel(ABC, Generic[State, Action, Example]):
    def __init__(self) -> None:
        self.example = None
        self.prompt = None

    @abstractmethod
    def init_state(self) -> State: ...

    @abstractmethod
    def step(self, state: State, action: Action) -> Union[State, Tuple[State, dict]]:
        """Returns the next state and optionally an auxiliary data dict

        :param state: The current state
        :param action: The action to take
        :return: The next state and optionally an auxiliary data dict
        """
        ...

    @abstractmethod
    def is_terminal(self, state: State) -> bool: ...

    def update_example(self, example: Example, prompt=None) -> None:
        if prompt is not None:
            self.prompt = prompt
        self.example = example


class DefaultWorldModel(WorldModel):
    # A default implementation of WorldModel that only
    # saves the action sequence as the state

    def __init__(self, base_model) -> None:
        super().__init__()
        self.base_model = base_model

    def init_state(self):
        return []

    def step(self, state, action):
        return state + [action], {}

    def is_terminal(self, state):
        # By default the state is never terminal
        return False


class SearchConfig(ABC, Generic[State, Action, Example]):
    def __init__(self) -> None:
        self.example = None
        self.prompt = None

    @abstractmethod
    def get_actions(self, state: State) -> list[Action]: ...

    def fast_reward(self, state: State, action: Action) -> tuple[float, dict]:
        return 0, {}

    @abstractmethod
    def reward(self, state, action, **kwargs) -> tuple[float, dict]: ...

    def update_example(self, example: Example, prompt=None) -> None:
        if prompt is not None:
            self.prompt = prompt
        self.example = example


@runtime_checkable
class AlgorithmOutput(Protocol[State]):
    terminal_state: State
    trace: Trace


class SearchAlgorithm(ABC):
    def __init__(self, **kwargs): ...

    @abstractmethod
    def __call__(
        self, world_model: WorldModel, search_config: SearchConfig, **kwargs
    ) -> AlgorithmOutput: ...


class Reasoner(ABC, Generic[State, Action, Example]):
    def __init__(
        self,
        world_model: WorldModel[State, Action, Example],
        search_config: SearchConfig[State, Action, Example],
        search_algo: SearchAlgorithm,
    ) -> None:
        self.world_model = world_model
        self.search_config = search_config
        self.search_algo = search_algo

    def __call__(
        self, example: Example, prompt=None, **kwargs
    ) -> AlgorithmOutput[State]:
        self.world_model.update_example(example, prompt=prompt)
        self.search_config.update_example(example, prompt=prompt)
        return self.search_algo(self.world_model, self.search_config, **kwargs)

    def update(self, example: Example, prompt=None, **kwargs) -> AlgorithmOutput[State]:
        self.world_model.update_example(example, prompt=prompt)
        self.search_config.update_example(example, prompt=prompt)


def l2_normalize(vecs):
    """标准化"""
    norms = (vecs**2).sum(axis=1, keepdims=True) ** 0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


import difflib


def eval_trajectory(trajectory):
    question = trajectory.split("\n")[0].split(":")[-1].split(",")[-1]
    states = trajectory.split("\n")
    if len(states) % 2 == 0:
        return False
    while states[-1] == "":
        states.pop(-1)
    for j in range(1, len(states) - 4, 2):
        distance = difflib.SequenceMatcher(
            None, states[j].split(":")[1], question
        ).ratio()
        if distance > 0.8:
            return True
    return False


class Evaluator:
    @abstractmethod
    def __init__(
        self,
        skil_lib=None,
        encoder=None,
        tokenizer=None,
        output_extractor=None,
        answer_extractor=None,
        text_max_len=256,
        eval_quality=False,
        thersold=0.85,
        batch_size=16,
    ) -> None:
        self.disable_tqdm = True
        self.batch_size = batch_size
        self.output_extractor = output_extractor
        self.answer_extractor = answer_extractor
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        self.index = None
        self.thersold = thersold
        if skil_lib is not None:
            assert encoder is not None and tokenizer is not None
            self.libs = []
            # 读取
            questions = []
            s = [[], []]
            if type(skil_lib) == str:
                f = open(skil_lib, "r+")
                skil_lib = jsonlines.Reader(f)
            else:
                f = None
            for item in skil_lib:
                if item["trajectory"][-1] == "\n":
                    item["trajectory"] = item["trajectory"][:-1]
                if (
                    "问题 {idx}" not in item["trajectory"]
                    and "问题" in item["trajectory"]
                ):
                    prefix = item["trajectory"].split(":")[0]
                    item["trajectory"] = item["trajectory"].replace(
                        prefix, "问题 {idx}"
                    )
                if "question" not in item.keys():
                    item["question"] = item["trajectory"].split("\n")[0].split(":")[-1]
                if item["question"] in questions:
                    continue
                if len(item["question"]) == 0 or len(item["trajectory"]) == 0:
                    continue
                min_iter = 6
                if eval_quality and (
                    len(item["trajectory"].split("\n")) < min_iter
                    or eval_trajectory(item["trajectory"])
                ):
                    if len(item["trajectory"].split("\n")) < min_iter:
                        s[0].append(item["trajectory"])
                    else:
                        s[1].append(item["trajectory"])
                    continue
                questions.append(item["question"])
                self.libs.append(item)
            if eval_quality:
                print(len(s[0]), len(s[1]))
            vectors = self.text2vector([t["question"] for t in self.libs])
            if self.index == None:
                self.index = faiss.IndexFlatIP(vectors.shape[-1])
            self.index.add(vectors)
            print("skill length is ", len(self.libs))
            if f is not None:
                f.close()
            gc.collect()

    def MaxSim(self, vector):
        sim, ids = self.index.search(
            vector[None] if np.ndim(vector) == 1 else vector, 1
        )
        return sim[0][0]

    def build_lib(self, libs):
        self.libs = libs
        for t in self.libs:
            if "question" not in t.keys():
                t["question"] = t["trajectory"].split("\n")[0].split(":")[-1]
        vectors = self.text2vector([t["question"] for t in self.libs])
        self.index = faiss.IndexFlatIP(vectors.shape[-1])
        self.index.add(vectors)

    def add_lib(self, extrlib):
        for t in extrlib:
            if "question" not in t.keys():
                t["question"] = t["trajectory"].split("\n")[0].split(":")[-1]

        self.libs += extrlib
        vectors = self.text2vector([t["question"] for t in extrlib])
        self.index.add(vectors)

    @abstractmethod
    def sample_prompt(self, shuffle_prompt, num_shot, sample_prompt_type):
        pass

    def text2vector(self, texts):
        from bert4keras3.snippets import sequence_padding

        tokens = []
        for text in texts:
            token = self.tokenizer.encode(text)
            if isinstance(token[0], list):
                token = token[0]
            tokens.append(token)
        tokens = sequence_padding(tokens, length=self.text_max_len)
        tokens = tokens.astype(np.int32)
        if len(self.encoder.input) == 2:
            vectors = self.encoder.predict(
                [tokens, np.zeros_like(tokens)], batch_size=self.batch_size
            )
        else:
            vectors = self.encoder.predict([tokens], batch_size=self.batch_size)
        gc.collect()
        return l2_normalize(vectors)

    def sample_skill(self, vector, topk=50, num_shot=4, threshold=0.9, alpha=1, beta=2):
        prompt = self.sample_prompt(num_shot=num_shot)
        prompt["meta-prompt-flag"] = 0
        prompt["interactive_examples"] = list(prompt["interactive_examples"])

        if self.index is None:
            return prompt
        sim, ids = self.index.search(
            vector[None] if np.ndim(vector) == 1 else vector, topk
        )
        sim, ids = sim[0], ids[0]
        ids = [ids[i] for i in range(topk) if sim[i] >= threshold]
        sim = [sim[i] for i in range(topk) if sim[i] >= threshold]

        if len(sim) == 0:
            return prompt
        scores = []
        for j, i in enumerate(ids):
            scores.append(self.libs[i]["reward"] * alpha + beta / self.libs[i]["ppl"])
        for i, j in enumerate(np.argsort(scores)[-num_shot:]):
            prompt["interactive_examples"][-i] = self.libs[ids[j]]["trajectory"]
        prompt["meta-prompt-flag"] = 1
        return prompt

    def renew_data(
        self,
        unsuccess_index,
        vectors,
    ):
        new_unsuccess_index = []
        for i in unsuccess_index:
            if self.evaluator.MaxSim(vectors[i]) < self.thersold:
                new_unsuccess_index.append(i)
        return new_unsuccess_index

    def evaluate(
        self, reasoner, shuffle_prompt=True, num_shot=4, resume=0, log_dir=None
    ):
        self.dataset = list(self.full_dataset)[resume:]
        try:
            algo_name = reasoner.search_algo.__class__.__name__
        except:
            algo_name = "unknown"

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            if log_dir is None:
                log_dir = (
                    f"logs/{self._dataset_name}_"
                    f"{algo_name}/"
                    f"{datetime.now().strftime('%m%d%Y-%H%M%S')}"
                )
            os.makedirs(log_dir, exist_ok=resume > 0)
            os.makedirs(os.path.join(log_dir, "algo_output"), exist_ok=True)

            with open(os.path.join(log_dir, "args.txt"), "w") as f:
                print(sys.argv, file=f)

        correct_count = 0

        disable_tqdm = self.disable_tqdm or (
            torch.distributed.is_initialized() and torch.distributed.get_rank() != 0
        )
        for i, example in enumerate(
            tqdm(
                self.dataset,
                total=resume + len(self.dataset),
                initial=resume,
                desc=self._dataset_name,
                disable=self.disable_tqdm,
            )
        ):
            print(example)
            algo_output = reasoner(
                self.input_processor(example),
                prompt=self.sample_prompt(
                    shuffle_prompt=shuffle_prompt, num_shot=num_shot
                ),
            )
            print(algo_output)
            output = self.output_extractor(algo_output)
            answer = self.answer_extractor(example)
            print(answer, output)
            correct = self.eval_output(answer, output)
            correct_count += correct
            accuracy = correct_count / (i + 1)
            log_str = (
                f"Case #{resume + i + 1}: {correct=}, {output=}, {answer=};"
                f"{accuracy=:.3f} ({correct_count}/{i + 1})"
            )
            tqdm.write(log_str)

            if (not self.disable_log) and (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                with open(os.path.join(log_dir, "result.log"), "a") as f:
                    print(log_str, file=f)

                with open(
                    os.path.join(log_dir, "algo_output", f"{resume + i + 1}.pkl"), "wb"
                ) as f:
                    pickle.dump(algo_output, f)

        return accuracy

    def evaluate_sc(
        self, reasoner, shuffle_prompt=True, num_shot=4, resume=0, n_sc=10, log_dir=None
    ):
        self.dataset = list(self.full_dataset)[resume:]
        try:
            algo_name = reasoner.search_algo.__class__.__name__
        except:
            algo_name = "unknown"

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            if log_dir is None:
                log_dir = (
                    f"logs/{self._dataset_name}_"
                    f"{algo_name}/"
                    f"{datetime.now().strftime('%m%d%Y-%H%M%S')}"
                )
            os.makedirs(log_dir, exist_ok=resume > 0)
            os.makedirs(os.path.join(log_dir, "algo_output"), exist_ok=True)

            with open(os.path.join(log_dir, "args.txt"), "w") as f:
                print(sys.argv, file=f)

        correct_count = 0

        disable_tqdm = self.disable_tqdm or (
            torch.distributed.is_initialized() and torch.distributed.get_rank() != 0
        )
        for i, example in enumerate(
            tqdm(
                self.dataset,
                total=resume + len(self.dataset),
                initial=resume,
                desc=self._dataset_name,
                disable=self.disable_tqdm,
            )
        ):
            prompt = self.sample_prompt(
                shuffle_prompt=shuffle_prompt, num_shot=num_shot
            )
            output_list = []
            save_list = []
            for j in range(n_sc):
                algo_output = reasoner(self.input_processor(example), prompt=prompt)
                terminal_state = algo_output.terminal_state
                path = ""
                for k in range(len(terminal_state)):
                    path += (
                        terminal_state[k].sub_question
                        + " "
                        + terminal_state[k].sub_answer
                        + " "
                    )
                save_list.append(path)
                output = self.output_extractor(algo_output)
                output_list.append(output)
                answer = self.answer_extractor(example)
            from collections import Counter

            output = Counter(output_list).most_common(1)[0][0]
            correct = self.eval_output(answer, output)
            correct_count += correct
            accuracy = correct_count / (i + 1)
            log_str = (
                f"Case #{resume + i + 1}: {correct=}, {output=}, {answer=};"
                f"{accuracy=:.3f} ({correct_count}/{i + 1})"
            )
            tqdm.write(log_str)

            if (not self.disable_log) and (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                with open(os.path.join(log_dir, "result.log"), "a") as f:
                    print(log_str, file=f)
                with open(os.path.join(log_dir, "algo_output.txt"), "a") as f1:
                    print(save_list, file=f1)

        return accuracy

    @abstractmethod
    def eval_output(self, answer, output):
        pass
