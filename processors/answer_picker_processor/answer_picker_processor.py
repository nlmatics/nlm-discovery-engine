import re
from collections import defaultdict
from typing import List

import numpy as np
from nlm_utils.model_client.classification import ClassificationClient
from nlm_utils.model_client.encoder import EncoderClient
from nltk import jaccard_distance
from scipy.spatial.distance import cdist

from discovery_engine.objects import TaskData
from processors.base_processor import BaseProcessor


MULTIUSE = {
    "Mutli-Use",
    "Multi Use",
    "Multiple Uses",
    "Mixed Use",
    "Mixed-Use",
    "Multiuse",
    "multi-use",
    "multi use",
    "multiple uses",
    "mixed use",
    "mixed-use",
    "multiuse",
}


class AnswerPickerProcessor(BaseProcessor):

    processor_type = "post_processor"

    def __init__(
        self,
        choices: List[str],
        encoder_name="stsb",
        multi_answer_label=None,
        debug=False,
        settings: dict = {},
        **kwargs,
    ):
        super().__init__(settings)
        self.choices = choices
        self.set_choices = set(self.choices)
        self.multi_answer_label = multi_answer_label
        self.debug = debug
        self.is_encoder_based = encoder_name in ["distilbert", "sif"]
        if self.is_encoder_based:
            self.client = EncoderClient(
                model=encoder_name,
                url=self.settings["MODEL_SERVER_URL"],
            )
            self.choice_embeddings = (
                self._encode_texts(self.choices) if self.choices else []
            )
        else:
            self.client = ClassificationClient(
                model="roberta",
                task=encoder_name,
                url=self.settings["MODEL_SERVER_URL"],
            )

    def string_similarity(self, x, y):
        return jaccard_distance(set(x.lower()), set(y.lower()))

    def _encode_texts(self, text):
        return self.client(text)["embeddings"]

    def run(self, task: TaskData, *args, **kwargs):
        for criteria in task.criterias:
            for _, matches in criteria.matches.items():
                for match in matches:
                    if match.answer:
                        match.formatted_answer = self._run(
                            match.answer,
                            match.match_text,
                        )
                        match.answer_details = {
                            "formatted_value": match.formatted_answer,
                            "raw_value": match.formatted_answer,
                        }                        

    def _run_encoder(self, string_input, match_text):
        query_embedding = self._encode_texts([string_input])
        matches = defaultdict(int)

        # USE STRING SIMILARITY
        words = [re.sub("[/-]", " ", word) for word in string_input.split()]
        for word in words:
            for choice in self.choices:
                d = self.string_similarity(word, choice)
                if d < 0.25:
                    matches[choice] += 1

        # USE WORD EMBEDDING SIMILARITY
        for choice, choice_embedding in zip(self.choices, self.choice_embeddings):
            distance = cdist(query_embedding, [choice_embedding], "cosine")[0]
            if distance < 0.36:
                matches[choice] += 1

        multi_choices = MULTIUSE.intersection(self.set_choices)
        if multi_choices and len(matches) > 1:
            return multi_choices.pop()

        matches = sorted(ans for ans, count in matches.items() if count > 0)[::-1]
        return ", ".join(matches)

    def _run_classifier(self, answer, match_text, debug=True):
        rights = self.choices
        lefts = [answer for i in range(len(rights))]
        res = np.array(
            self.client(
                lefts,
                rights,
                multi_answer_label=self.multi_answer_label,
                return_labels=False,
                return_logits=True,
            )["logits"],
        )
        logits_dim = res.shape[1]
        answers = []  # if using stsb or similar
        if logits_dim == 1:
            threshold = 0.6
            axis = 0
            indices = np.where(res[:, axis] > threshold)[0]
        else:  # if using mnli
            threshold = 0.6
            axis = -1
            indices = np.where(res[:, axis] > threshold)[0]
            if len(indices) == 0:
                threshold = 0.9
                axis = -2
                indices = np.where(res[:, axis] > threshold)[0]
        # sorted_indices = (-1 * np.take(res, indices, axis=0)[:, axis]).argsort()
        indices = np.take(
            indices,
            (-1 * np.take(res, indices, axis=0)[:, axis]).argsort(),
        )
        answers = np.take(rights, indices)
        scores = np.take(res, indices, axis=0).squeeze(axis=1)

        close_weak_scores = (
            len(scores) > 1 and scores[0] < 0.7 and (scores[1] / scores[0] > 0.75)
        )
        strong_multi_scores = (
            len(scores) > 1 and scores[0] < 0.75 and (scores >= 0.4).sum() > 1
        )
        multi_answer = close_weak_scores or strong_multi_scores
        if multi_answer:
            if self.multi_answer_label:
                result = self.multi_answer_label
            else:
                result = ", ".join(answers[0:2].tolist())
        else:
            result = answers[0] if len(answers) > 0 else "Other" if answer != "" else ""
        if debug:
            buf = []
            for idx, (class_name, score) in enumerate(zip(answers, scores)):
                buf.append(
                    f"{class_name}: {scores[idx]} {idx,  max(idx -1, 0)} {scores[idx]/scores[max(idx -1, 0)]}",
                )
            # logger.info(f"-- {answer}---------------")
            # logger.info("\n".join(buf))
            # logger.info(
            #     f"{close_weak_scores} --  {strong_multi_scores} = {multi_answer}",
            # )
            # logger.info(f"-- {answer} -> {result}---------------")
        return result, scores

    def _run(self, string_input, match_text):
        if self.is_encoder_based:
            return self._run_encoder(string_input, match_text)
        else:
            result, _ = self._run_classifier(string_input, match_text, self.debug)
            return result
