from timeit import default_timer
from typing import Dict

import numpy as np
from nlm_utils.model_client.classification import ClassificationClient

from de_utils.utils import apply_private_dictionary
from discovery_engine.objects import GroupTaskData
from engines.base_engine import BaseEngine


class RetrievalEngine(BaseEngine):
    def __init__(self, settings: dict = {}):

        super().__init__(settings)
        self.retriever = "qnli" if settings["RUN_QNLI"] else "CrossEncoder"
        self.client = ClassificationClient(
            model=settings["QNLI_MODEL"],
            task=self.retriever,
            url=settings["MODEL_SERVER_URL"],
        )

    def run(self, group_task: GroupTaskData, **kwargs) -> Dict:
        if not self.settings["RUN_CROSS_ENCODER"] and not self.settings["RUN_QNLI"]:
            return
        tasks = group_task.tasks

        topics = kwargs["override_topic"]
        disable_extraction = group_task.settings.get("search_settings", {}).get(
            "disable_extraction",
            False,
        )

        questions = []
        sentences = []

        # loop over all matches to construct query
        for topic in topics:

            self.logger.info(f"Processing topic '{topic}'")
            task = tasks[topic]

            # Disable only when it's specified in the task or
            # (adhoc search and specified in workspace settings and its a file level search)
            if task.disable_extraction or (
                kwargs["ad_hoc"]
                and disable_extraction
                and kwargs["file_idx"]
                and isinstance(kwargs["file_idx"], str)
            ):
                self.logger.info(f"Extraction disabled for '{topic}'")
                continue

            for criteria in task.criterias:

                # current task has no question
                if not criteria.is_question:
                    continue

                question, _ = apply_private_dictionary(
                    criteria.question,
                    group_task.settings.get("applied_private_dictionary", {}),
                    change_to_key=True,
                )

                for _, matches in criteria.matches.items():
                    for match in matches:
                        if match.group_type == "table":
                            continue

                        sentence, _ = apply_private_dictionary(
                            match.qa_text,
                            group_task.settings.get("applied_private_dictionary", {}),
                            change_to_key=True,
                        )

                        sentences.append(sentence)
                        questions.append(question)

        if len(questions) == 0:
            return

        # query QNLI
        self.logger.info(f"Sending out {len(questions)} questions to {self.retriever}")
        model_server_wall_time = default_timer()

        response = self.client(
            questions,
            sentences,
            return_labels=False,
            return_probs=self.settings["RUN_QNLI"],
            return_logits=self.settings["RUN_CROSS_ENCODER"],
        )

        model_server_wall_time = (default_timer() - model_server_wall_time) * 1000
        self.logger.info(f"Entailment check takes {model_server_wall_time:.2f}ms.")

        # loop over all matches to assign qnli scores
        index = 0

        def sigmoid(z):
            return 1 / (1 + np.exp(-z / 4))

        if self.settings["RUN_CROSS_ENCODER"]:
            raw_scores = np.array(response["logits"]).squeeze()

        for topic in topics:

            task = tasks[topic]

            # Disable only when it's specified in the task or
            # (adhoc search and specified in workspace settings and its a file level search)
            if task.disable_extraction or (
                kwargs["ad_hoc"]
                and disable_extraction
                and kwargs["file_idx"]
                and isinstance(kwargs["file_idx"], str)
            ):
                continue

            for criteria in task.criterias:
                # current task has no question
                if (
                    # not task.question
                    not criteria.is_question
                    # or task.is_keyword_search
                    # or task.is_debug_search
                ):
                    continue

                for _, matches in criteria.matches.items():
                    cross_encoder_scores = []
                    # set all scores to 1 so that
                    # even if the engine is not on, we have a score that can be multiplied
                    match.raw_scores["cross_encoder_raw_score"] = 1
                    for match in matches:
                        if match.group_type == "table":
                            match.raw_scores["cross_encoder_raw_score"] = 0.0
                            continue
                        if self.settings["RUN_QNLI"]:
                            match.raw_scores["qnli_score"] = response["probs"][index][0]
                            match.has_answer = (
                                match.raw_scores["qnli_score"]
                                > self.settings["QNLI_THRESHOLD"]
                            )
                        elif self.settings["RUN_CROSS_ENCODER"]:
                            cross_encoder_scores.append(raw_scores[index])
                            match.raw_scores["cross_encoder_raw_score"] = raw_scores[
                                index
                            ]
                            match.has_answer = False
                        match.answer = ""
                        index += 1
                    if self.settings["RUN_CROSS_ENCODER"]:
                        index = 0
                        cross_encoder_scores = np.array(cross_encoder_scores)
                        cross_encoder_scores = sigmoid(cross_encoder_scores)
                        top_k_indices = np.argsort(cross_encoder_scores)[-5:]
                        for match in matches:
                            if match.group_type == "table":
                                match.raw_scores["cross_encoder_score"] = 0.0
                                continue
                            elif self.settings["RUN_CROSS_ENCODER"]:
                                match.raw_scores[
                                    "cross_encoder_score"
                                ] = cross_encoder_scores[index]
                                match.raw_scores["qnli_score"] = cross_encoder_scores[
                                    index
                                ]  # sometimes this is needed by various parts of the code
                                match.has_answer = index in top_k_indices
                            index += 1
