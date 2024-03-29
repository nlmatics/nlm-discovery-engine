from timeit import default_timer
from typing import Dict

from nlm_utils.model_client.classification import ClassificationClient

from de_utils.utils import apply_private_dictionary
from discovery_engine.objects import GroupTaskData
from engines.base_engine import BaseEngine


class BooleanEngine(BaseEngine):
    def __init__(self, settings: dict = {}):

        super().__init__(settings)
        self.client = ClassificationClient(
            model=settings["BOOLQ_MODEL"],
            task="boolq",
            url=settings["BOOLQ_MODEL_SERVER_URL"],
        )

    def run(self, group_task: GroupTaskData, **kwargs) -> Dict:
        tasks = group_task.tasks

        boolq_model = group_task.settings.get("search_settings", {}).get(
            "boolq_model",
            self.settings["BOOLQ_MODEL"],
        )
        boolq_model_server_url = group_task.settings.get("search_settings", {}).get(
            "boolq_model_server_url",
            self.settings["BOOLQ_MODEL_SERVER_URL"],
        )
        disable_extraction = group_task.settings.get("search_settings", {}).get(
            "disable_extraction",
            False,
        )

        topics = kwargs["override_topic"]

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
                if not criteria.is_bool_question:
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
                        if (
                            self.settings["RUN_QNLI"]
                            and match.raw_scores["qnli_score"]
                            < self.settings["QNLI_THRESHOLD"]
                        ):
                            continue
                        sentence, _ = apply_private_dictionary(
                            match.qa_text,
                            group_task.settings.get("applied_private_dictionary", {}),
                            change_to_key=True,
                        )

                        sentences.append(sentence)
                        questions.append(question)

        # empty match, skip
        if len(questions) == 0:
            return

        # query model-server
        self.logger.info(
            f"Sending out {len(questions)} to Boolq",
        )
        model_server_wall_time = default_timer()

        if (
            boolq_model == self.settings["BOOLQ_MODEL"]
            and boolq_model_server_url == self.settings["BOOLQ_MODEL_SERVER_URL"]
        ):
            response = self.client(
                questions,
                sentences,
                return_probs=True,
            )
        else:
            self.logger.info(
                f"Creating a new boolq client with model {boolq_model}",
            )
            boolq_client = ClassificationClient(
                model=boolq_model,
                task="boolq",
                url=boolq_model_server_url,
            )
            response = boolq_client(
                questions,
                sentences,
                return_probs=True,
            )

        model_server_wall_time = (default_timer() - model_server_wall_time) * 1000
        self.logger.info(
            f"Get boolean answer span takes {model_server_wall_time:.2f}ms",
        )

        # loop over all matches to assign qnli scores
        index = 0
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
                if not criteria.is_bool_question:
                    continue

                for _, matches in criteria.matches.items():
                    for match in matches:
                        if match.group_type == "table":
                            continue
                        if (
                            self.settings["RUN_QNLI"]
                            and match.raw_scores["qnli_score"]
                            < self.settings["QNLI_THRESHOLD"]
                        ):
                            continue
                        prediction = response["predictions"][index].capitalize()
                        match.has_answer = False
                        if prediction != "Neutral":

                            match.raw_scores["boolq_score"] = max(
                                response["probs"][index],
                            )

                            # model-server returns True or False, but boolq always give answer of Yes/No to UI
                            match.answer = "Yes" if prediction == "True" else "No"
                            match.has_answer = True

                        index += 1
