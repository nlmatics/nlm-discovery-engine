from timeit import default_timer
from typing import Dict

from nlm_utils.model_client.classification import ClassificationClient

from de_utils.utils import apply_private_dictionary
from de_utils.utils import remove_quotes
from discovery_engine.objects import GroupTaskData
from engines.base_engine import BaseEngine


class AnsweringEngine(BaseEngine):
    def __init__(self, settings: dict = {}):

        super().__init__(settings)
        self.qa_client = ClassificationClient(
            model=settings["QA_MODEL"],
            task=settings["QA_TASK"],
            url=settings["QA_MODEL_SERVER_URL"],
        )
        self.phrase_qa_client = ClassificationClient(
            model=settings["QA_MODEL"],
            task=settings["PHRASE_QA_TASK"],
            url=settings["MODEL_SERVER_URL"],
        )
        self.phrase_on_top_k_doc = 1000
        self.phrase_on_top_k_doc_for_topic_processor = 5

    def run(self, group_task: GroupTaskData, **kwargs) -> Dict:
        tasks = group_task.tasks

        topics = kwargs["override_topic"]

        questions = []
        sentences = []

        qa_model = group_task.settings.get("search_settings", {}).get(
            "qa_model",
            self.settings["QA_MODEL"],
        )
        qa_model_server_url = group_task.settings.get("search_settings", {}).get(
            "qa_model_server_url",
            self.settings["QA_MODEL_SERVER_URL"],
        )
        qa_task = group_task.settings.get("search_settings", {}).get(
            "qa_task",
            self.settings["QA_TASK"],
        )
        disable_extraction = group_task.settings.get("search_settings", {}).get(
            "disable_extraction",
            False,
        )

        # loop over all matches to construct query
        for topic in topics:

            self.logger.info(f"Processing topic '{topic}'")
            task = tasks[topic]

            # check if we need phrase extraction for TopicProcessor
            topic_processor = False
            for post_processor in task.post_processors:
                if post_processor.lower().startswith("topicprocessor"):
                    topic_processor = True
                    break

            if topic_processor:
                self.logger.info("Running phrase extraction for topic processor")

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
                # skip boolean question
                if criteria.is_bool_question:
                    continue

                # remove question mark
                question, _ = apply_private_dictionary(
                    criteria.question,
                    group_task.settings.get("applied_private_dictionary", {}),
                    change_to_key=True,
                )
                file_index = 0
                for _, matches in criteria.matches.items():
                    file_index += 1

                    for match_index, match in enumerate(matches):
                        if not criteria.is_question:
                            continue

                        if match.group_type == "table" and match.block_type not in [
                            "para",
                            "header",
                        ]:
                            continue

                        # break if match has been filtered out by QNLI
                        if (
                            criteria.is_question
                            and self.settings["RUN_QNLI"]
                            and match.raw_scores["qnli_score"]
                            < self.settings["QNLI_THRESHOLD"]
                        ):
                            continue

                        if (
                            criteria.is_question
                            and self.settings["RUN_CROSS_ENCODER"]
                            and not match.has_answer
                        ):
                            continue

                        # run phrase extraction for top 2 matches for TopicProcessor
                        if topic_processor:
                            if (
                                match_index
                                > self.phrase_on_top_k_doc_for_topic_processor
                            ):
                                break
                            # run phrase extraction for top run_level_summarization_top_n documents
                            elif file_index > self.phrase_on_top_k_doc:
                                break

                        context = match.qa_text
                        if task.search_type == "relation-node":
                            context = match.match_text

                        sentence, _ = apply_private_dictionary(
                            context,
                            group_task.settings.get("applied_private_dictionary", {}),
                            change_to_key=True,
                        )
                        sentences.append(sentence)
                        questions.append(remove_quotes(question))

        # no match, skip
        if len(questions) == 0:
            return

        client = self.qa_client

        if (
            qa_model != self.settings["QA_MODEL"]
            or qa_task != self.settings["QA_TASK"]
            or qa_model_server_url != self.settings["QA_MODEL_SERVER_URL"]
        ):
            client = ClassificationClient(
                model=qa_model,
                task=qa_task,
                url=qa_model_server_url,
            )

        if self.settings["DEBUG"] and False:
            self.logger.info(f"Questions: {questions}")
            self.logger.info(f"Sentences: {sentences}")
        # query
        self.logger.info(
            f"Sending out {len(questions)} to {client.model}:{client.task}",
        )
        model_server_wall_time = default_timer()
        answers = client(questions, sentences)
        answers = list(answers["answers"][0].values())
        model_server_wall_time = (default_timer() - model_server_wall_time) * 1000
        self.logger.info(f"Extracting answer span takes {model_server_wall_time:.2f}ms")

        qa_threshold = group_task.settings.get("search_settings", {}).get(
            "qa_threshold",
            self.settings["SQUAD_THRESHOLD"],
        )
        self.logger.info(f"Applying QA Threshold: {qa_threshold}")

        # loop over all matches to assign squad scores
        index = 0
        for topic in topics:

            self.logger.info(f"Processing topic '{topic}'")
            task = tasks[topic]

            # check if we need phrase extraction for TopicProcessor
            topic_processor = False
            for post_processor in task.post_processors:
                if post_processor.lower().startswith("topicprocessor"):
                    topic_processor = True
                    break

            if topic_processor:
                self.logger.info("Running phrase extraction for topic processor")

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

                # skip boolean question
                if criteria.is_bool_question or not criteria.is_question:
                    continue

                file_index = 0
                for _, matches in criteria.matches.items():
                    file_index += 1
                    for match_index, match in enumerate(matches):
                        if match.group_type == "table" and match.block_type not in [
                            "para",
                            "header",
                        ]:
                            continue

                        # break if match has been filtered out by QNLI
                        if (
                            criteria.is_question
                            and self.settings["RUN_QNLI"]
                            and match.raw_scores["qnli_score"]
                            < self.settings["QNLI_THRESHOLD"]
                        ):
                            continue

                        if (
                            criteria.is_question
                            and self.settings["RUN_CROSS_ENCODER"]
                            and not match.has_answer
                        ):
                            continue

                        # run phrase extraction for top 2 matches for TopicProcessor
                        if topic_processor:
                            if (
                                self.settings["RUN_QNLI"]
                                and match.raw_scores["qnli_score"]
                                < self.settings["QNLI_THRESHOLD"]
                            ):
                                continue
                        else:
                            # run phrase extraction for top 2 matches for TopicProcessor
                            if topic_processor:
                                if (
                                    match_index
                                    > self.phrase_on_top_k_doc_for_topic_processor
                                ):
                                    break
                                # run phrase extraction for top run_level_summarization_top_n documents
                                elif file_index > self.phrase_on_top_k_doc:
                                    break

                        answer = answers[index]

                        if answer["start_probs"] < qa_threshold:
                            answer["text"] = ""

                        # answer should come from match_text only (in case of normal extraction)
                        if answer["text"] and answer["text"] not in match.match_text:
                            answer["text"] = ""

                        match.answer = answer["text"]

                        match.has_answer = answer["text"] != ""
                        match.raw_scores["squad_score"] = answer["start_probs"]
                        # if match.has_answer:
                        #     match.raw_scores["squad_score"] = answer["probability"]
                        # else:
                        #     match.raw_scores["squad_score"] = answer["probability"]
                        match.raw_scores["start_byte"] = answer.get("start_byte", -1)
                        match.raw_scores["end_byte"] = answer.get("end_byte", -1)
                        index += 1
