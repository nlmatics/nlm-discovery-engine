import math

from nlm_utils.model_client.classification import ClassificationClient

from discovery_engine.objects import RelationData
from discovery_engine.objects import TaskData
from processors.base_processor import BaseProcessor


class RelationExtractionProcessor(BaseProcessor):

    processor_type = "post_processor"

    def __init__(
        self,
        settings: dict = {},
        **kwargs,
    ):
        super().__init__(settings)

        self.qa_client = ClassificationClient(
            model=settings["QA_MODEL"],
            task=settings["QA_TASK"],
            url=self.settings["QA_MODEL_SERVER_URL"],
        )

    def run(self, task: TaskData, *args, **kwargs):
        if task.search_type == "relation-triple":
            for criteria in task.criterias:
                for _, matches in criteria.matches.items():
                    self._run(criteria, matches)

    def _run(self, criteria, matches):

        questions = []
        sentences = []
        source_question = None
        target_question = None
        if len(criteria.additional_questions) == 0 and (
            criteria.question and criteria.question != "" and not criteria.is_question
        ):
            source_question = f"what {criteria.question} it"
            target_question = f"what does it {criteria.question}"
        elif len(criteria.additional_questions) >= 2:
            source_question = criteria.additional_questions[0]
            target_question = criteria.additional_questions[1]
        if source_question and target_question:
            for match in matches:
                questions.append(source_question)
                sentences.append(match.match_text)
                questions.append(target_question)
                sentences.append(match.match_text)

            if len(questions) > 0:
                predictions = self.qa_client(questions, sentences)
                answers = list(predictions["answers"][0].values())
                for idx, answer in enumerate(answers):
                    match = matches[math.floor(idx / 2)]
                    if not match.relation_data:
                        match.relation_data = RelationData(
                            head=answer["text"],
                            tail="",
                            tail_prob=0.0,
                            prop=0.0,
                            head_prob=answer["probability"],
                        )
                    else:
                        match.relation_data.tail = answer["text"]
                        match.relation_data.tail_prob = answer["probability"]
