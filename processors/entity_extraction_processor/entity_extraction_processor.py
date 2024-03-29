from typing import List

from nlm_ingestor.ingestor import line_parser
from nlm_utils.model_client.classification import ClassificationClient

from discovery_engine.objects import TaskData
from processors.base_processor import BaseProcessor


def get_noun_chunks(text):
    company_abbs = [
        # "co.",
        # "company",
        # "corp.",
        # "corporation",
        # "inc",
        # "incorporated",
        # "l.l.c.",
        # "limited",
        # "llc",
        # "ltd",
        # "ltd.",
        # "p.c.",
        # "llp",
    ]
    return line_parser.Line(
        text,
        noun_chunk_ending_tokens=company_abbs,
    ).noun_chunks


class EntityExtractionProcessor(BaseProcessor):

    processor_type = "post_processor"

    def __init__(
        self,
        entity_samples: List[str],
        question="$ENTITY legal matters",
        threshold=0.25,
        settings: dict = {},
        **kwargs,
    ):
        super().__init__(settings)

        self.threshold = threshold
        self.entity_samples = entity_samples
        self.question = question

        self.stsb_client = ClassificationClient(
            model="roberta",
            task="stsb",
            url=self.settings["MODEL_SERVER_URL"],
        )

        self.qa_client = ClassificationClient(
            model=settings["QA_MODEL"],
            task=settings["QA_TASK"],
            url=self.settings["QA_MODEL_SERVER_URL"],
        )

    def run(self, task: TaskData, *args, **kwargs):
        for criteria in task.criterias:
            for _, matches in criteria.matches.items():
                self._run(matches)

    def _run(self, matches):

        noun_chunks_by_match = {}
        for match in matches:
            noun_chunks_by_match[match.match_idx] = get_noun_chunks(match.match_text)

        left_sents = []
        right_sents = []
        for entity_sample in self.entity_samples:
            for noun_chunks in noun_chunks_by_match.values():
                for noun_chunk in noun_chunks:
                    left_sents.append(entity_sample)
                    right_sents.append(noun_chunk)

        logits = self.stsb_client(
            left_sents,
            right_sents,
            return_logits=True,
        )["logits"]
        scores = {}
        for right_sent, (score,) in zip(right_sents, logits):
            scores[right_sent] = score

        candidates = []
        questions = []
        sentences = []
        entities = []
        for match in matches:
            # get the best_entity
            if not noun_chunks_by_match[match.match_idx]:
                continue

            entities_in_cur_match = [
                x
                for x in noun_chunks_by_match[match.match_idx]
                if scores[x] > self.threshold
            ]
            # skip if similarity for best entity is lower than threshold
            if not entities_in_cur_match:
                self.logger.debug("best entity is below threshold")
                continue
            if self.question:
                for entity in entities_in_cur_match:
                    questions.append(self.question.replace("$ENTITY", entity))
                    sentences.append(match.match_text)
                    candidates.append(match)
                    entities.append(entity)
                    match.formatted_answer = ""
            else:
                match.formatted_answer = ", ".join(entities_in_cur_match)

        if self.question and questions and sentences:
            predictions = self.qa_client(questions, sentences)
            answers = list(predictions["answers"][0].values())
            for entitie, answer, candidate in zip(entities, answers, candidates):
                if answer["probability"] < 0.5 or answer["text"] == "":
                    pass
                    # candidate.formatted_answer = ""
                else:
                    if candidate.formatted_answer:
                        candidate.formatted_answer += f"; {entitie} => {answer['text']}"
                    else:
                        candidate.formatted_answer += f"{entitie} => {answer['text']}"

        for match in matches:
            match.answer = match.formatted_answer or ""
            match.has_answer = match.answer != ""
