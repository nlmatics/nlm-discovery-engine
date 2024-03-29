from timeit import default_timer

from nlm_utils.model_client.classification import ClassificationClient

from de_utils.utils import apply_private_dictionary
from de_utils.utils import expand_cross_references
from de_utils.utils import OPENAI_MAX_NUM_TOKENS
from de_utils.utils import openai_tokenizer
from de_utils.utils import remove_quotes
from discovery_engine.objects import MatchData
from discovery_engine.objects import TaskData
from processors.base_processor import BaseProcessor

MAX_ALLOWED_CANDIDATES = 25
CANDIDATE_CONCAT_DELIM = "\n"
END_OF_SENTENCE_DELIM = [".", ",", ":", "!", ";", "?"]


class AbstractiveSummaryProcessor(BaseProcessor):

    processor_type = "abstractive_processor"

    def __init__(
        self,
        debug=False,
        settings: dict = None,
        **kwargs,
    ):
        super().__init__(settings)
        self.debug = debug
        self.group_task_settings = kwargs.get("group_task_settings")
        self.search_settings = self.group_task_settings.get("search_settings", {})
        self.summarization_top_n = kwargs.get("summarization_top_n", 10)
        self.enable_workspace_summarization = kwargs.get(
            "enable_workspace_summarization",
            None,
        )
        self.workspace_summarization_top_n = kwargs.get(
            "workspace_summarization_top_n",
            None,
        )

        self.summarization_model = self.search_settings.get(
            "summarization_model",
            self.settings["SUMMARIZATION_MODEL"],
        )
        self.summarization_model_server_url = self.search_settings.get(
            "summarization_model_server_url",
            self.settings["SUMMARIZATION_MODEL_SERVER_URL"],
        )
        self.summarization_task = self.search_settings.get(
            "summarization_task",
            self.settings["SUMMARIZATION_TASK"],
        )
        self.open_ai_model = self.search_settings.get(
            "openai_model",
            "gpt-3.5-turbo",
        )
        self.expand_references = self.search_settings.get(
            "expand_references",
            False,
        )
        self.max_num_tokens = self.search_settings.get(
            "max_summarization_tokens",
            OPENAI_MAX_NUM_TOKENS,
        )

        self.candidate_concat_delimiter = self.search_settings.get(
            "candidate_concat_delimiter",
            CANDIDATE_CONCAT_DELIM,
        )

        self.enable_staged_summarization = self.search_settings.get(
            "enable_staged_summarization",
            False,
        )

        if self.enable_workspace_summarization is None:
            self.enable_workspace_summarization = self.search_settings.get(
                "enable_workspace_summarization",
                False,
            )
        if self.workspace_summarization_top_n is None:
            self.workspace_summarization_top_n = self.search_settings.get(
                "workspace_summarization_top_n",
                0,
            )

    def run(self, task: TaskData, *args, **kwargs):
        questions = []
        sentences = []
        references = []
        summarization_top_n = self.summarization_top_n
        is_workspace_summarization_enabled = False

        # Reset summarization_top_n in case of workspace search
        if kwargs["ad_hoc"] and kwargs["file_idx"] is None:
            if (
                self.enable_workspace_summarization
                and self.workspace_summarization_top_n
            ):
                summarization_top_n = self.workspace_summarization_top_n
                is_workspace_summarization_enabled = True
            else:
                self.logger.info(
                    f"Exiting Workspace summarization: {self.enable_workspace_summarization}, "
                    f"{self.workspace_summarization_top_n}",
                )
                return
        # Rough token estimation limit.
        context_passage_len = 0
        for criteria in task.criterias:
            # remove question mark
            question, _ = apply_private_dictionary(
                criteria.question,
                self.group_task_settings.get("applied_private_dictionary", {}),
                change_to_key=True,
            )
            quest = remove_quotes(question)
            file_index = 0
            # Give preference to matches having an answer
            for file_idx, matches in sorted(
                criteria.matches.items(),
                key=lambda e: (any(x.has_answer for x in e[1])),
                reverse=True,
            ):
                file_index += 1
                has_any_answer = any(x.has_answer for x in matches)

                def sort_func(x):
                    nonlocal has_any_answer
                    if has_any_answer:
                        return (
                            x.has_answer,
                            x.raw_scores["boolq_score"] * x.raw_scores["squad_score"],
                            x.raw_scores["raw_match_score"],
                        )
                    else:
                        return x.raw_scores["raw_match_score"]

                final_sorted_matches = sorted(matches, key=sort_func, reverse=True)
                # Take only the first summarization_top_n matches
                final_matches = final_sorted_matches[:summarization_top_n]

                context_passage = ""
                context_block_idxs = []
                context_references = {}

                for match_index, match in enumerate(final_matches):
                    data_in_qa_text = match.qa_text
                    if ":" in match.qa_text:
                        data_in_qa_text = match.qa_text.split(":")[1].strip()
                    hierarchy_headers = []
                    for header in match.hierarchy_headers:
                        hierarchy_headers.append(" ".join(header.split()))
                    context = (
                        " ".join(hierarchy_headers)
                        + " "
                        + data_in_qa_text
                        + (
                            "."
                            if data_in_qa_text
                            and data_in_qa_text[-1] not in END_OF_SENTENCE_DELIM
                            else ""
                        )
                    )
                    # context = match.qa_text
                    # context = context.replace(":", self.candidate_concat_delimiter)

                    sentence, _ = apply_private_dictionary(
                        context,
                        self.group_task_settings.get("applied_private_dictionary", {}),
                        change_to_key=True,
                    )

                    if not sentence or sentence.isspace():
                        sentence_len = 0
                    else:
                        tokens = openai_tokenizer.encode(
                            sentence,
                            disallowed_special=(),
                        )
                        sentence_len = len(tokens)
                    if context_passage_len + sentence_len < self.max_num_tokens:
                        if context_passage:
                            context_passage = (
                                context_passage
                                + self.candidate_concat_delimiter
                                + sentence
                            )
                        else:
                            context_passage = sentence
                        if match.block_idx not in context_block_idxs:
                            context_block_idxs.append(match.block_idx)
                        if match.cross_references:
                            for k, v in match.cross_references.items():
                                context_references[k] = v
                        context_passage_len += sentence_len
                    else:
                        break

                if quest and context_passage:
                    reference = ""
                    if (
                        self.expand_references
                        and context_passage_len < self.max_num_tokens
                    ):
                        reference = expand_cross_references(
                            file_idx,
                            context_passage,
                            context_block_idxs,
                            self.max_num_tokens,
                        )
                        if not reference and context_references:
                            for k, v in context_references.items():
                                ref_string = f"{k} means "
                                for idx, i in enumerate(v):
                                    if idx > 0:
                                        ref_string += self.candidate_concat_delimiter
                                    ref_string += f"({idx + 1}) {i}"
                                ref_string += self.candidate_concat_delimiter
                                tokens = openai_tokenizer.encode(
                                    ref_string,
                                    disallowed_special=(),
                                )
                                ref_string_len = len(tokens)
                                if (
                                    context_passage_len + ref_string_len
                                    > self.max_num_tokens
                                ):
                                    break
                                reference += ref_string
                                context_passage_len += ref_string_len

                    if not is_workspace_summarization_enabled:
                        # Add more retrieved candidates from the pool till we reach a maximum
                        # of self.max_num_tokens words.
                        # Allow up to MAX_ALLOWED_CANDIDATES.
                        idx = summarization_top_n
                        tokens = openai_tokenizer.encode(
                            reference,
                            disallowed_special=(),
                        )
                        reference_len = len(tokens)
                        while (
                            context_passage_len + reference_len
                        ) < self.max_num_tokens and idx < MAX_ALLOWED_CANDIDATES:
                            if len(final_sorted_matches) > idx:
                                match = final_sorted_matches[idx]
                                data_in_qa_text = match.qa_text
                                if ":" in match.qa_text:
                                    data_in_qa_text = match.qa_text.split(":")[
                                        1
                                    ].strip()
                                hierarchy_headers = []
                                for header in match.hierarchy_headers:
                                    hierarchy_headers.append(" ".join(header.split()))
                                context = (
                                    " ".join(hierarchy_headers)
                                    + " "
                                    + data_in_qa_text
                                    + (
                                        "."
                                        if data_in_qa_text
                                        and data_in_qa_text[-1]
                                        not in END_OF_SENTENCE_DELIM
                                        else ""
                                    )
                                )
                                # context = match.qa_text.replace(
                                #     ":",
                                #     self.candidate_concat_delimiter,
                                # )
                                sentence, _ = apply_private_dictionary(
                                    context,
                                    self.group_task_settings.get(
                                        "applied_private_dictionary",
                                        {},
                                    ),
                                    change_to_key=True,
                                )
                                context_passage = (
                                    context_passage
                                    + self.candidate_concat_delimiter
                                    + sentence
                                )
                                tokens = openai_tokenizer.encode(
                                    sentence,
                                    disallowed_special=(),
                                )
                                sentence_len = len(tokens)
                                context_passage_len += sentence_len
                                idx += 1
                            else:
                                break

                    questions.append(quest)
                    sentences.append(context_passage)
                    references.append(reference)
                # Reset context_passage_len if staged summarization is set.
                if self.enable_staged_summarization:
                    context_passage_len = 0

        # no questions, skip
        if len(questions) == 0:
            return

        if is_workspace_summarization_enabled and not self.enable_staged_summarization:
            questions = questions[:1]
            sentences = [" ".join(sentence for sentence in sentences)]

        client = ClassificationClient(
            model=self.summarization_model,
            task=self.summarization_task,
            url=self.summarization_model_server_url,
            openai_model=self.open_ai_model,
            debug=True,
        )
        summary_qa_prompt_template = self.search_settings.get(
            "summary_qa_prompt_template",
            "",
        )
        summary_boolq_prompt_template = self.search_settings.get(
            "summary_boolq_prompt_template",
            "",
        )

        if self.settings["DEBUG"] and False:
            self.logger.info(f"Questions: {questions}")
            self.logger.info(f"Sentences: {sentences}")
        # query
        self.logger.info(
            f"Sending out {len(questions)} to {client.model}:{client.task}",
        )
        model_server_wall_time = default_timer()
        answers = client(
            questions,
            sentences,
            references=references,
            is_summarization=is_workspace_summarization_enabled,
            summary_qa_prompt_template=summary_qa_prompt_template,
            summary_boolq_prompt_template=summary_boolq_prompt_template,
        )

        answers = list(answers["answers"][0].values())
        new_sentences = []
        if self.enable_staged_summarization:
            val = [
                " ".join(
                    answer["text"]
                    for answer in answers
                    if answer["text"] != "No Answer Present."
                ),
            ]
            if val[0].strip():
                new_sentences.append(val[0].strip())
            if len(new_sentences):
                candidate_sentences = [
                    sentences[idx]
                    for idx, answer in enumerate(answers)
                    if answer["text"] != "No Answer Present."
                ]
                candidate_blob = ""
                for cand in candidate_sentences:
                    candidate_blob += cand + (
                        ". " if cand and cand[-1] not in END_OF_SENTENCE_DELIM else " "
                    )
                    candidate_blob = candidate_blob.strip()
                new_sentences.extend(
                    [
                        ". ".join(
                            sentences[idx]
                            for idx, answer in enumerate(answers)
                            if answer["text"] != "No Answer Present."
                        ),
                    ],
                )
                questions = questions[: len(new_sentences)]
                answers = client(
                    questions,
                    new_sentences,
                    references=references,
                    is_summarization=is_workspace_summarization_enabled,
                    summary_qa_prompt_template=summary_qa_prompt_template,
                    summary_boolq_prompt_template=summary_boolq_prompt_template,
                )

                answers = list(answers["answers"][0].values())

        model_server_wall_time = (default_timer() - model_server_wall_time) * 1000
        self.logger.info(f"Extracting answer span takes {model_server_wall_time:.2f}ms")

        for criteria in task.criterias:
            file_index = 0
            # Create a dummy match_data
            new_summary_match = None
            item_idx = -1
            for k, matches in criteria.matches.items():
                item_idx += 1
                match_added = False
                if len(matches) > 0 and len(answers) > 0:
                    summary_match = MatchData(
                        match_idx=10000,
                        match_text="",
                        parent_text="",
                        entity_types="",
                        hierarchy_headers=[""],
                        oid="Summary",
                        parent_oid="Summary",
                        qa_text="",
                        block_text="",
                        header_text="",
                        raw_scores={
                            "match_score": 0.99,
                            "raw_match_score": matches[0].raw_scores["raw_match_score"],
                            "file_score": matches[0].raw_scores.get(
                                "file_score",
                                0,
                            ),
                            "sif_score": matches[0]
                            .raw_scores.get("explanation", {})
                            .pop("sif_score", 0),
                            "dpr_score": matches[0]
                            .raw_scores.get("explanation", {})
                            .pop("dpr_score", 0),
                        },
                        page_idx=-1,
                        block_type="summary",
                        group_type="summary",
                        criteria=matches[0].criteria,
                        explanation={},
                        bbox=[-1, -1, -1, -1],
                        entity_list=[],
                        cross_references={},
                    )
                    if not is_workspace_summarization_enabled:
                        matches.insert(0, summary_match)
                        match_added = True
                    if is_workspace_summarization_enabled and item_idx == 0:
                        new_summary_match = summary_match
                        break
                if match_added:
                    criteria.matches[k] = matches
                    break

            if is_workspace_summarization_enabled and new_summary_match:
                new_summary_matches = {"summary": [new_summary_match]}
                new_summary_matches.update(criteria.matches)
                criteria.matches = new_summary_matches

            answer = answers[0]
            if not answer.get("text", ""):
                answer["text"] = "We couldn't find an answer for this question"
            if (
                self.enable_staged_summarization
                and len(answers) > 1
                and answer["text"] == "No Answer Present."
            ):
                answer = answers[1]

            for match_key, matches in criteria.matches.items():
                file_index += 1
                match = None
                if is_workspace_summarization_enabled:
                    # First item is the summary match
                    if match_key == "summary":
                        match = matches[0]
                else:
                    # First item is the summary match
                    match = matches[0]
                if match:
                    match.answer = answer["text"]
                    match.match_text = match.answer
                    match.has_answer = answer["text"] != ""
                    match.raw_scores["squad_score"] = answer["start_probs"]
                    match.raw_scores["start_byte"] = answer.get("start_byte", -1)
                    match.raw_scores["end_byte"] = answer.get("end_byte", -1)
                    match.explanation["summary_references"] = {
                        "first_stage": sentences,
                    }
                    if self.enable_staged_summarization and new_sentences:
                        match.explanation["summary_references"][
                            "second_stage"
                        ] = new_sentences
                    break
