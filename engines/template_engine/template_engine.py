from timeit import default_timer
from typing import Dict

from nlm_utils.model_client import ClassificationClient
from nlm_utils.model_client import NlpClient
from server.storage import nosql_db

from de_utils.utils import is_bool_question
from discovery_engine.objects import CriteriaData
from discovery_engine.objects import GroupTaskData
from discovery_engine.objects import TaskData
from engines.base_engine import BaseEngine


class TemplateEngine(BaseEngine):
    def __init__(
        self,
        settings,
    ):
        super().__init__(settings)
        self._db = nosql_db

        self.qa_type_client = ClassificationClient(
            model=settings["QATYPE_MODEL"],
            task="qa_type",
            url=settings["QATYPE_MODEL_SERVER_URL"],
        )

        self.nlp_client = NlpClient(
            url=settings["QUERY_DECOMPOSITION_MODEL_SERVER_URL"],
        )

    def run(self, group_task: GroupTaskData, **kwargs) -> Dict:
        tasks = group_task.tasks
        table_flag = None
        # If workspace summarization or default summarization is enabled,
        # table_search flag from workspace settings take precedence.
        enable_workspace_summarization = group_task.settings.get(
            "search_settings",
            {},
        ).get(
            "enable_workspace_summarization",
            False,
        )
        enable_summarization_by_default = group_task.settings.get(
            "search_settings",
            {},
        ).get(
            "enable_summarization_by_default",
            False,
        )
        enable_linguistic_question_decomposition = group_task.settings.get(
            "search_settings",
            {},
        ).get(
            "enable_linguistic_question_decomposition",
            False,
        )

        if enable_workspace_summarization or enable_summarization_by_default:
            table_flag = group_task.settings.get("search_settings", {}).get(
                "table_search",
                False,
            )
            table_flag = "enable" if table_flag else "disable"

        # build field for ad hoc search
        if kwargs["ad_hoc"]:
            self.logger.info("Build template_dataframe for ad hoc search")

            fields = [
                {
                    "id": "Ad hoc",
                    "name": "Ad hoc",
                    "search_criteria": {
                        "criterias": kwargs["criterias"],
                        "post_processors": kwargs["post_processors"],
                        "aggregate_post_processors": kwargs[
                            "aggregate_post_processors"
                        ],
                        "group_by_file": kwargs.get("group_by_file", True),
                        "search_type": kwargs.get("search_type", "extraction"),
                        "extractors": kwargs["extractors"],
                        "disable_extraction": kwargs["disable_extraction"],
                        "abstractive_processors": kwargs["abstractive_processors"],
                    },
                    "pagination": {
                        "workspace": {
                            "offset": kwargs.get("offset", -1),
                            "result_per_page": kwargs.get("doc_per_page", -1),
                        },
                    },
                },
            ]
        # get field from db
        else:
            fields = []
            for field_id in kwargs["override_topic"]:
                field = self._db.get_field_by_id(field_id)
                if field:
                    fields.append(field)

        # get expected answer type
        answer_types = {}
        # List of Questions for linguistic decomposition.
        question_list = []
        linguistic_keywords = []
        linguistic_direct_keywords = []
        # infer answer type
        for field in fields:
            for criteria in field["search_criteria"]["criterias"]:
                question = criteria["question"].lower().strip()

                if question.endswith((".", "?")):
                    question = question[:-1]

                criteria["question"] = question
                # only infer answer type for question
                if not criteria["question"] or criteria["expected_answer_type"] == "":
                    criteria["expected_answer_type"] = ""
                elif (
                    criteria["expected_answer_type"] == "auto" and criteria["question"]
                ):
                    # For Boolean questions, we overwrite the answer_type in CriteriaData object instantiation.
                    # So save some time here by not doing an answer_type detection.
                    if not is_bool_question(criteria["question"]):
                        answer_types[criteria["question"]] = ""
                if criteria["question"]:
                    question_list.append(question)

        if answer_types and self.settings["USE_QATYPE"]:
            # query
            self.logger.info(
                f"Sending out {len(answer_types)} to {self.qa_type_client.model}:{self.qa_type_client.task}",
            )
            model_server_wall_time = default_timer()
            response = self.qa_type_client(list(answer_types.keys()))["predictions"]

            model_server_wall_time = (default_timer() - model_server_wall_time) * 1000
            self.logger.info(f"Infer answer type takes {model_server_wall_time:.2f}ms")

            # query answer type
            for k, v in zip(
                answer_types,
                response,
            ):
                answer_types[k] = v

        # Enable linguistic decomposition for workspace search.
        if (
            enable_linguistic_question_decomposition
            and kwargs["ad_hoc"]
            and kwargs["file_idx"] is None
        ):
            # query
            self.logger.info(
                f"Sending out {len(question_list)} to NlpClient:get_question_keywords",
            )
            resp = self.nlp_client(texts=question_list, option="get_question_keywords")
            resp = resp or []
            for r in resp:
                linguistic_keywords.append(r.get("keywords", []))
                linguistic_direct_keywords.append(r.get("direct_keywords", []))

        for field_idx, field in enumerate(fields):
            criteria_datas = []
            file_filter_text = []
            criteria_len = 0

            for criteria_idx, criteria in enumerate(
                field["search_criteria"]["criterias"],
            ):
                if table_flag is not None:
                    criteria["table_flag"] = table_flag

                if criteria["expected_answer_type"] in {"any", "disable"}:
                    answer_type = ""
                elif criteria["question"] in answer_types:
                    answer_type = answer_types[criteria["question"]]
                else:
                    answer_type = criteria["expected_answer_type"]

                # parsing question
                if "|" in criteria["question"]:
                    filter_text, question = criteria["question"].rsplit("|", 1)
                    file_filter_text.append(filter_text.strip())
                else:
                    question = criteria["question"]

                criteria_data = CriteriaData(
                    # searching criteria from kwargs
                    question=criteria["question"],
                    templates=criteria["templates"],
                    headers=criteria["headers"],
                    group_flag=criteria["group_flag"],
                    table_flag=criteria["table_flag"],
                    page_start=criteria["page_start"],
                    page_end=criteria["page_end"],
                    criteria_rank=criteria["criteria_rank"],
                    # expected answer type
                    expected_answer_type=answer_type,
                    # Similar search --> SIF
                    enable_similar_search=criteria["enable_similar_search"],
                    # Entity Type filters
                    entity_types=criteria["entity_types"],
                    # Additional extraction questions
                    additional_questions=criteria["additional_questions"],
                    before_context_window=criteria["before_context_window"],
                    after_context_window=criteria["after_context_window"],
                    linguistic_keywords=linguistic_keywords[
                        (field_idx * criteria_len) + criteria_idx
                    ]
                    if len(linguistic_keywords)
                    else [],
                    linguistic_direct_keywords=linguistic_direct_keywords[
                        (field_idx * criteria_len) + criteria_idx
                    ]
                    if len(linguistic_direct_keywords)
                    else [],
                )
                criteria_datas.append(criteria_data)

            tasks[field["id"]] = TaskData(
                topic=field["name"],
                topic_idx=field["id"],
                file_filter_text=" ".join(file_filter_text),
                post_processors=field["search_criteria"]["post_processors"],
                aggregate_post_processors=field["search_criteria"][
                    "aggregate_post_processors"
                ],
                criterias=criteria_datas,
                group_by_file=field["search_criteria"]["group_by_file"],
                search_type=field["search_criteria"]["search_type"],
                extractors=field["search_criteria"]["extractors"],
                disable_extraction=field["search_criteria"]["disable_extraction"],
                abstractive_processors=field["search_criteria"][
                    "abstractive_processors"
                ],
            )
            criteria_len += 1

        if self.settings["DEBUG"]:
            self.logger.debug(group_task.json(indent=2))
