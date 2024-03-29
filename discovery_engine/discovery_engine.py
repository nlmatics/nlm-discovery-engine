"""
Discovery Engine
"""
import logging
from datetime import datetime
from pprint import pformat
from timeit import default_timer

from nlm_utils.utils.answer_type import answer_type_options
from server.storage import nosql_db
from xxhash import xxh32_hexdigest as hash

from discovery_engine.objects import DocumentData
from discovery_engine.objects import GroupTaskData
from engines import *  # noqa: F403, F401

REGISTERED_ENGINES = {
    "TemplateEngine",
    "SearchEngine",
    "TableExtractionEngine",
    "RetrievalEngine",
    "AnsweringEngine",
    "BooleanEngine",
    "PostProcessorEngine",
    "AbstractiveProcessorEngine",
    "GroupingEngine",
    "PostRankingEngine",
    "AggregatePostProcessorEngine",
}

ALL_FILES_IDX = "all_files"


class DiscoveryEngine:
    def __init__(self, settings: dict, mode: str = "pipeline"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.settings = settings

        self.engines = {}
        self.pre_pipeline = []
        self.pipeline = []

        # self.post_processors = {}
        # self.agg_post_processors = {}

        self.mode = mode

        self.logger.info("Discovery Engine initilized!")
        self.logger.info(f"DE-LITE settings:\n{pformat(self.settings)}")

    def init_workers(self):
        # init engine for discovery_engine instance
        for name, engine_class in self.engines.items():
            self.engines[name] = engine_class(self.settings)

        if self.mode == "restful":
            self.logger.info("Welcome to DE-LITE Restful services.")
        elif self.mode == "grpc":
            self.logger.info("Welcome to DE-LITE gRPC services.")
        else:
            self.logger.info("Welcome to DE-LITE. We are running on local now.")

        self.inited = True

    def add_engine(
        self,
        engine,
        engine_type: str = "engine",
    ):
        """ """
        if isinstance(engine, str):
            engine_class = eval(engine)
        else:
            engine_class = engine
        engine_name = engine_class.__name__

        if engine_name not in REGISTERED_ENGINES:
            raise ValueError(
                f"Engine {engine_name} is not registered.\n"
                f"Possible choices are {REGISTERED_ENGINES}",
            )

        if engine_type.lower() == "preprocessing":
            self.pre_pipeline.append(engine_name)
        else:
            self.pipeline.append(engine_name)
        self.engines[engine_name] = engine_class

    def apply_template(
        self,
        workspace_idx: str = None,
        file_idx: str = None,
        **kwargs,
    ):
        """Main entry for discovery engine

        Args:
            :param file_idx: File ID on which search needs to be performed
            :param workspace_idx: Workspace ID on which search needs to be performed
        Returns:
            facts: jsonified result
            result_df: A DataFrame that contains the result
        """

        wall_time = default_timer() * 1000

        self.logger.info(f"Get query of {kwargs}")

        ad_hoc = kwargs.get("ad_hoc", False)
        file_filter_struct = kwargs.get("file_filter_struct", {})
        user_acl = kwargs.get("user_acl", [])

        # build search kwargs
        if ad_hoc:
            if "criterias" not in kwargs:
                raise ValueError(
                    "discovery engine must receive list of criterias for ad hoc search",
                )
            criterias = kwargs["criterias"]
            for idx, criteria in enumerate(criterias):
                if not criteria:
                    raise ValueError("received empty criteria")
                criterias[idx] = {
                    "orig_question": criteria.get("question", ""),
                    "question": criteria.get("question", ""),
                    "templates": criteria.get("templates", []),
                    "headers": criteria.get("headers", []),
                    "expected_answer_type": criteria.get(
                        "expected_answer_type",
                        "auto",
                    ),
                    "group_flag": criteria.get("group_flag", "auto"),
                    "table_flag": criteria.get("table_flag", "auto"),
                    "page_start": criteria.get("page_start", -1),
                    "page_end": criteria.get("page_end", -1),
                    "criteria_rank": criteria.get("criteria_rank", -1),
                    "enable_similar_search": criteria.get(
                        "enable_similar_search",
                        True,
                    ),
                    "entity_types": criteria.get("entity_types", []),
                    "additional_questions": criteria.get("additional_questions", []),
                    "before_context_window": criteria.get("before_context_window", 0),
                    "after_context_window": criteria.get("after_context_window", 0),
                }
            kwargs = {
                # key params
                "workspace_idx": workspace_idx,
                "file_idx": file_idx,
                # ad hoc flag
                "ad_hoc": True,
                # override topic if needed
                "override_topic": {"Ad hoc"},
                "field_bundle_idx": None,
                # for ad_hoc search
                "criterias": criterias,
                "post_processors": kwargs.get("post_processors", []),
                "aggregate_post_processors": kwargs.get(
                    "aggregate_post_processors",
                    [],
                ),
                # pagination, default to first page, 20 docs per page
                "doc_per_page": kwargs.get("doc_per_page", 20),
                "match_per_doc": kwargs.get("match_per_doc", 20),
                "offset": kwargs.get("offset", 0),
                "file_filter_struct": file_filter_struct,
                "user_acl": user_acl,
                # debug
                "topn": kwargs.get("topn", -1),
                "group_by_file": kwargs.get("group_by_file", True),
                "search_type": kwargs.get("search_type", "extraction"),
                "debug": kwargs.get("debug", True),
                # control the return
                "return_raw": kwargs.get("return_raw", False),
                "return_df": kwargs.get("return_df", False),
                "extractors": kwargs.get("extractors", []),
                "disable_extraction": kwargs.get("disable_extraction", False),
                "abstractive_processors": kwargs.get("abstractive_processors", []),
            }
        else:
            if "field_bundle_idx" not in kwargs:
                raise ValueError(
                    "discovery engine must receive field_bundle_idx for field search",
                )

            override_topic = kwargs.get("override_topic", None) or []
            if isinstance(override_topic, str):
                override_topic = override_topic.split(",")

            kwargs = {
                # key params
                "workspace_idx": workspace_idx,
                "file_idx": file_idx,
                # ad hoc flag
                "ad_hoc": False,
                # override topic if needed
                "override_topic": set(override_topic),
                # field_bundle_idx
                "field_bundle_idx": kwargs["field_bundle_idx"],
                # pagination, 10_000 docs per page for extraction
                "doc_per_page": kwargs.get("doc_per_page", 10_000),
                "match_per_doc": kwargs.get("match_per_doc", 20),
                "offset": kwargs.get("offset", 0),
                "batch_idx": kwargs.get("batch_idx", ""),
                "file_filter_struct": None,
                "user_acl": user_acl,
                # debug
                "topn": kwargs.get("topn", -1),
                "group_by_file": kwargs.get("group_by_file", True),
                "search_type": kwargs.get("search_type", "extraction"),
                "debug": kwargs.get("debug", True),
                # control the return
                "return_raw": kwargs.get("return_raw", False),
                "return_df": kwargs.get("return_df", False),
            }

        # group task is a object hold the result
        group_task = GroupTaskData()

        self.logger.info("Collecting document infos")
        # for search across files (no grouping)
        if not kwargs.get("group_by_file", True):
            kwargs["workspace_idx"] = workspace_idx
            group_task.documents[ALL_FILES_IDX] = DocumentData(
                file_idx=ALL_FILES_IDX,
                file_name="All Documents in Workspace",
                file_title="All Documents in Workspace",
                file_meta={},
            )
        # for single file search
        elif kwargs["file_idx"]:
            doc_info = nosql_db.get_document_info_by_id(file_idx)
            kwargs["workspace_idx"] = doc_info.workspace_id
            group_task.documents[doc_info.id] = DocumentData(
                file_idx=doc_info.id,
                file_name=doc_info.name,
                file_title=doc_info.title if doc_info.title else doc_info.name,
                file_meta=doc_info.meta,
            )
        # for field_bundle_extraction
        elif kwargs["field_bundle_idx"]:
            doc_infos = nosql_db.get_folder_contents(
                kwargs["workspace_idx"],
                docs_per_page=kwargs["doc_per_page"],
                offset=kwargs["offset"],
            )["documents"]
            kwargs["file_idx"] = []
            for doc_info in doc_infos:
                kwargs["file_idx"].append(doc_info.id)
                group_task.documents[doc_info.id] = DocumentData(
                    file_idx=doc_info.id,
                    file_name=doc_info.name,
                    file_title=doc_info.title if doc_info.title else doc_info.name,
                    file_meta=doc_info.meta,
                )

        self.logger.info("done")

        self.logger.info("Collecting workspace info")
        settings = nosql_db.get_workspace_by_id(
            kwargs["workspace_idx"],
            remove_private_data=True,
        ).settings
        group_task.settings["private_dictionary"] = settings.get(
            "private_dictionary",
            {},
        )
        group_task.settings["search_settings"] = settings.get(
            "search_settings",
            {},
        )
        group_task.settings["index_settings"] = settings.get(
            "index_settings",
            {},
        )
        self.logger.info(f"Collecting workspace: Settings: {settings}")

        # run extraction pipeline
        self.run_pipeline(group_task, **kwargs)
        if kwargs.get("return_raw", False):
            data = group_task
        else:
            data = self.get_json_data(group_task, **kwargs)

            if kwargs.get("return_df", False):
                data["df"] = self.get_result_dataframe(data["outputs"], **kwargs)

        self.logger.info(
            f"DiscoveryEngine finished in {default_timer()*1000-wall_time:.4f}ms on workspace",
        )
        return data

    def run_pipeline(self, group_task, **kwargs):
        # pass override_topic into pipeline
        self.logger.info(f"Executing pipeline with kwargs: {kwargs}")
        if kwargs["override_topic"]:
            dependent_workflow_fields = []
            if kwargs["override_topic"] == {"ALL"}:
                fields = nosql_db.get_fields_in_bundle(
                    kwargs["field_bundle_idx"],
                )
                kwargs["override_topic"] = [x["id"] for x in fields]
                if kwargs.get("file_idx", ""):
                    # Ingestion / Re-ingestion scenario?
                    for field in fields:
                        if (
                            field.is_entered_field
                            and field.is_dependent_field
                            and field.options
                        ):
                            if field.options.get(
                                "deduct_from_file_meta",
                                False,
                            ) and field.options.get("meta_param", ""):
                                self.logger.info(
                                    f"Updating Meta dependent workflow field: {field.workspace_id} -- {field.id}",
                                )
                                nosql_db.create_workflow_fields_from_doc_meta(
                                    field.workspace_id,
                                    field.parent_bundle_id,
                                    field.id,
                                    field.options.get("meta_param", ""),
                                    "ingestion_task",
                                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    file_idx=kwargs.get("file_idx", ""),
                                )
                            elif field.options.get(
                                "deduct_from_fields",
                                False,
                            ) and field.options.get("parent_fields", []):
                                dependent_workflow_fields.append(field)

            # run main extraction pipeline
            for engine_name in self.pipeline:
                engine = self.engines[engine_name]
                # inplace tasks
                engine(group_task, **kwargs)

            # Process dependent workflow fields after the pipeline has executed.
            file_idx = kwargs.get("file_idx", "")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for field in dependent_workflow_fields:
                if (
                    field.is_entered_field
                    and field.is_dependent_field
                    and field.options
                    and field.options.get("deduct_from_fields", False)
                ):
                    self.logger.info(
                        f"Updating field dependent workflow field: {field.workspace_id} -- {field.id}",
                    )
                    nosql_db.create_fields_dependent_workflow_field_values(
                        field.workspace_id,
                        field.parent_bundle_id,
                        field.id,
                        field.options,
                        "ingestion_task",
                        current_time,
                        file_idx=file_idx,
                    )

    @staticmethod
    def add_pagination_json(outputs, task):
        pagination = task.pagination["workspace"]
        outputs["pagination"] = (
            {
                "workspace": {
                    "offset": pagination["offset"],
                    "total": pagination["total"],
                    "result_per_page": pagination["result_per_page"],
                },
            },
        )

    def get_json_data(self, group_task, **kwargs):
        # single file output
        if isinstance(kwargs["file_idx"], str) and kwargs["file_idx"]:
            return self.get_json(group_task, **kwargs)
        elif not kwargs.get("group_by_file", True):
            kwargs["file_idx"] = ALL_FILES_IDX
            json_data = self.get_json(group_task, **kwargs)
            outputs = {
                "outputs": [json_data],
            }
            key = next(iter(group_task.tasks.keys()))
            DiscoveryEngine.add_pagination_json(outputs, group_task.tasks[key])
            return outputs
        else:
            json_data = self.get_group_json(group_task, **kwargs)
            return json_data

    def get_json(
        self,
        group_task,
        file_idx,
        workspace_idx,
        ad_hoc,
        workspace_search=False,
        **kwargs,
    ) -> dict:
        """Return the result in list, which will be serialized into json later

        Params:
            dfr: source dataframe having template matches
            template_df: template dataframe

        Returns:
            res: answers in a list
        """
        self.logger.info("Generating Json")
        wall_time = default_timer()

        json_data = []

        # How many results to return?
        show_n = kwargs["topn"]
        if self.settings["DEBUG"] or kwargs["debug"]:
            show_n = 10000

        for _, task in group_task.tasks.items():
            matches = task.matches.get(file_idx, [])

            # init output for current task
            cur_json = {
                "file_idx": file_idx,
                "file_name": group_task.documents[file_idx].file_name
                if file_idx != "summary"
                else "",
                "file_meta": group_task.documents[file_idx].file_meta
                if file_idx != "summary"
                else {},
                "topic": task.topic,
                "topicId": task.topic_idx,
                "criterias": [x.dict(exclude={"matches"}) for x in task.criterias],
                "criteria_uid": hash(":".join([x.uid for x in task.criterias])),
                "post_processors": task.post_processors,
                "answers": 0,
                "topic_facts": [],
            }

            grouped_topic_facts = {}

            for match in matches:
                # break if reach the limit and current match is not in top n groups
                if (
                    0 < show_n <= len(grouped_topic_facts)
                    and match.group not in grouped_topic_facts
                ):
                    break

                if match.raw_scores["scaled_score"] == 0:
                    continue

                # init a new groups
                if match.group not in grouped_topic_facts:
                    grouped_topic_facts[match.group] = []

                grouped_topic_facts[match.group].append(match)

            # return topic_facts groups
            grouped_topic_facts = list(grouped_topic_facts.values())

            for matches in grouped_topic_facts:
                topic_facts = {"matches": []}

                for match in matches:
                    topic_fact = {
                        # attributes
                        "match_idx": match.match_idx,
                        "page_idx": match.page_idx,
                        "block_type": match.block_type,
                        # "level": match.level,
                        "answer": match.answer,
                        "entity_types": match.entity_types,
                        "entity_list": match.entity_list,
                        "cross_references": match.cross_references,
                        "table": match.table_answer or match.table,
                        "table_all": match.table,
                        "formatted_answer": match.formatted_answer
                        if match.formatted_answer is not None
                        else match.answer,
                        "answer_details": match.answer_details,
                        "scaled_score": match.raw_scores["scaled_score"],
                        "match_score": match.raw_scores["match_score"],
                        "raw_match_score": match.raw_scores["raw_match_score"],
                        "relevancy_score": match.raw_scores["relevancy_score"],
                        "qa_score": match.raw_scores["squad_score"],
                        "group_score": match.raw_scores["group_score"],
                        "retriever_score": match.raw_scores["cross_encoder_score"]
                        if "cross_encoder_score" in match.raw_scores
                        else match.raw_scores["qnli_score"],
                        "retriever_raw_score": match.raw_scores[
                            "cross_encoder_raw_score"
                        ]
                        if "cross_encoder_score" in match.raw_scores
                        else 0.0,
                        "table_score": match.raw_scores["table_score"],
                        "file_score": match.raw_scores["file_score"],
                        "semantic_score": match.explanation.get("sif_score", 0.0),
                        "boolq_score": match.raw_scores["boolq_score"]
                        if "boolq_score" in match.raw_scores
                        else 0.0,
                        "is_override": False,
                        "group_type": match.group_type,
                        # bbox
                        "bbox": match.bbox,
                        # match
                        "phrase": match.block_text
                        if match.block_type == "header"
                        else match.match_text,
                        "match_text_terms": match.explanation.get(
                            "match_text_terms",
                            [],
                        ),
                        "match_semantic_terms": match.explanation.get(
                            "match_semantic_terms",
                            [],
                        ),
                        "parent_text": match.parent_text,
                        "criteria_question": match.criteria.question,
                        # header
                        "header_text": match.header_text,
                        "header_text_terms": match.explanation.get(
                            "header_text_terms",
                            [],
                        ),
                        "header_semantic_terms": match.explanation.get(
                            "header_semantic_terms",
                            [],
                        ),
                        # header_chain
                        "hierarchy_headers": match.hierarchy_headers,
                        "hierarchy_headers_text_terms": match.explanation.get(
                            "header_chain_text_terms",
                            [],
                        ),
                        "header_semantic_terms_terms": match.explanation.get(
                            "header_semantic_terms_terms",
                            [],
                        ),
                        # others
                        "block_text_terms": match.explanation.get(
                            "block_text_terms",
                            [],
                        ),
                        # summary references
                        "summary_references": match.explanation.get(
                            "summary_references",
                            {},
                        ),
                    }
                    # Do not show scores when we are dealing with keyword based searches.
                    if not match.criteria.is_question:
                        topic_fact.pop("scaled_score")
                    if match.document_data:
                        topic_fact["file_idx"] = match.document_data.file_idx
                        topic_fact["file_name"] = match.document_data.file_name
                        topic_fact["file_title"] = match.document_data.file_title
                        topic_fact["file_meta"] = match.document_data.file_meta

                    if match.relation_data:
                        topic_fact["relation_head"] = match.relation_data.head
                        topic_fact["relation_tail"] = match.relation_data.tail
                        topic_fact["relation_head_prob"] = match.relation_data.head_prob
                        topic_fact["relation_tail_prob"] = match.relation_data.tail_prob

                    # generate uniq_id for extracted answer
                    topic_fact["uniq_id"] = hash(
                        # criteria
                        f"{cur_json['criteria_uid']}"
                        # match
                        f":{match.uid}"
                        # answer
                        f":{topic_fact['answer']}:{topic_fact['formatted_answer']}",
                    )
                    topic_facts["matches"].append(topic_fact)

                # use first child as the parent match
                topic_facts.update(topic_facts["matches"][0])

                # remove parent form the group matches for list_item
                if len(matches) > 1 and matches[0].group_type in {
                    "list_item",
                }:
                    topic_facts["matches"].pop(0)

                # remove header parent form the group matches for header_summary
                if (
                    len(matches) > 1
                    and matches[0].group_type
                    in {
                        "header_summary",
                    }
                    and matches[0].block_type == "header"
                ):
                    topic_facts["matches"].pop(0)

                # single and table group do not need child
                if topic_facts["group_type"] in {"single", "table"}:
                    del topic_facts["matches"]

                # header with one child do not need child
                elif (
                    topic_facts["group_type"] in {"header_summary", "table"}
                    and len(topic_facts["matches"]) == 1
                ):
                    if not len(topic_facts["matches"][0].get("answer", "")):
                        del topic_facts["matches"]

                if (
                    # has child
                    "matches" in topic_facts
                    # grouping is enable
                    and task.criterias[0].group_flag == "enable"
                    # answer from group types needs concat
                    and matches[0].group_type
                    in {"header_summary", "list_item", "same_location"}
                ):
                    topic_facts["formatted_answer"] = "/n".join(
                        [
                            x["formatted_answer"] or x["phrase"]
                            for x in topic_facts["matches"]
                        ],
                    )

                cur_json["topic_facts"].append(topic_facts)

            if len(cur_json["topic_facts"]) > 0 or kwargs.get(
                "keep_empty_matches",
                True,
            ):
                json_data.append(cur_json)

        if not ad_hoc:
            extracted_answer_in_db = []
            if len(json_data) == 1 and ALL_FILES_IDX == file_idx:
                res = json_data[0]
                nosql_db.add_results_to_extracted_field(
                    field_id=res["topicId"],
                    new_results=res["topic_facts"],
                    batch_idx=kwargs.get("batch_idx", ""),
                )
            else:
                for res in json_data:
                    extracted_answer_in_db.append(
                        {
                            "file_idx": res["file_idx"],
                            "file_name": res["file_name"],
                            "workspace_idx": workspace_idx,
                            "field_idx": res["topicId"],
                            "field_bundle_idx": kwargs["field_bundle_idx"],
                            "topic_facts": res["topic_facts"],
                        },
                    )
                nosql_db.create_extracted_field(extracted_answer_in_db)

        # reorder results by topics
        if not ad_hoc:
            topic_id_json_map = {}
            for json_row in json_data:
                topic_id_json_map[json_row["topicId"]] = json_row
            ordered_json_data = []

            for topic_idx in group_task.tasks:
                if topic_idx in topic_id_json_map:
                    ordered_json_data.append(topic_id_json_map[topic_idx])
        else:
            ordered_json_data = json_data

        wall_time = default_timer() - wall_time
        self.logger.info(f"Generating json in {wall_time:.4f}s")

        return ordered_json_data

    def get_group_json(
        self,
        group_task,
        workspace_idx,
        **kwargs,
    ):
        ad_hoc = kwargs["ad_hoc"]

        wall_time = default_timer()

        json_outputs = []

        topic_ids = {}

        # get templates for grid view
        grid_templates = []
        if not ad_hoc:
            # get all topicIds
            topics = nosql_db.get_fields_in_bundle(kwargs.get("field_bundle_idx"))
            for topic in topics:
                topic_ids[topic.id] = None

            # use topic order
            unordered_templates = {}
            for topic_idx, task in group_task.tasks.items():
                unordered_templates[topic_idx] = {
                    "topicId": topic_idx,
                    "topic": task.topic,
                    "post_processors": task.post_processors,
                }

            for topicId in topic_ids:
                if topicId in unordered_templates:
                    grid_templates.append(unordered_templates[topicId])
        else:
            for topic_idx, task in group_task.tasks.items():
                options = {}
                if len(task.criterias) > 0:
                    if task.criterias[0].expected_answer_type in answer_type_options:
                        options = answer_type_options[
                            task.criterias[0].expected_answer_type
                        ]

                grid_templates.append(
                    {
                        "topicId": topic_idx,
                        "topic": task.topic,
                        "options": options,
                        "post_processors": task.post_processors,
                    },
                )

        # get outputs for each file
        file_idxs = {}
        if kwargs["file_idx"] and not ad_hoc:
            for file_idx in kwargs["file_idx"]:
                file_idxs[file_idx] = None

        for topic_idx, task in group_task.tasks.items():
            for file_idx in task.matches:
                if file_idx not in file_idxs:
                    file_idxs[file_idx] = None

        del kwargs["file_idx"]

        for file_idx in file_idxs:
            output = self.get_json(
                group_task,
                file_idx=file_idx,
                workspace_idx=workspace_idx,
                workspace_search=True,
                topic_ids=topic_ids,
                **kwargs,
            )
            if output:
                json_outputs.append(output)

        if ad_hoc:
            # sort outputs for ad_hoc search
            def sort_func(output):
                summary = output[0]["topic_facts"][0]["block_type"] == "summary"
                match_score = output[0]["topic_facts"][0]["match_score"]
                relevancy_score = output[0]["topic_facts"][0]["relevancy_score"]
                file_score = output[0]["topic_facts"][0]["file_score"]
                answer = output[0]["topic_facts"][0]["answer"]
                has_answer = len(answer) > 0
                if self.settings["USE_FILE_SCORE"]:
                    return [
                        summary,
                        has_answer,
                        file_score,
                        match_score * relevancy_score,
                    ]
                else:
                    return [
                        summary,
                        has_answer,
                        match_score * relevancy_score,
                        file_score,
                    ]

            json_outputs.sort(key=sort_func, reverse=True)

        # generate grid output for each file:
        grid_outputs = []
        for file_output in json_outputs:
            grid_view = {}
            for fact in file_output:

                grid_view[fact["topic"]] = {
                    "match": "",
                    "match_idx": 0,
                    # "avg_score": 0,
                    "answer": "",
                    "formatted_answer": "",
                    "answer_prob": 0,
                    "scaled_score": 0,
                    "header_text": "",
                    "is_override": False,
                    # "group_count": 0,
                }

                if fact["topic_facts"]:
                    grid_view[fact["topic"]] = {
                        "match": fact["topic_facts"][0]["phrase"],
                        "match_idx": fact["topic_facts"][0]["match_idx"],
                        # "avg_score": fact["topic_facts"][0]["match_score"],
                        "answer": fact["topic_facts"][0]["answer"],
                        "formatted_answer": fact["topic_facts"][0]["formatted_answer"],
                        "answer_details": fact["topic_facts"][0]["answer_details"],
                        # "answer_prob": fact["topic_facts"][0]["answer_score"],
                        "scaled_score": fact["topic_facts"][0].get("scaled_score", 0),
                        "is_override": "is_override" in fact["topic_facts"][0]
                        and fact["topic_facts"][0]["is_override"],
                        # "group_count": fact["group_count"],
                        "post_processors": fact["post_processors"],
                        "header_text": fact["topic_facts"][0].get("header_text", ""),
                        # "topic_facts": fact["topic_facts"],
                    }
                if fact:
                    grid_view["file_name"] = fact["file_name"]
                    grid_view["file_idx"] = fact["file_idx"]

            grid_outputs.append(grid_view)

        outputs = {
            "aggregate_post_processors": {
                task.topic: task.aggs for task in group_task.tasks.values()
            },
            "grid": [grid_templates, grid_outputs],
            "outputs": json_outputs,
        }
        if ad_hoc:
            DiscoveryEngine.add_pagination_json(outputs, group_task.tasks["Ad hoc"])
        wall_time = default_timer() - wall_time
        self.logger.info(f"Generating group json in {wall_time:.4f}s")

        return outputs
