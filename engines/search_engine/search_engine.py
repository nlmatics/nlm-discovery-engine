import copy
import pickle
from timeit import default_timer
from typing import Dict

import pandas as pd
from nlm_utils.utils import preprocessor
from server.storage import nosql_db

from .es_client import ElasticsearchEngine
from de_utils.utils import apply_private_dictionary
from discovery_engine.objects import DocumentData
from discovery_engine.objects import GroupTaskData
from discovery_engine.objects import MatchData
from engines.base_engine import BaseEngine


def calculate_relevancy_score(templates, raw_text, block_type=None):
    text = preprocessor(raw_text, use_stemmer=True)

    score = 0
    # minimum score
    score_scaler = 0.7
    for template in templates:
        template = preprocessor(template, use_stemmer=True)
        # score_scaler = min(len(template), score_scaler)
        if not template:
            continue
        # keyword match
        # intersect(match_text, query) / len(query)
        _score = len(set(text).intersection(set(template))) / len(set(template))
        score = max(_score, score)

    # score_scaler calculates the base weight for confident score
    # few templates needs stronger matches, thus score_scaler is small --> must match more keywords
    # when multiple template words provided, score_scaler is big --> many few keywords is still okay
    # score_scaler = score_scaler * 0.3 - 1
    # score_scaler = 1 / (1 + math.exp(-score_scaler))
    # score_scaler = min(1, max(0, score_scaler))

    # table should be more restricted of matching all keywords
    if block_type == "table":
        score_scaler = 0.1
        # score = score ** 2
    return score_scaler + (1 - score_scaler) * score


class SearchEngine(BaseEngine):
    def __init__(
        self,
        settings,
    ):
        super().__init__(settings)
        self.elastic_search = ElasticsearchEngine(
            self.settings["ES_URL"],
            settings=self.settings,
        )

    def run(self, group_task: GroupTaskData, **kwargs) -> Dict:
        tasks = group_task.tasks
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

        topics = kwargs["override_topic"]

        workspace_idx = kwargs["workspace_idx"]
        file_idx = kwargs["file_idx"]
        file_filter_struct = kwargs["file_filter_struct"]
        user_acl = kwargs["user_acl"]

        for topic in topics:
            self.logger.info(f"Processing topic '{topic}'")
            task = tasks[topic]

            for criteria in task.criterias:
                act_criteria = copy.deepcopy(criteria)
                pre_process_time = default_timer()

                keywords = criteria.templates
                questions = [criteria.question]
                headers = criteria.headers
                entity_types = criteria.entity_types

                # clean out empty query
                def clean_text(texts, is_question=False):
                    cleaned_texts = []
                    for text in texts:
                        # Remove any chatGPT specific prompts.
                        # Use the actual question to do the retrieval.
                        if (
                            enable_workspace_summarization
                            or enable_summarization_by_default
                        ) and is_question:
                            text = text.split("-")[0].strip()

                        mod_texts, applied_dict = apply_private_dictionary(
                            text,
                            group_task.settings["private_dictionary"],
                        )
                        if not is_question:
                            for t in mod_texts:
                                if t and t not in cleaned_texts:
                                    cleaned_texts.append(t)
                        else:
                            t = mod_texts[0]
                            if t and t not in cleaned_texts:
                                cleaned_texts.append(t)
                        for key, val in applied_dict.items():
                            group_task.settings["applied_private_dictionary"][key] = val
                            if is_question:
                                group_task.settings[
                                    "applied_question_private_dictionary"
                                ][key] = val
                    return cleaned_texts

                if not group_task.settings.get("applied_private_dictionary", None):
                    group_task.settings["applied_private_dictionary"] = {}
                    group_task.settings["applied_question_private_dictionary"] = {}

                questions = clean_text(questions, is_question=True)
                headers = clean_text(headers)
                keywords = clean_text(keywords)

                pre_process_time = (default_timer() - pre_process_time) * 1000
                self.logger.debug(f"Preprocess takes {pre_process_time:.2f}ms")

                page_range = None
                if criteria.page_start != -1 or criteria.page_end != -1:
                    page_range = (
                        criteria.page_start,
                        criteria.page_end,
                    )

                candidate_length = 0

                search_time = default_timer()

                explained_matches = []
                sif_threshold = self.settings["SIF_THRESHOLD"]
                dpr_threshold = self.settings["DPR_THRESHOLD"]
                parents = {}

                if len(questions) == 0 and len(keywords) == 0 and len(headers) == 0:
                    return
                for res in self.elastic_search.get_candidates(
                    questions=questions,
                    keywords=keywords,
                    headers=headers,
                    task=task,
                    criteria=criteria,
                    entity_type=criteria.expected_answer_type,
                    page_range=page_range,
                    sif_threshold=sif_threshold,
                    dpr_threshold=dpr_threshold,
                    workspace_idx=workspace_idx,
                    file_idx=file_idx,
                    file_filter_struct=file_filter_struct,
                    matches_per_doc=kwargs.get("match_per_doc", 20),
                    offset=kwargs["offset"],
                    doc_per_page=kwargs["doc_per_page"],
                    table_flag=criteria.table_flag,
                    file_filter_text=task.file_filter_text,
                    settings=group_task.settings,
                    entity_types=entity_types,
                    group_by_file=task.group_by_file,
                    block_types=[],
                    search_type=task.search_type,
                    user_acl=user_acl,
                ):
                    candidate_length += 1

                    # build match instance
                    hierarchy_headers = [
                        x["block_text"] for x in res.get("level_chain", [])[::-1]
                    ]

                    match = MatchData(
                        match_idx=res["match_idx"],
                        match_text=res["match_text"],
                        parent_text=res.get("parent_text", ""),
                        entity_types=res.get("entity_types", ""),
                        hierarchy_headers=hierarchy_headers,
                        oid=res["_id"],
                        parent_oid=res["parent_uid"],
                        qa_text=res.get("qa_text", "") or res["match_text"],
                        block_idx=res["block_idx"],
                        block_text=res["block_text"],
                        header_text=res["header_text"]
                        if res["block_type"] != "header"
                        else res["match_text"],
                        raw_scores={
                            "match_score": res["match_score"],
                            "raw_match_score": res["raw_match_score"],
                            "file_score": res.get("file_score", 0),
                            "sif_score": res.get("explanation", {}).pop("sif_score", 0),
                            "dpr_score": res.get("explanation", {}).pop("dpr_score", 0),
                        },
                        page_idx=res["page_idx"],
                        block_type=res["block_type"],
                        group_type=res.get("group_type", "single"),
                        criteria=act_criteria,
                        explanation=res.get("explanation", {}),
                        bbox=res.get("bbox", [-1, -1, -1, -1]),
                        entity_list=res.get("entity_list", []),
                        cross_references=res.get("cross_references", {}),
                    )

                    if not task.group_by_file:
                        file_idx = res["file_idx"]
                        doc_info = nosql_db.get_document_info_by_id(file_idx)
                        document_data = DocumentData(
                            file_idx=doc_info.id,
                            file_name=doc_info.name,
                            file_title=doc_info.title
                            if doc_info.title
                            else doc_info.name,
                            file_meta=doc_info.meta,
                        )
                        match.document_data = document_data

                    # assign table data
                    if match.block_type == "table":
                        try:
                            match.table_data = pickle.loads(res.get("table_data", None))
                            match.table_index = res.get("table_index", [])
                        except Exception:
                            self.logger.error("failed to parse pandas table.")
                            match.table_data = pd.DataFrame()
                            match.table_index = []

                    # record parent
                    parents[match.oid] = match

                    if task.group_by_file:
                        if res["file_idx"] not in criteria.matches:
                            criteria.matches[res["file_idx"]] = []
                        criteria.matches[res["file_idx"]].append(match)

                        if res["file_idx"] not in group_task.documents:
                            group_task.documents[res["file_idx"]] = None
                    else:
                        if "all_files" not in criteria.matches:
                            criteria.matches["all_files"] = []
                        criteria.matches["all_files"].append(match)

                for _, matches in criteria.matches.items():
                    for match in matches:
                        # build qa_text based on the return
                        # load table
                        # print("match....", match.match_text)
                        if match.block_type == "table":
                            match.match_text = match.block_text = ""
                            match.qa_text = " ".join(match.hierarchy_headers)

                            for table_index in match.table_index:
                                match.qa_text += " " + table_index["text"]
                        # expand qa_text by its parent
                        elif match.parent_oid != match.oid:
                            parent_text = parents[match.parent_oid].match_text.strip()
                            # replace ending punctuation with ":"
                            if parent_text.endswith((",", ".", "!", "?")):
                                parent_text = parent_text[:-1]
                            # QA_text is "parent_text: match_text"
                            match.qa_text = f"{parent_text}: {match.qa_text}"

                        # expand qa_text by explanation
                        elif match.explanation and match.block_type != "header":
                            # QA_text is "explain: match_text"
                            qa_text = []
                            if match.parent_text != "":
                                qa_text.append(match.parent_text)
                                qa_text.append(match.qa_text)
                                match.qa_text = " ".join([x for x in qa_text if x])
                            else:  # if parent text is present, adding the header chain is noisy confuses q & a
                                qa_text.append(
                                    " ".join(
                                        match.explanation.get(
                                            "header_chain_text_terms",
                                            [],
                                        ),
                                    ),
                                )
                                qa_text.append(match.qa_text)
                                match.qa_text = ": ".join([x for x in qa_text if x])

                        # build bulk explain for sif
                        if match.explanation:
                            explained_matches.append(match)

                        # calculate relevancy score
                        match.raw_scores["relevancy_score"] = calculate_relevancy_score(
                            keywords + [criteria.question],
                            f"{' '.join(match.hierarchy_headers)} {match.qa_text}",
                            # f"{match.header_text} {match.qa_text}",
                            block_type=match.block_type,
                        )

                # bulk explain sif match
                if len(explained_matches) > 0:
                    self.elastic_search.add_semantic_explanation(
                        [criteria.question_keywords_text],
                        explained_matches,
                        sif_threshold=sif_threshold,
                    )

                search_time = (default_timer() - search_time) * 1000
                self.logger.debug(f"Search takes {search_time:.2f}ms.")

                self.logger.info(
                    f"SearchEngine found {candidate_length} candidates for {len(criteria.matches)} "
                    f"files for topic {topic}",
                )
                # Correct the number of total if the length of the retrieved candidates is lesser.
                if task.pagination["workspace"].get("total", 0) and len(
                    criteria.matches,
                ) < task.pagination["workspace"].get("total", 0):
                    task.pagination["workspace"]["total"] = len(criteria.matches)

        # ad_hoc search don't have DocumentData yet. retrieving
        if kwargs["ad_hoc"]:
            self.logger.info("Getting document infos for workspace level search")
            doc_infos = nosql_db.get_document_infos_by_ids(
                list(group_task.documents.keys()),
            )
            # create DocumentData for group_task
            for doc_info in doc_infos:
                group_task.documents[doc_info.id] = DocumentData(
                    file_idx=doc_info.id,
                    file_name=doc_info.name,
                    file_title=doc_info.title if doc_info.title else doc_info.name,
                    file_meta=doc_info.meta,
                )

        self.logger.info(f"SearchEngine finished with {len(tasks)} tasks")

        # if self.settings["DEBUG"]:
        #     self.logger.debug(group_task.json(indent=2))
