import logging
import re
from collections import defaultdict
from timeit import default_timer

import numpy as np
from bson.objectid import ObjectId
from elasticsearch import Elasticsearch
from nlm_utils.model_client import EncoderClient
from server.storage import nosql_db

from de_utils.utils import filter_keywords
from de_utils.utils import get_words
from de_utils.utils import NLTK_STOP_WORDS
from de_utils.utils import remove_punctuation
from engines.search_engine.es_client_utils import ESQueryBuilder


class ElasticsearchEngine:
    def __init__(self, url=None, settings: dict = {}):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.settings = settings

        if self.settings["DEBUG"]:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        if self.settings["ES_SECRET"]:
            self.client = Elasticsearch(
                [url],
                http_auth=("elastic", self.settings["ES_SECRET"]),
                timeout=3600,
                http_compress=True,
            )
        else:
            self.client = Elasticsearch([url], timeout=3600, http_compress=True)

        self.encoder = EncoderClient(
            model="sif",
            url=self.settings["MODEL_SERVER_URL"],
        )

        self.dpr_question_encoder = EncoderClient(
            model="dpr-question",
            url=self.settings["DPR_MODEL_SERVER_URL"],
            normalization=True,
            lower=True,
            dummy_number=False,
            use_msgpack=True,
            retry=1,
        )

        self.dpr_context_encoder = EncoderClient(
            model="dpr-context",
            url=self.settings["DPR_MODEL_SERVER_URL"],
            normalization=True,
            lower=True,
            dummy_number=False,
            use_msgpack=True,
            retry=1,
        )

        self.highlight_regex = re.compile(r"<_HIGHLIGHT_>(.+?)</_HIGHLIGHT_>")

    def add_semantic_explanation(
        self,
        templates,
        matches,
        sif_threshold,
    ):
        wall_time = default_timer()
        if len(matches) == 0:
            self.logger.info("semantic matching has no results to match")
            return
        template_uniq_words = list(set(" ".join(templates).split()))
        if not template_uniq_words:
            return
        all_words = []

        # get sif scores by combining all words - extend this to all words
        key_field_names = ["header", "match"]
        # first collect all words from header and match
        all_match_values = []  # store dictionary match values here
        for match in matches:
            match_values = match.dict(include={"header_text", "match_text"})
            all_match_values.append(match_values)
            for key_field in key_field_names:
                field_name = f"{key_field}_text"
                # get words in the field and remove stop words
                uniq_words = filter_keywords(
                    get_words(match_values[field_name]),
                    NLTK_STOP_WORDS,
                )
                # now also remove anything that appeared in the keyword match already
                uniq_words = filter_keywords(
                    uniq_words,
                    [x.lower() for x in match.explanation[f"{field_name}_terms"]],
                )
                match_values[f"{field_name}_uniq"] = uniq_words
                all_words.extend(uniq_words)

        # these are all lower case
        all_words = list(set(all_words))
        if not all_words:
            return

        # now get embeddings of all unique words in either header or match
        sims = self.encoder(
            sentences_a=template_uniq_words,
            sentences_b=all_words,
            compare=True,
        )["sims"]
        keyword_idx, matchword_idx = np.where(sims > sif_threshold)

        # for every word in match or header, get the keyword it is closest to
        match_2_keyword = {}
        for i, j in zip(matchword_idx, keyword_idx):
            if not all_words[i] in match_2_keyword:
                match_2_keyword[all_words[i]] = set()
            match_2_keyword[all_words[i]].add(template_uniq_words[j])

        # now iterate through matches and pick the keyword if it is close
        for match, match_values in zip(matches, all_match_values):
            for key_field in key_field_names:
                # get all words in fields that already don't have a keyword match
                field_name = f"{key_field}_text"
                semantic_terms_key = f"{key_field}_semantic_terms"
                original_words_in_field = remove_punctuation(
                    match_values[field_name],
                ).split()
                filtered_words_in_field = match_values[f"{field_name}_uniq"]
                for word in original_words_in_field:
                    if (
                        word.lower() in filtered_words_in_field
                        and word.lower() in match_2_keyword
                    ):  # significant word
                        match.explanation[semantic_terms_key].append(
                            word,
                        )  # [word, match_2_keyword[word]])
                match.explanation[semantic_terms_key] = list(
                    set(match.explanation[semantic_terms_key]),
                )
        wall_time = (default_timer() - wall_time) * 1000
        self.logger.debug(f"Add Semantic Explanation takes {wall_time:.2f}ms")

    def add_keywords_explanation(
        self,
        match,
        explanation,
    ):
        wall_time = default_timer()

        def flatten(details, detail_list):
            for detail in details:
                detail_list.append(
                    {"description": detail["description"], "value": detail["value"]},
                )
                if "details" in detail:
                    flatten(detail["details"], detail_list)

        detail_list = [
            {"description": explanation["description"], "value": explanation["value"]},
        ]
        details = explanation["details"]
        flatten(details, detail_list)
        # field name and weight of the field
        term_weights = []
        # all the terms matching a given field
        field_terms = {}
        sif_score = 0.0
        dpr_score = 0.0
        for detail in detail_list:
            description = detail["description"]
            if description.startswith("weight"):
                keys = detail["description"].split()[0].split(":")
                field = keys[0][7:]
                if len(keys) <= 1:
                    continue
                term = keys[1]
                term_weights.append(
                    {"field": field, "term": term, "score": detail["value"]},
                )
                if field not in field_terms:
                    field_terms[field] = [term]
                else:
                    field_terms[field].append(term)
            elif description.startswith("script score"):
                if "embeddings.sif.match" in description:
                    sif_score = detail["value"]
                elif "embeddings.dpr.match" in description:
                    dpr_score = detail["value"]

        explanation = {
            "term_weights": term_weights,
            "header_text_terms": [],
            "match_text_terms": [],
            "block_text_terms": [],
            "title_text_terms": [],
            "entity_types_terms": [],
            "all_matching_terms": [],
            "header_semantic_terms": [],
            "match_semantic_terms": [],
        }
        for field_name, lemma_list in field_terms.items():
            key = field_name + "_terms"
            if key in explanation:
                if field_name in match:
                    text = match[field_name]
                    for term in lemma_list:
                        pattern_match = re.search(rf"\b{term}\w*", text, re.IGNORECASE)
                        if pattern_match:
                            matching_word = pattern_match.group()
                            explanation[key].append(matching_word)
                            explanation["all_matching_terms"].append(
                                matching_word.lower(),
                            )
                elif field_name == "entity_types":
                    explanation[key].append(" ".join(lemma_list))

        explanation["sif_score"] = sif_score
        explanation["dpr_score"] = dpr_score

        explanation["field_terms"] = field_terms
        for key, value in explanation.items():
            if key.endswith("_terms"):
                explanation[key] = list(set(explanation[key]))

        filtered_block_terms = []
        for term in explanation["block_text_terms"]:
            if not (
                term in explanation["header_text_terms"]
                or term in explanation["match_text_terms"]
            ):
                filtered_block_terms.append(term)
        explanation["block_text_terms"] = filtered_block_terms
        # if match is a header, transfer match explanation to header
        # if match.block_type == "header":
        #     print(">>>>> header match", explanation)
        #     explanation["header_text_terms"] = explanation["match_text_terms"]
        #     explanation["header_semantic_terms"] = explanation["match_semantic_terms"]
        #     explanation["match_text_terms"] = []
        #     explanation["match_semantic_terms"] = []

        wall_time = (default_timer() - wall_time) * 1000
        self.logger.debug(f"Add Explanation takes {wall_time:.2f}ms")
        return explanation

    def get_candidates(
        self,
        questions,
        keywords,
        headers,
        workspace_idx,
        task,
        criteria,
        entity_type=None,
        page_range=None,
        file_idx=[],
        file_filter_struct=None,
        file_filter_text=None,
        sif_threshold=0.45,
        dpr_threshold=0,
        matches_per_doc=20,
        offset=0,
        doc_per_page=20,
        table_flag="auto",
        settings=None,
        entity_types=None,
        group_by_file=True,
        block_types=None,
        search_type="extraction",
        user_acl=None,
    ):
        wall_time = default_timer()
        query_sif_embeddings = []
        settings = settings or {}
        user_acl = user_acl or []
        linguistic_keywords = []
        if criteria.linguistic_keywords:
            linguistic_keywords = criteria.linguistic_keywords

        if (
            questions
            and len(questions) > 0
            and criteria.enable_similar_search
            and not self.settings["USE_DPR"]
        ):
            query_sif_embeddings = self.encoder(questions)["embeddings"]

        # Perform DPR only if we have enabled USE_DPR env variable or "perform_only_dpr"
        # workspace_settings and if there is a question in the criteria
        perform_only_dpr = (
            (
                self.settings["USE_DPR"]
                or settings.get("search_settings", {}).get("perform_only_dpr", False)
            )
            and criteria.enable_similar_search
            and questions
            and len(questions) > 0
            and criteria.is_question
        )
        if perform_only_dpr:

            def process_question(question):
                if question.endswith((".", "?")):
                    return question[:-1]
                return question

            query_dpr_embeddings = (
                self.dpr_question_encoder([process_question(x) for x in questions])[
                    "embeddings"
                ]
                # + self.dpr_context_encoder([x for x in template])["embeddings"]
            )
        else:
            query_dpr_embeddings = []

        filter_texts = {}

        filter_sif_embeddings = {}

        booster_texts = {}

        booster_sif_embeddings = {}

        if headers and len(headers) > 0:
            filter_texts["header_chain_text"] = headers
            booster_texts["header_chain_text"] = headers
            booster_texts["header_text"] = headers
            # Enable SIMILAR Search only when there are questions
            if questions and len(questions) > 0 and criteria.enable_similar_search:
                booster_sif_embeddings["header"] = self.encoder(headers)["embeddings"]

        if (
            self.settings["USE_QATYPE"]
            and entity_type
            and criteria.is_question
            and not criteria.is_bool_question
        ):
            booster_texts["entity_types"] = [entity_type.replace(":", " ")]

        # print("Query questions: ", questions)
        # print("Filter Text: ", filter_texts)
        # print("Booster Text: ", booster_texts)
        # print("Keywords: ", keywords)
        # print("Filter Entity Types: ", entity_types)
        # print("File Linguistic Filters: ", linguistic_keywords)
        # print("File Linguistic Direct Filters: ", criteria.linguistic_direct_keywords)

        query_builder = ESQueryBuilder(
            query_texts=questions,
            keywords=keywords or criteria.linguistic_direct_keywords,
            query_sif_embeddings=query_sif_embeddings,
            query_dpr_embeddings=query_dpr_embeddings,
            criteria=criteria,
            filter_texts=filter_texts,
            filter_sif_embeddings=filter_sif_embeddings,
            booster_texts=booster_texts,
            booster_sif_embeddings=booster_sif_embeddings,
            page_range=page_range,
            sif_threshold=sif_threshold,
            dpr_threshold=dpr_threshold,
            matches_per_doc=matches_per_doc,
            doc_per_page=doc_per_page,
            file_filter_text=file_filter_text,
            offset=offset,
            table_flag=table_flag,
            settings=settings,
            filter_entity_types=entity_types,
            filter_block_types=block_types,
            search_type=search_type,
            perform_only_dpr=perform_only_dpr,
        )
        file_infos = {}

        es_index = workspace_idx
        file_boosters = settings.get("search_settings", {}).get("file_boosters", [])
        enable_file_dynamic_boosters_for_questions = settings.get(
            "search_settings",
            {},
        ).get(
            "enable_file_dynamic_boosters_for_questions",
            False,
        )

        file_dynamic_boosters = settings.get("search_settings", {}).get(
            "file_dynamic_boosters",
            {},
        )

        # Enable dynamic file boosters for questions if configured.
        if criteria.is_question and not enable_file_dynamic_boosters_for_questions:
            file_dynamic_boosters = {}

        doc_per_page = settings.get("search_settings", {}).get(
            "doc_per_page",
            doc_per_page,
        )
        enable_access_control = settings.get("search_settings", {}).get(
            "enable_access_control",
            False,
        )
        user_access_list = None
        if enable_access_control:
            user_access_list = user_acl

        if settings:
            es_index = settings.get("index_settings", {}).get("index", workspace_idx)

        if file_idx:
            query_builder.filter_by_file_idxs(file_idx)
        elif group_by_file and settings.get("index_settings", {}).get(
            "create_file_level_index",
            True,
        ):
            file_level_filter_time = default_timer()
            filter_file_ids = None
            total_match_count = None
            if file_filter_struct:
                filter_file_ids = file_filter_struct.get("results", None)
                total_match_count = file_filter_struct.get("totalMatchCount", None)
            if total_match_count == 0 and filter_file_ids == []:
                self.logger.info(
                    f"{self.__class__.__name__} Field Filters are empty. No files to be matched.",
                )
                return
            first_phase_results = 0

            if linguistic_keywords and not keywords:
                file_level_query = query_builder.build_file_level_query(
                    filter_file_ids,
                    file_dynamic_boosters,
                    user_access_list=user_access_list,
                    linguistic_keywords=linguistic_keywords,
                )
                if file_level_query and file_boosters:
                    if file_level_query.get("query", {}).get("bool", {}):
                        if not file_level_query["query"]["bool"].get("should", []):
                            file_level_query["query"]["bool"]["should"] = []
                        file_level_query["query"]["bool"]["should"].extend(
                            file_boosters,
                        )
                self.logger.info(
                    f"{self.__class__.__name__} "
                    f"Performing File level Query with {linguistic_keywords}. ",
                )
                try:
                    search_time = default_timer()
                    response = self.client.search(
                        index=f"{es_index}_file_level",
                        body=file_level_query,
                        from_=offset,
                        size=doc_per_page,
                        timeout="3600s",
                        explain=False,
                        filter_path=[
                            "took",
                            "hits.hits._source.file_idx",
                            "hits.hits._score",
                            "hits.total",
                        ],
                    )
                    task.pagination["workspace"] = {
                        "total": response["hits"]["total"]["value"],
                        "offset": offset,
                        "result_per_page": doc_per_page,
                    }
                    if response["hits"]["total"]["value"] <= offset:
                        return
                    for file_info in response["hits"]["hits"]:
                        _file_idx = file_info["_source"]["file_idx"]
                        file_infos[_file_idx] = {
                            "file_score": file_info.get("_score", 0),
                            "highlight": {},
                        }
                    file_info_keys = list(file_infos.keys())
                    if file_info_keys:
                        query_builder.filter_by_file_idxs(file_info_keys)
                        first_phase_results = len(file_info_keys)

                    search_time = (default_timer() - search_time) * 1000

                    self.logger.info(
                        f"{self.__class__.__name__} "
                        f"Received {first_phase_results} results from First Phase File Index. "
                        f"Wall time: {search_time:.2f}ms. "
                        f"Actual elastic search time: {response['took']}",
                    )

                except Exception as e:
                    self.logger.error(
                        f"Error during query from elastic search with query:\n{file_level_query}",
                        exc_info=True,
                    )
                    raise e

                file_level_filter_time = (
                    default_timer() - file_level_filter_time
                ) * 1000
                self.logger.debug(
                    f"First Phase File level filter takes {file_level_filter_time:.2f}ms",
                )

            # Fallback to the default method.
            if not first_phase_results:
                file_level_query = query_builder.build_file_level_query(
                    filter_file_ids,
                    file_dynamic_boosters,
                    user_access_list=user_access_list,
                )
                if file_level_query and file_boosters:
                    if file_level_query.get("query", {}).get("bool", {}):
                        if not file_level_query["query"]["bool"].get("should", []):
                            file_level_query["query"]["bool"]["should"] = []
                        file_level_query["query"]["bool"]["should"].extend(
                            file_boosters,
                        )

                try:
                    search_time = default_timer()
                    response = self.client.search(
                        index=f"{es_index}_file_level",
                        body=file_level_query,
                        from_=offset,
                        size=doc_per_page,
                        timeout="3600s",
                        explain=False,
                        filter_path=[
                            "took",
                            # "hits.hits._explanation",
                            "hits.hits._source.file_idx",
                            "hits.hits._score",
                            "hits.total",
                            # "hits.hits.highlight",
                        ],
                    )
                    task.pagination["workspace"] = {
                        "total": response["hits"]["total"]["value"],
                        "offset": offset,
                        "result_per_page": doc_per_page,
                    }
                    if response["hits"]["total"]["value"] <= offset:
                        return
                    for file_info in response["hits"]["hits"]:
                        _file_idx = file_info["_source"]["file_idx"]
                        file_infos[_file_idx] = {
                            "file_score": file_info.get("_score", 0),
                            "highlight": {},
                        }
                    # print("File Level Query Response: ", list(file_infos.keys()))
                    query_builder.filter_by_file_idxs(list(file_infos.keys()))

                    search_time = (default_timer() - search_time) * 1000

                    self.logger.info(
                        f"{self.__class__.__name__} Received results from ElasticSearch. Wall time: {search_time:.2f}ms. "
                        f"Actual elastic search time: {response['took']}",
                    )

                except Exception as e:
                    self.logger.error(
                        f"Error during query from elastic search with query:\n{file_level_query}",
                        exc_info=True,
                    )
                    raise e

                file_level_filter_time = (
                    default_timer() - file_level_filter_time
                ) * 1000
                self.logger.debug(
                    f"File level filter takes {file_level_filter_time:.2f}ms",
                )

        self.logger.info(
            f"Getting candidates for workspace {workspace_idx}, files {file_idx}",
        )

        query = query_builder.build_query()
        agg_prefix = "aggregations.matches.buckets.docs." if group_by_file else ""
        if es_index != workspace_idx:
            query_builder.filter_by_workspace_idx(workspace_idx)
        # print("Workspace Query: ", query)
        try:
            search_time = default_timer()
            response = self.client.search(
                index=es_index,
                body=query,
                from_=(offset if file_idx else 0) if group_by_file else offset,
                size=matches_per_doc if group_by_file else doc_per_page,
                timeout="3600s",
                explain=False,
                filter_path=[
                    "took",
                    "hits.total",
                    "hits.hits._id",
                    # "hits.hits._explanation",
                    f"{agg_prefix}hits.hits._source.child_idxs",
                    f"{agg_prefix}hits.hits._source.file_idx",
                    f"{agg_prefix}hits.hits._id",
                    f"{agg_prefix}hits.hits._score",
                    f"{agg_prefix}hits.hits._source.*_text",
                    f"{agg_prefix}hits.hits._source.block_type",
                    f"{agg_prefix}hits.hits._source.block_idx",
                    f"{agg_prefix}hits.hits._source.entity_types",
                    f"{agg_prefix}hits.hits.inner_hits",  # used by tables
                    f"{agg_prefix}hits.hits.highlight",
                ],
                ignore_unavailable=True,
            )
            search_time = (default_timer() - search_time) * 1000

            self.logger.info(
                f"{self.__class__.__name__} Received results from ElasticSearch. Wall time: {search_time:.2f}ms. "
                f"Actual elastic search time: {response['took']}",
            )
            # print("Response is ", response)

        except Exception as e:
            self.logger.error(
                f"Error during query from elastic search with query:\n{query}",
                exc_info=True,
            )
            raise e

        # # go through hits and get their explanation first
        # hit_explanations = {}
        # # self.logger.info(f"--------hits {response['hits']}")
        # if "hits" in response:
        #     if "hits" in response["hits"]:
        #         hits = response["hits"]["hits"]
        #         for hit in hits:
        #             # print("adding explanation...", hit["_explanation"])
        #             hit_explanations[hit["_id"]] = hit["_explanation"]
        # # print("all explanations: ", hit_explanations)
        candidates_idxs2hit = defaultdict()
        best_candidates_score_by_file = defaultdict(lambda: 1)

        child_to_parent = {}
        table_inner_hits = defaultdict(list)
        highlights = {}

        if group_by_file:
            if (
                "aggregations" in response
                and "buckets" in response["aggregations"]["matches"]
            ):
                # get candidates_idxs and scores
                for bucket in response["aggregations"]["matches"]["buckets"]:
                    # print(bucket["docs"]["hits"])
                    hits = bucket["docs"]["hits"]["hits"]
                    self.process_hits(
                        search_type,
                        hits,
                        group_by_file,
                        best_candidates_score_by_file,
                        candidates_idxs2hit,
                        child_to_parent,
                        highlights,
                        questions,
                        table_inner_hits,
                    )

        elif "hits" in response:
            if "hits" in response["hits"]:
                hits = response["hits"]["hits"]
                task.pagination["workspace"]["total"] = response["hits"]["total"][
                    "value"
                ]

                self.process_hits(
                    search_type,
                    hits,
                    group_by_file,
                    best_candidates_score_by_file,
                    candidates_idxs2hit,
                    child_to_parent,
                    highlights,
                    questions,
                    table_inner_hits,
                )
                # for hit in hits:
                #     # print("adding explanation...", hit["_explanation"])
                #     hit_explanations[hit["_id"]] = hit["_explanation"]

        post_search_wall_time = default_timer()

        # skip if no candidate returned
        if len(candidates_idxs2hit.items()) > 0:
            # get candidates
            candidates_idxs2hit = {
                ObjectId(key): value for key, value in candidates_idxs2hit.items()
            }
            candidates = nosql_db.get_es_entry(
                list(candidates_idxs2hit.keys()),
                workspace_idx,
            )

            for candidate in candidates:
                # expand the context window if needed
                if criteria.before_context_window or criteria.after_context_window:
                    match_idx_list = [
                        x
                        for x in range(
                            candidate["match_idx"] - criteria.before_context_window,
                            candidate["match_idx"] + criteria.after_context_window + 1,
                        )
                    ]
                    projection = {
                        "_id": 0,
                        "match_text": 1,
                    }
                    matched_candidates = nosql_db.get_match_es_entry(
                        workspace_idx,
                        candidate["file_idx"],
                        match_idx_list,
                        None,
                        projection,
                    )
                    new_string = " ".join(c["match_text"] for c in matched_candidates)
                    candidate["match_text"] = new_string
                hit = candidates_idxs2hit[candidate["_id"]]
                # remove tables from db retrieved by parent.
                if (
                    criteria.table_flag == "disable"
                    and candidate["block_type"] == "table"
                ):
                    continue
                candidate["block_idx"] = hit["_source"]["block_idx"]
                candidate["raw_match_score"] = hit["_score"]
                candidate["entity_types"] = hit["_source"]["entity_types"]
                # scale match_score to [0,1]
                candidate["match_score"] = min(
                    1,
                    hit["_score"]
                    / best_candidates_score_by_file[candidate["file_idx"]],
                )

                # convert ObjectId to str
                candidate["_id"] = str(candidate["_id"])

                if candidate["_id"] in child_to_parent:
                    candidate["parent_uid"] = child_to_parent[candidate["_id"]]
                    candidate["is_child"] = True
                else:  # parent candidate point to itself.
                    candidate["parent_uid"] = candidate["_id"]
                    candidate["is_child"] = False

                # explanation = None
                # if candidate["_id"] in hit_explanations:
                #     explanation = self.add_keywords_explanation(
                #         candidate,
                #         hit_explanations[candidate["_id"]],
                #     )
                # elif candidate["parent_uid"] in hit_explanations:
                #     explanation = self.add_keywords_explanation(
                #         candidate,
                #         hit_explanations[candidate["parent_uid"]],
                #     )

                # else:
                candidate["explanation"] = {
                    "term_weights": [],
                    "header_text_terms": [],
                    "match_text_terms": [],
                    "block_text_terms": [],
                    "title_text_terms": [],
                    "all_matching_terms": [],
                    "header_semantic_terms": [],
                    "match_semantic_terms": [],
                }

                # use highlight to enhance explanation
                # parent id is same as candidate id when candidate is parent
                candidate_highlights = highlights.get(candidate["parent_uid"], {})

                for field, terms in candidate_highlights.items():
                    key = f"{field}_terms"
                    if key in candidate["explanation"]:
                        for term in terms:
                            if term not in candidate["explanation"][key]:
                                candidate["explanation"][key].append(term)
                    else:
                        candidate["explanation"][key] = terms
                # candidate['highlight'] = highlights[candidate['_id']]

                # print("\n---match---", candidate["is_child"])
                # print(candidate["match_text"])
                # print("---header---")
                # print(candidate["header_text"])
                # print("---explanation---")
                # print(explanation)
                # print("---highlights---")
                # print(candidate["explanation"])

                # assign table_info
                if candidate["block_type"] == "table" and "table_data" in candidate:
                    # TODO: do we need to return a table matched by header name only?
                    # edge case: table retrieved by sif that has no keywords hit, skipping
                    if (
                        candidate["_id"] not in table_inner_hits
                        and candidate["_id"] == candidate["parent_uid"]
                    ):
                        continue
                    try:
                        candidate["table_index"] = table_inner_hits[candidate["_id"]]

                    except Exception as e:
                        self.logger.error(
                            f"Error when loading dataframe {e}",
                            exc_info=True,
                        )
                        continue

                # assign file_score for workspace_level search
                if file_infos:
                    file_info = file_infos[candidate["file_idx"]]
                    candidate["file_score"] = file_info["file_score"]
                    # for field, terms in file_info['highlight'].items():
                    #     candidate["explanation"][key] = terms
                else:
                    candidate["file_score"] = 0

                yield candidate
            post_search_wall_time = (default_timer() - post_search_wall_time) * 1000
            self.logger.debug(
                f"{self.__class__.__name__} Post Search wall time: {post_search_wall_time:.2f}ms",
            )

        wall_time = (default_timer() - wall_time) * 1000

        self.logger.info(
            f"{self.__class__.__name__} Finished. Wall time: {wall_time:.2f}ms",
        )

    def process_hits(
        self,
        search_type,
        hits,
        group_by_file,
        best_candidates_score_by_file,
        candidates_idxs2hit,
        child_to_parent,
        highlights,
        questions,
        table_inner_hits,
    ):
        for hit in hits:
            current_hit_idx = hit["_id"]
            # record and save the score for current hit
            candidates_idxs2hit[current_hit_idx] = hit
            include_children = search_type not in ["relation-triple", "relation-node"]
            # Expand child_idxs in case we are in Query MODE and if block_type is not para
            if include_children and not hit["_source"]["block_type"] == "para":
                for child_idx in hit["_source"]["child_idxs"]:
                    # Child has better score than its parent if child is already retrieved
                    # Only assign child to its parent if child score lower than its parent
                    if child_idx not in candidates_idxs2hit:
                        # child share parent match_score
                        candidates_idxs2hit[child_idx] = hit
                    if child_idx not in child_to_parent:
                        # assign child to its parent
                        child_to_parent[child_idx] = current_hit_idx

            # get highlight text from ES
            highlight = hit.get("highlight", {})
            for field in [
                "match_text",
                "header_text",
                "header_chain_text",
            ]:
                if field in highlight:
                    highlight[field] = re.findall(
                        self.highlight_regex,
                        " ".join(highlight[field]),
                    )
                else:
                    highlight[field] = []

            # in case of a header, matches show up in the match_text
            # If no match_text, then the match might be in header_chain_text.
            if hit["_source"]["block_type"] == "header":
                highlight["header_text"].extend(
                    highlight["match_text"]
                    if highlight["match_text"]
                    else highlight.get("header_chain_text", []),
                )

            highlights[current_hit_idx] = highlight

            # retrieve table index/columns
            for inner_hits in hit.get("inner_hits", {}).values():
                for inner_hit in inner_hits["hits"]["hits"]:
                    table_inner_hits[current_hit_idx].append(
                        {
                            "score": inner_hit["_score"],
                            "type": inner_hit["_source"]["type"],
                            # "text": inner_hit["_source"].get("text", ""),
                            # "index": inner_hit["_source"].get("index", []),
                            "text": inner_hit["_source"]["text"],
                            "idx": inner_hit["_source"]["idx"],
                        },
                    )

            if group_by_file:
                # Record best score by file_idx.
                # The value will be used to normalize raw_match_score into match_score
                # if current_hit_idx not in table_inner_hits:
                best_candidates_score_by_file[hit["_source"]["file_idx"]] = max(
                    best_candidates_score_by_file[hit["_source"]["file_idx"]],
                    hit["_score"],
                )
