import copy
import re
from math import e

from nlm_utils.utils import query_preprocessing as preprocess
from nltk import ngrams


phrase_search_rule = re.compile(r'"([^"]*)"')
# Borrowed from line_parser of Ingestor.
phrase_words_rule = re.compile(
    r'(?:(?<=\W)|(?<=^))["\']+(?!\D\s)(?!\d+)(.*?)[,;.]?["\']+',
)


class ESQueryBuilder:
    def __init__(
        self,
        query_texts,
        keywords,
        query_sif_embeddings,
        query_dpr_embeddings,
        criteria,
        filter_texts={},
        filter_sif_embeddings={},
        booster_texts={},
        booster_sif_embeddings={},
        page_range=None,
        sif_threshold=0.45,
        dpr_threshold=0,
        matches_per_doc=10,
        doc_per_page=None,
        offset=0,
        table_flag="auto",
        file_filter_text=None,
        settings=None,
        filter_entity_types=None,
        filter_block_types=[],
        search_type="extraction",
        perform_only_dpr=False,
    ):
        self.matches_per_doc = matches_per_doc
        self.sif_threshold = sif_threshold
        self.dpr_threshold = dpr_threshold

        self.sif_filter_threshold = 0.9

        self.table_flag = table_flag

        self.doc_per_page = doc_per_page or 10_000
        self.offset = offset

        self.main_query = []
        self.booster_query = []
        self.filter_query = []
        self.must_not_query = []
        self.keyword_query = []
        self.must_not_keywords = []
        self.file_level_query = []
        self.file_filter_text = file_filter_text or ""
        self.file_filter_query = []
        self.settings = settings
        self.filter_block_types = filter_block_types
        self.search_type = search_type
        self.criteria = criteria
        self.perform_only_dpr = perform_only_dpr
        self.is_debug_search = self.settings.get("search_settings", {}).get(
            "debug_search",
            False,
        )

        enable_similar_search = True
        if criteria.enable_similar_search is not None:
            enable_similar_search = criteria.enable_similar_search

        self.add_bm25_query(
            query_texts,
            booster_texts,
            filter_texts,
            keywords,
            filter_entity_types,
        )

        # if not phrase search, we are adding sif
        if (
            enable_similar_search
            and query_texts
            and not any([phrase_search_rule.match(x) for x in query_texts])
        ):
            if not self.perform_only_dpr:
                self.add_sif_query(
                    query_sif_embeddings,
                    booster_sif_embeddings,
                    filter_sif_embeddings,
                )

            self.add_dpr_query(query_dpr_embeddings)

        if len(self.filter_block_types) > 0:
            self.add_block_type_filter()

        if self.table_flag == "disable":
            self.add_table_filter()
        # Commenting out the original table code, now that we have table_cell logic
        # else:
        #     self.add_table_query(query_texts)

        if page_range:
            self.add_page_range_filter(page_range)

    def filter_by_file_idxs(self, file_idx):
        # single file
        if isinstance(file_idx, str):
            self.filter_query.append({"term": {"file_idx": file_idx}})
        elif isinstance(file_idx, list):
            self.filter_query.append(
                {"terms": {"file_idx": [x for x in file_idx]}},
            )
            # self.filter_query.append(
            #     {"bool": {"should": [{"term": {"file_idx": x}} for x in file_idx]}},
            # )

    def filter_by_workspace_idx(self, workspace_idx):
        if isinstance(workspace_idx, str):
            self.filter_query.append({"term": {"workspace_idx": workspace_idx}})
        elif isinstance(workspace_idx, list):
            self.filter_query.append(
                {"terms": {"workspace_idx": [x for x in workspace_idx]}},
            )

    def add_bm25_query(
        self,
        query_texts,
        boosters={},
        filters={},
        keywords=[],
        filter_entity_types=None,
    ):
        question_keywords = []
        filter_entity_types = filter_entity_types or []
        query_quotes_to_keywords = (
            []
        )  # List of query quotes that we need to consider while filtering
        is_search_workspace = self.settings.get("search_settings", {}).get(
            "is_search_workspace",
            False,
        )

        for query_text in query_texts:
            # Remove any stopwords from the query
            # Initial effort to find the keywords in the query string
            query_text = query_text.lower()
            # Remove punctuation
            preprocess_query_words = preprocess.get_words(query_text)
            # Filter out NLM specific stopwords
            question_keywords = preprocess.filter_keywords(
                preprocess_query_words,
                preprocess.NLM_STOP_WORDS
                if not is_search_workspace
                else preprocess.STOP_WORDS,
            )
            query_keywords_text = " ".join(question_keywords).strip()
            phrase_search_rule_match = False

            # phrase search use simple_query_string
            if phrase_search_rule.match(query_text):
                phrase_search_rule_match = True
                # must query
                if not self.perform_only_dpr:
                    self.main_query.append(
                        {
                            "simple_query_string": {
                                "query": query_text,
                                "fields": ["match_text"],
                            },
                        },
                    )
            else:
                answer_types = boosters.get("entity_types", None)
                if answer_types:
                    for answer_type in answer_types:
                        if not self.perform_only_dpr:
                            self.main_query.append(
                                {
                                    "multi_match": {
                                        "query": answer_type,
                                        "type": "cross_fields",
                                        "fields": [
                                            # Answer types will have more priority
                                            "entity_types^2",
                                        ],
                                    },
                                },
                            )
                    del boosters["entity_types"]
                # all other query use n-gram search
                if len(query_keywords_text) > 0 and not self.perform_only_dpr:
                    self.main_query.append(
                        self.build_ngram_query(
                            " ".join(preprocess_query_words).strip(),
                            query_keywords_text,
                        ),
                    )

            # boost on designed field
            # irrespective of matches, let's add this boost to control ranking
            if not self.perform_only_dpr and not self.is_debug_search:
                self.booster_query.append(
                    {
                        "multi_match": {
                            "query": query_keywords_text,
                            "type": "cross_fields",
                            "fields": [
                                "match_text^2.0",
                                "header_text^1.5",
                                "header_chain_text",
                                "block_text^0.2",
                                "parent_text",
                            ],
                        },
                    },
                )
            query_quote_words = phrase_words_rule.findall(query_text)
            # Add quoted words from query to the list, for applying in filter section
            for quote_word in query_quote_words:
                if quote_word not in query_quotes_to_keywords:
                    query_quotes_to_keywords.append(f'"{quote_word}"')
            if not query_quote_words and not self.criteria.is_question:
                query_quote_words = query_texts
            if query_quote_words:
                combined_quote_words = " ".join(query_quote_words)
                quote_word_query = {
                    "bool": {
                        # uni-gram + bi-gram + tri-gram
                        "should": [
                            {
                                "bool": {
                                    # n-gram_0 + n-gram_1 ... + n-gram_n
                                    "should": [
                                        {
                                            "multi_match": {
                                                "query": " ".join(gram),
                                                "type": "phrase",
                                                "fields": [
                                                    f"key_values^{round((1/e)**i*2,2)}",
                                                ],
                                                "slop": "0",  # do not allow extra token in n-gram
                                            },
                                        }
                                        for gram in ngrams(
                                            combined_quote_words.split(),
                                            i + 1,
                                        )
                                    ],
                                },
                            }
                            for i in range(min(3, len(combined_quote_words.split())))
                        ],
                    },
                }
                if not self.perform_only_dpr:
                    self.booster_query.append(quote_word_query)
            # file_level_query
            if not phrase_search_rule_match:
                self.file_level_query.append(query_keywords_text)
            else:
                self.file_level_query.append(query_text)

        # Add any private dictionary keywords from the question to keywords.
        dict_keywords = []
        if self.settings and self.settings.get(
            "applied_question_private_dictionary",
            {},
        ):
            for key, values in self.settings[
                "applied_question_private_dictionary"
            ].items():
                for v in values:
                    if v not in dict_keywords:
                        dict_keywords.append(v)

        # user specified filter by field
        final_keyword_list = keywords + dict_keywords
        # Check whether we have an existing keyword list.
        # If not make a keyword filter query with the quoted words from the query.
        if not final_keyword_list and query_quotes_to_keywords:
            new_keyword = ""
            for qqw in query_quotes_to_keywords:
                if qqw not in new_keyword:
                    new_keyword += f" {qqw}"
            self.keyword_query.append(
                {
                    "simple_query_string": {
                        "query": new_keyword,
                        "fields": ["match_text"],
                        "default_operator": "and",
                    },
                },
            )
            # file_level_query
            self.file_level_query.append(new_keyword)

        keyword_quote_words = []  # Quote words in keywords.
        # Process the original keyword list.
        for keyword in final_keyword_list:
            if keyword.startswith("not:"):
                keyword = keyword.replace("not:", "").strip()
                self.must_not_keywords.append(keyword)
            else:
                # Add the query quoted words to the filter list.
                new_keyword = keyword
                for qqw in query_quotes_to_keywords:
                    if qqw not in new_keyword:
                        new_keyword += f" {qqw}"
                self.keyword_query.append(
                    {
                        "simple_query_string": {
                            "query": new_keyword,
                            "fields": ["match_text"],
                            "default_operator": "and",
                        },
                    },
                )
                # file_level_query
                self.file_level_query.append(new_keyword)
                # Add original keyword quotes.
                key_quote_words = phrase_words_rule.findall(keyword)
                for item in key_quote_words:
                    if item not in keyword_quote_words:
                        keyword_quote_words.append(item)

        if keyword_quote_words:
            kw_quote_word_query = {
                "bool": {
                    "should": [
                        {
                            "bool": {
                                "should": [
                                    {
                                        "multi_match": {
                                            "query": quote_word,
                                            "type": "phrase",
                                            "fields": [
                                                "key_values^2",  # Boost the occurrence in key-value pairs.
                                            ],
                                            "slop": "0",
                                        },
                                    }
                                    for quote_word in keyword_quote_words
                                ],
                            },
                        },
                    ],
                },
            }
            self.booster_query.append(kw_quote_word_query)

        if self.keyword_query or self.must_not_keywords:
            keyword_must_query = {
                "bool": {
                    "must": [],
                },
            }
            # Match @least one of the keywords.
            keyword_must_query["bool"]["must"].append(
                {
                    "bool": {
                        "should": self.keyword_query,
                    },
                },
            )
            # Do not add must_not_query to file filter query.
            self.file_filter_query.append(copy.deepcopy(keyword_must_query))
            if self.must_not_keywords:
                keyword_must_query["bool"]["must_not"] = [
                    {
                        "simple_query_string": {
                            "query": word,
                            "fields": ["match_text"],
                            "default_operator": "and",
                        },
                    }
                    for word in self.must_not_keywords
                ]

            # If there are no query keywords, then do an additional filter on header block_types
            if not question_keywords:
                keyword_must_query["bool"]["must"].append(
                    {
                        "bool": {
                            "must_not": {"terms": {"block_type": ["header"]}},
                        },
                    },
                )
            self.filter_query.append(keyword_must_query)

        for field_name, texts in filters.items():
            filter_must_not_list = []
            filter_must_query = {
                "bool": {
                    "must": [],
                },
            }
            file_filter_must_query = {
                "bool": {
                    "must": [],
                },
            }
            ind_filter_query = []
            file_filters = []
            for x in texts:
                if x.startswith("not:") and field_name == "header_chain_text":
                    x = x.replace("not:", "").strip()
                    filter_must_not_list.append(
                        {
                            "simple_query_string": {
                                "query": f"({x})",
                                "fields": [field_name, "header_text"],
                                "default_operator": "and",
                            },
                        },
                    )
                else:
                    simple_query = {
                        "simple_query_string": {
                            "query": f"({x})",
                            "fields": [field_name],
                            "default_operator": "and",
                        },
                    }
                    if field_name == "header_chain_text":
                        simple_query["simple_query_string"]["fields"].append(
                            "header_text",
                        )
                    ind_filter_query.append(simple_query)

                    file_filters.append(
                        {
                            "simple_query_string": {
                                "query": f"({x})",
                                "fields": ["header_text"],
                                "default_operator": "and",
                            },
                        },
                    )
                    # file_level_query
                    self.file_level_query.append(x)

            if ind_filter_query:
                filter_must_query["bool"]["must"].append(
                    {
                        "bool": {
                            "should": ind_filter_query,
                        },
                    },
                )
            if filter_must_not_list:
                filter_must_query["bool"]["must_not"] = filter_must_not_list

            if file_filters:
                file_filter_must_query["bool"]["must"].append(
                    {
                        "bool": {
                            "should": file_filters,
                        },
                    },
                )
            self.filter_query.append(filter_must_query)
            self.file_filter_query.append(file_filter_must_query)

        if filter_entity_types:
            # Do an "OR" operation with the entity types.
            filter_entity_str = " | ".join(
                ['"' + x.replace(":", " ") + '"' for x in filter_entity_types],
            )
            filter_entity_must_query = {
                "bool": {
                    "must": {
                        "simple_query_string": {
                            "query": filter_entity_str,
                            "fields": ["entity_types"],
                        },
                    },
                },
            }
            self.filter_query.append(filter_entity_must_query)
            # File level do not have the entity type yet, add once that is done.

        # Booster for Header text / Header Chain text
        if "header_chain_text" in boosters.keys():
            if boosters.get("header_chain_text", "") == boosters.get("header_text", ""):
                texts = boosters.get("header_text")
                outer_dis_max_queries = []
                for text in texts:
                    inner_dis_max_queries = []
                    for field in ["header_text", "header_chain_text"]:
                        field_ngram_query = {
                            "bool": {
                                # uni-gram + bi-gram + tri-gram
                                "should": [
                                    {
                                        "bool": {
                                            # n-gram_0 + n-gram_1 ... + n-gram_n
                                            "should": [
                                                {
                                                    "multi_match": {
                                                        "query": " ".join(gram),
                                                        "type": "phrase",
                                                        "fields": [
                                                            f"{field}^{round((1/e)**i,2)}",
                                                        ],
                                                        "slop": "0",  # do not allow extra token in n-gram
                                                    },
                                                }
                                                for gram in ngrams(text.split(), i + 1)
                                            ],
                                        },
                                    }
                                    for i in range(min(3, len(text.split())))
                                ],
                            },
                        }
                        inner_dis_max_queries.append(field_ngram_query)
                    outer_dis_max_queries.append(
                        {"dis_max": {"queries": inner_dis_max_queries}},
                    )

                self.booster_query.append(
                    {"dis_max": {"queries": outer_dis_max_queries}},
                )
            # Remove the booster fields
            del boosters["header_chain_text"], boosters["header_text"]

        # user specified field booster by field
        for field_name, texts in boosters.items():
            for text in texts:
                self.booster_query.append(
                    {
                        "multi_match": {
                            "query": text,
                            "type": "cross_fields",
                            "fields": [
                                # user specified boost field have more weights
                                f"{field_name}^2",
                            ],
                        },
                    },
                )
                if field_name not in ["header_chain_text", "header_text"]:
                    # file_level_query
                    self.file_level_query.append(text)

    def build_ngram_query(self, query_text, query_keywords_text, max_gram=3):
        """
        This function build a dummy n-gram search by boosting in ES
        example:
        suppose each match give score of 1

        query = "A B C"
        sent1 = "A B C"
        sent2 = "B C A"
        sent3 = "C B A"

        score(query, sent1) = sum(
            # unigram, A, B, C
            [(1/e)^0 + (1/e)^0 + (1/e)^0] = 3.00,
            # bigram
            [2*(1/e)^1 + 2*(1/e)^1] ≈ 1.47
            # trigram
            [3*(1/e)^2] ≈ 0.41
        ) = 4.88

        score(query, sent2) = sum(
            # unigram: "B", "C", "A"
            [(1/e)^0 + (1/e)^0 + (1/e)^0] = 3.00,
            # bigram: "B C"
            [2*(1/e)^1] ≈ 0.74
        ) = 3.74

        score(query, sent3) = sum(
            # unigram: "C", "B", "A"
            [(1/e)^0 + (1/e)^0 + (1/e)^0] = 3.00,
        ) = 3.00
        """
        return {
            "bool": {
                # uni-gram + bi-gram + tri-gram
                "should": [
                    {
                        "bool": {
                            # n-gram_0 + n-gram_1 ... + n-gram_n
                            "should": [
                                {
                                    "multi_match": {
                                        "query": " ".join(gram),
                                        "type": "phrase",
                                        # The weighting factor set to 1/e ≈ 0.36788
                                        #
                                        # Do not set the factor larger than 0.5.
                                        # It will flavor the phrase over tokens.
                                        # e.g. "Executive Officer" > "Officer Executive Chief"
                                        "fields": [
                                            f"match_text^{round((1/e)**i,2)}"
                                            if not self.is_debug_search
                                            else f"full_text^{round((1/e)**i,2)}",
                                            "",
                                            # f"header_text^{round((1/e)**i,2)}",
                                        ],
                                        "slop": "0",  # do not allow extra token in n-gram
                                    },
                                }
                                for gram in ngrams(query_keywords_text.split(), i + 1)
                                if " ".join(gram) in query_text
                            ],
                        },
                    }
                    for i in range(min(max_gram, len(query_keywords_text.split())))
                ],
            },
        }

    def build_query_from_embedding(
        self,
        embedding,
        field,
        embedding_name,
        similarity_threshold,
        scaler=1,
    ):
        filter_query = {"match_all": {}}

        return {
            # use function score to support local min_score
            "function_score": {
                "query": {
                    "script_score": {
                        "query": filter_query,
                        "script": {
                            "source": f"Math.max(dotProduct(params.queryVector, "
                            f"'embeddings.{embedding_name}.{field}') * {scaler},0)",
                            "params": {"queryVector": embedding},
                        },
                    },
                },
                "min_score": similarity_threshold,
            },
        }

    def add_sif_query(
        self,
        embeddings,
        boosters={},
        filters={},
    ):
        # query must satisfied main_query
        for embedding in embeddings:
            self.main_query.append(
                self.build_query_from_embedding(
                    embedding,
                    field="match",
                    embedding_name="sif",
                    similarity_threshold=self.sif_threshold,
                    # scaler=100,
                ),
            )
            self.booster_query.append(
                self.build_query_from_embedding(
                    embedding,
                    field="header",
                    embedding_name="sif",
                    similarity_threshold=self.sif_threshold,
                    # scaler=100,
                ),
            )

        # user specified filter by field
        for field_name, embeddings in filters.items():
            for embedding in embeddings:
                self.filter_query.append(
                    self.build_query_from_embedding(
                        embedding,
                        field=field_name,
                        embedding_name="sif",
                        similarity_threshold=self.sif_filter_threshold,
                        # scaler=100,
                    ),
                )

        # user specified filter by field
        for field_name, embeddings in boosters.items():
            for embedding in embeddings:
                self.booster_query.append(
                    self.build_query_from_embedding(
                        embedding,
                        field=field_name,
                        embedding_name="sif",
                        similarity_threshold=self.sif_threshold,
                    ),
                )

    def add_dpr_query(
        self,
        embeddings,
    ):
        # query must satisfied main_query
        for embedding in embeddings:
            self.booster_query.append(
                self.build_query_from_embedding(
                    embedding,
                    field="match",
                    embedding_name="dpr",
                    similarity_threshold=self.dpr_threshold,
                    scaler=1,
                ),
            )

    def add_table_query(self, texts, max_size=100):
        for idx, text in enumerate(texts):
            self.main_query.append(
                {
                    "bool": {
                        "must": [
                            {"term": {"block_type": {"value": "table", "boost": 0}}},
                            {
                                "nested": {
                                    "path": "table",
                                    # "query": {
                                    #     # search index with table_analyzer
                                    #     "multi_match": {
                                    #         "query": text,
                                    #         "type": "cross_fields",
                                    #         "fields": [
                                    #             "table.index",
                                    #             "table.text^0.5",
                                    #         ],
                                    #     },
                                    # },
                                    "query": {
                                        "bool": {
                                            "must": [
                                                {
                                                    "nested": {
                                                        "path": "table.index",
                                                        "query": {
                                                            # search index with table_analyzer
                                                            # "match": {
                                                            #     "table.index.text": text
                                                            # },
                                                            "multi_match": {
                                                                "query": text,
                                                                "type": "cross_fields",
                                                                "fields": [
                                                                    "table.index.text",
                                                                ],
                                                            },
                                                        },
                                                        "score_mode": "sum",
                                                    },
                                                },
                                                {
                                                    "multi_match": {
                                                        "query": text,
                                                        "type": "cross_fields",
                                                        "fields": [
                                                            # "table.text^0.1",
                                                            "table.index_text^0.2",
                                                        ],
                                                    },
                                                },
                                            ],
                                        },
                                    },
                                    # table score is the best column match score
                                    "score_mode": "max",
                                    "inner_hits": {
                                        "name": f"inner_hits_{idx}",
                                        "size": max_size,
                                    },
                                },
                            },
                        ],
                    },
                },
            )

    def add_table_filter(self):
        self.must_not_query.append({"terms": {"block_type": ["table", "table_cell"]}})

    def add_block_type_filter(self):
        self.main_query.append({"terms": {"block_type": self.filter_block_types}})

    def build_query(self, group_by_file=True):
        aggs = {
            "matches": {
                # "execution_hint": "map" ==> without leveraging global ordinals
                # "file_idx" is a high cardinal field.
                "terms": {
                    "field": "file_idx",
                    "size": self.doc_per_page,
                    "execution_hint": "map",
                },
                "aggs": {
                    "docs": {
                        "top_hits": {
                            "size": self.matches_per_doc,
                            # highlight the keywords
                            "highlight": {
                                "fields": {
                                    "match_text": {},
                                    "header_text": {},
                                    "header_chain_text": {},
                                },
                                "pre_tags": ["<_HIGHLIGHT_>"],
                                "post_tags": ["</_HIGHLIGHT_>"],
                            },
                        },
                    },
                },
            },
        }
        query = {
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "must": [
                                # must contain one of the main _query
                                {
                                    "dis_max": {
                                        "queries": self.main_query
                                        or [{"match_all": {}}],
                                        "tie_breaker": 0.5,
                                    },
                                },
                            ],
                            "filter": self.filter_query,
                            "must_not": self.must_not_query,
                            "should": self.booster_query,
                            # candidate should match at least one type of search
                        },
                    },
                    # sum scores of different query
                    "score_mode": "sum",
                },
            },
            "highlight": {
                "fields": {
                    "match_text": {},
                    "header_text": {},
                    "header_chain_text": {},
                },
                "pre_tags": ["<_HIGHLIGHT_>"],
                "post_tags": ["</_HIGHLIGHT_>"],
            },
            "aggs": aggs if group_by_file else {},
        }
        return query

    def add_page_range_filter(self, page_range):
        if page_range[0] > 0 and page_range[1] > 0:
            self.filter_query.append(
                {
                    "range": {
                        "page_idx": {
                            "gte": page_range[0] - 1,
                            "lte": page_range[1] - 1,
                            "boost": 0,
                        },
                    },
                },
            )
        elif page_range[0] < 0 and page_range[1] < 0:
            self.filter_query.append(
                {
                    "range": {
                        "reverse_page_idx": {
                            "gte": page_range[0],
                            "lte": page_range[1],
                            "boost": 0,
                        },
                    },
                },
            )

    def build_file_level_query(
        self,
        filter_file_ids=None,
        file_dynamic_boosters=None,
        user_access_list=None,
        linguistic_keywords=None,
    ):
        filter_query = []
        must_not_query = []
        filter_file_ids = filter_file_ids or []
        file_level_combined_words = ""
        if self.file_filter_query and not linguistic_keywords:
            filter_query = self.file_filter_query
        if linguistic_keywords:
            keyword_query = []
            for keyword in linguistic_keywords:
                keyword_query.append(
                    {
                        "simple_query_string": {
                            "query": keyword,
                            "fields": ["match_text"],
                            "default_operator": "and",
                        },
                    },
                )
            filter_query.append(
                {
                    "bool": {
                        "must": [
                            {
                                "bool": {
                                    "should": keyword_query,
                                },
                            },
                        ],
                    },
                },
            )
        if self.file_filter_text:
            filter_query.append(
                {
                    "multi_match": {
                        "query": self.file_filter_text,
                        "type": "cross_fields",
                        "fields": [
                            "header_text^1.5",
                            "title_text^2",
                            "match_text",
                        ],
                    },
                },
            )
        if filter_file_ids:
            filter_query.append(
                {"terms": {"file_idx": [x for x in filter_file_ids]}},
            )
        if user_access_list is not None:
            filter_should_query = []
            if user_access_list:
                # (1) Match either one of the access control mentioned
                filter_should_query.append(
                    {
                        "terms": {
                            "meta.permit_list": [x for x in user_access_list],
                        },
                    },
                )
                # (2) Or Match where the permit list is empty
                filter_should_query.append(
                    {
                        "bool": {
                            "must_not": [
                                {
                                    "exists": {
                                        "field": "meta.permit_list",
                                    },
                                },
                            ],
                        },
                    },
                )
                filter_query.append(
                    {
                        "bool": {
                            "should": filter_should_query,
                        },
                    },
                )
                # (3) deny list should not contain user access control
                must_not_query.append(
                    {
                        "terms": {
                            "meta.deny_list": [x for x in user_access_list],
                        },
                    },
                )
            else:
                # (1) Match where the permit list is empty
                must_not_query.append(
                    {
                        "exists": {
                            "field": "meta.permit_list",
                        },
                    },
                )
        query = {
            "query": {
                "bool": {
                    "must": [],
                    "filter": filter_query,
                    "must_not": must_not_query,
                },
            },
            # "highlight": {
            #     "fields": {"match_text": {}, "header_text": {}, "title_text": {}},
            #     "pre_tags": ["<_HIGHLIGHT_>"],
            #     "post_tags": ["</_HIGHLIGHT_>"],
            # },
        }
        if self.file_level_query or self.file_filter_text:
            # Initial query for file level index.
            # query["query"]["bool"]["must"].append(
            #     {
            #         "multi_match": {
            #             "query": " ".join(
            #                 self.file_level_query + [self.file_filter_text],
            #             ),
            #             "type": "cross_fields",
            #             "fields": [
            #                 "header_text^1.5",
            #                 "title_text^2",
            #                 "match_text",
            #             ],
            #         },
            #     },
            # )
            file_level_combined_words = " ".join(
                self.file_level_query + [self.file_filter_text],
            )
            if phrase_search_rule.match(file_level_combined_words):
                # must query
                query["query"]["bool"]["must"].append(
                    {
                        "simple_query_string": {
                            "query": file_level_combined_words,
                            "fields": ["match_text"],
                        },
                    },
                )
            else:
                file_level_final_query = {
                    "bool": {
                        # uni-gram + bi-gram + tri-gram
                        "should": [
                            {
                                "bool": {
                                    # n-gram_0 + n-gram_1 ... + n-gram_n
                                    "should": [
                                        {
                                            "multi_match": {
                                                "query": " ".join(gram),
                                                "type": "phrase",
                                                "fields": [
                                                    f"match_text^{round((1/e)**i*2,2)}",
                                                ],
                                                "slop": "0",  # do not allow extra token in n-gram
                                            },
                                        }
                                        for gram in ngrams(
                                            file_level_combined_words.split(),
                                            i + 1,
                                        )
                                    ],
                                },
                            }
                            for i in range(
                                min(3, len(file_level_combined_words.split())),
                            )
                        ],
                    },
                }
                query["query"]["bool"]["must"].append(
                    {
                        "dis_max": {
                            "queries": file_level_final_query,
                            "tie_breaker": 0.5,
                        },
                    },
                )
        if file_dynamic_boosters and file_level_combined_words:
            query["query"]["bool"]["should"] = [
                {
                    "multi_match": {
                        "query": file_level_combined_words,
                        "type": "cross_fields",
                        "fields": [
                            f"{k}^{v}" for k, v in file_dynamic_boosters.items()
                        ],
                    },
                },
            ]
        return query
