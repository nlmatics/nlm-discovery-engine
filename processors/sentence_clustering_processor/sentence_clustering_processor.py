import json
import os
import re
from collections import Counter
from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
import requests
from nlm_utils.model_client.classification import ClassificationClient
from nlm_utils.model_client.encoder import EncoderClient
from nlm_utils.utils import preprocessor
from nlm_utils.utils import STOPWORDS
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import spectral_embedding
from sklearn.metrics import pairwise_distances

from de_utils import Rake
from discovery_engine.objects import GroupTaskData
from processors.base_processor import BaseProcessor


class SpectralClustering:
    """
    See https://github.com/scikit-learn/scikit-learn/blob/fd237278e/sklearn/cluster/_spectral.py#L277
    """

    def __init__(
        self,
        n_clusters=4,
        eigen_solver=None,
        n_components=None,
        n_init=5,
        gamma=1.0,
        affinity="rbf",
        n_neighbors=5,
        eigen_tol=0.0,
        assign_labels="kmeans",
        degree=3,
        coef0=1,
        kernel_params=None,
        n_jobs=1,
    ):
        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.n_components = n_components
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol
        self.assign_labels = assign_labels
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self.affinity_matrix_ = X
        self.labels_ = spectral_clustering(
            self.affinity_matrix_,
            n_clusters=self.n_clusters,
            n_components=self.n_components,
            eigen_solver=self.eigen_solver,
            n_init=self.n_init,
            eigen_tol=self.eigen_tol,
            assign_labels=self.assign_labels,
            n_jobs=self.n_jobs,
        )
        return self


def spectral_clustering(
    affinity,
    n_clusters=8,
    n_components=None,
    eigen_solver=None,
    n_init=5,
    eigen_tol=0.0,
    assign_labels="kmeans",
    n_jobs=1,
):
    n_components = n_clusters if n_components is None else n_components
    # The first eigen vector is constant only for fully connected graphs
    # and should be kept for spectral clustering (drop_first = False)
    # See spectral_embedding documentation.
    maps = spectral_embedding(
        affinity,
        n_components=n_components,
        eigen_solver=eigen_solver,
        eigen_tol=eigen_tol,
        drop_first=False,
        random_state=1337,
    )
    # This is edge case where number of samples is small
    # if n_components > len(maps):
    #     n_components = len(maps)
    labels = (
        MiniBatchKMeans(n_components, n_init=n_init, random_state=1337)
        .fit(maps)
        .labels_
    )
    return labels


class SentenceClusteringProcessor(BaseProcessor):
    processor_type = "agg_post_processor"

    def __init__(self, settings: dict = {}, **kwargs):
        super().__init__(settings)
        self.topn = kwargs.get("topn", 3)
        self.extract_method = kwargs.get("extract_method", "get_combination")
        self.client = EncoderClient(
            model=self.settings["ENCODERS"][0],
            url=self.settings["MODEL_SERVER_URL"],
        )
        self.model = ClusterSentences(
            self.settings,
            self.client,
            self.extract_method,
            **kwargs,
        )

    def run(self, group_task: GroupTaskData, *args, **kwargs) -> Dict:
        # Build a sorted dictionary of files with order implied relevancy scores
        idf_stats = group_task.group_stats
        (
            fname_to_text_scores,
            fname_to_ans_scores,
            fname_to_texts,
            fname_to_answers,
        ) = self.extract_data(group_task)
        # calculate the relevency scores for each document using kwargs.topn returned sentence scores
        rel_match_scores = {f: sum(s) for f, s in fname_to_text_scores.items()}
        rel_ans_scores = {f: sum(s) for f, s in fname_to_ans_scores.items()}
        # cluster only for the top 50 documents
        top_match_fnames = sorted(
            (fname for fname in rel_match_scores.keys()),
            key=lambda x: rel_match_scores[x],
        )[-20:]
        top_ans_fnames = sorted(
            (fname for fname in rel_ans_scores.keys()),
            key=lambda x: rel_ans_scores[x],
        )[-20:]

        fname_to_rel_sent = {}
        if fname_to_answers:
            rel_scores = rel_ans_scores
            for fname, sentences in fname_to_answers.items():
                if fname in top_ans_fnames:
                    fname_to_rel_sent[fname] = sentences
        else:
            rel_scores = rel_match_scores
            for fname, sentences in fname_to_texts.items():
                if fname in top_match_fnames:
                    fname_to_rel_sent[fname] = fname_to_texts[fname]

            if group_task.question:
                client = ClassificationClient(
                    model=self.settings["QA_MODEL"],
                    task=self.settings["QA_TASK"],
                    url=self.settings["MODEL_SERVER_URL"],
                )
                questions = []
                sentences = []
                for sents in fname_to_rel_sent.values():
                    # break if match has been filterred out by QNLI
                    for sent in sents:
                        sentences.append(sent)
                        questions.append(group_task.question)

                answers = {}
                for i in range(10):
                    try:
                        answers = client(questions, sentences)["answers"][0]
                        break
                    except Exception as error:
                        self.logger.error(
                            f"Error {error}, retrying {i+1}",
                            exc_info=True,
                        )
                        if i == 9:
                            raise RuntimeError("SQUAD model error")
                        continue
                answers = list(answers.values())

                offset = 0
                for fname in fname_to_rel_sent:
                    _answers = answers[offset : offset + len(fname_to_rel_sent[fname])]
                    offset += len(fname_to_rel_sent[fname])
                    fname_to_rel_sent[fname] = [
                        answer["text"]
                        for answer in _answers
                        if answer["probability"] > 0.5
                    ]

        keywords = group_task.keywords
        is_keyword_search = group_task.is_keyword_search
        result = {}
        if is_keyword_search:
            result = self.model.fit(
                fname_to_rel_sent,
                keywords=keywords,
                scores=rel_scores,
                idf_stats=idf_stats,
            )
        elif fname_to_answers:
            result = self.model.fit(
                fname_to_rel_sent,
                keywords=keywords,
                scores=rel_scores,
                idf_stats=idf_stats,
            )
        elif fname_to_texts:
            templates = group_task.template
            result = self.model.fit(
                fname_to_rel_sent,
                keywords=templates,
                scores=rel_scores,
                idf_stats=idf_stats,
            )
        if result == {}:
            result = [
                {
                    "cluster_phrase": "root",
                    "cluster_top_words": "root",
                    "num_cluster_members": 0,
                    "children": [],
                },
            ]
            return result
        return json.loads(result)[0]

    def extract_data(self, group_task: GroupTaskData):
        fname_to_text_scores = defaultdict(list)
        fname_to_ans_scores = defaultdict(list)
        fname_to_texts = defaultdict(list)
        fname_to_answers = defaultdict(list)
        for group_match in group_task.group_matches:
            for match in group_match:
                score = max(
                    float(match.raw_scores.get("question_score_raw", 0.0)),
                    float(match.raw_scores.get("match_score_raw", 0.0)),
                )
                if score:
                    fname = match.context.filename
                    fname_to_text_scores[fname].append(score)
                    fname_to_texts[fname].append(match.context.text)
                    answer = match.answer
                    if answer:
                        fname_to_ans_scores[fname].append(
                            float(match.raw_scores.get("scaled_score_raw", 0.0)),
                        )
                        fname_to_answers[fname].append(answer)
        for fname_dict in [
            fname_to_text_scores,
            fname_to_ans_scores,
            fname_to_texts,
            fname_to_answers,
        ]:
            for k, v in fname_dict.items():
                fname_dict[k] = v[: self.topn]
        return (
            fname_to_text_scores,
            fname_to_ans_scores,
            fname_to_texts,
            fname_to_answers,
        )


class ClusterSentences:
    """Perform Clustering for topic and subtopic on a json containing file name and list of sentences as key-value pair.
    To cluster for topics on the data we do the following:
    1. Extract phrases from sentences
    2. Convert those phrases to embeddings
    3. Run Spectral Clustering Algorithm on the phrases
    4. Run kmeans on each cluster to get subcluster
    5. Take top 3 words for the sub cluster naming and the closest phrase to sub cluster mean embedding
    6. Join the words and phrases from subcluster to get a cluster level representation

    Parameters
    ----------
    extract_method : ["local", "rake", "kpminer"]
        If set to True, clustering will be on phrase level else on sentence level.
    encoder_name : ["sif", "distilbert"]
    num_clusters : int, optional, default = None - uses heuristics
        The number of topics to be extracted by spectral clustering or the dimension of the projection subspace in Spectral Clustering.
    n_jobs : int, optional, default = None implies 1 jobs
        Used for paralellizing the kmeans and paiwise distances.
    Heuristics Parameter:
    ---------------------
    cluster_size : int, optional, default = 400
        minumber of phrases to be present in each cluster. Used to divide total number of phrases to get an approximate number of clusters for spectral clustering.
    subcluster_size : int, optional, default = 100
        Number of phrases to be used for subclustering for kmeans.
    min_num_clusters : int, optional, default = 2
        Minimum number of clusters - spectral clustering.
    max_num_clusters : int, optional, default = 15
        Maximum number of clusters - spectral clustering.
    min_num_subclusters : int, optional, default = 2
        Minimum number of subclusters - kmeans
    max_num_subclusters : int, optional, default = 5
        Maximum number of subclusters - kmeans
    """

    def __init__(
        self,
        settings,
        client,
        extract_method,
        topn=3,
        num_clusters=None,
        cluster_size=10,
        subcluster_size=5,
        min_num_clusters=3,
        max_num_clusters=20,
        min_num_subclusters=2,
        max_num_subclusters=3,
        n_jobs=1,
        **kwargs,
    ):
        self.settings = settings
        self.client = client
        self.extract_method = extract_method
        self.topn = topn
        self.num_clusters = num_clusters
        self.cluster_size = cluster_size
        self.subcluster_size = subcluster_size
        self.min_num_clusters = min_num_clusters
        self.max_num_clusters = max_num_clusters
        self.min_num_subclusters = min_num_subclusters
        self.max_num_subclusters = max_num_subclusters
        self.n_jobs = n_jobs

    def _ext_phrases(self, sentences, keywords, method="rake"):
        url = os.environ.get("MODEL_SERVER_URL", self.settings["MODEL_SERVER_URL"])

        if method == "local":
            url = f"{url}/phraser"
            req_data = {
                "text": sentences,
                "keywords": keywords,
                "option": "get_filtered_phrases",
            }
            resp = requests.post(url, json=req_data)
            if resp.ok:
                return dict(resp.json())["data"]
            else:
                raise Exception("exception: " + str(resp))
        elif method == "rake":
            r = Rake(stopwords=STOPWORDS)
            return r.extract_keywords(sentences)

        elif method == "get_combination":
            url = f"{url}/phraser"
            req_data = {
                "text": sentences,
                "keywords": keywords,
                "option": "get_combination",
            }
            resp = requests.post(url, json=req_data)
            if resp.ok:
                return dict(resp.json())["data"]
            else:
                raise Exception("exception: " + str(resp))

        else:
            phrases = []
            for sent in sentences:
                phrases.append([" ".join(re.sub("[^a-zA-Z0-9]", " ", sent).split())])
            return phrases

    def _file_phrase_extractor(self, data, keywords):
        """
        Extract phrases and sentences from a Dict {filename: list of sentences}
        Parameters
        ----------
        data : Dict[str, List[str]]
            dictionary should have file name as key and relevant list of sentences from the file as values.
        keywords: list of strings
            keywords is either ['question'] or ['template_1', 'template_2', ... ]
        Returns
        -------
        df_file_phrases : pandas DataFrame
            DF with 3 columns - filename, sentences, phrases from the sentences
        """
        filenames = [fname for fname in data.keys()]
        sentences = [s for s in data.values()]

        # Create a DataFrame to store hierarchy of relations
        df_file_sent = pd.DataFrame.from_dict(
            {"filename": filenames, "sentence": sentences},
        )

        sentence_series = (
            df_file_sent.apply(lambda x: pd.Series(x["sentence"]), axis=1)
            .stack()
            .reset_index(level=1, drop=True)
        )
        sentence_series.name = "sentence"
        df_file_sent = (
            df_file_sent.drop("sentence", axis=1)
            .join(sentence_series)
            .reset_index(drop=True)
        )
        # Extract phrases from sentences
        phrases = self._ext_phrases(
            df_file_sent["sentence"].fillna("").values.tolist(),
            keywords,
            method=self.extract_method,
        )
        query_keywords = []
        for keyword in preprocessor(" ".join(keywords)):
            query_keywords.append(keyword)

        def is_reasonable_phrase(phrase):
            return True
            # # Heuristics to filter out noisy phrases
            # if not phrase:
            #     return False
            # phrase_words = phrase.split()
            # max_min_str = 0 < len(max(phrase_words, key=len)) < 10
            # max_min_len = 0 < len(phrase_words) < 5
            # is_alpha = sum(w.isalpha() for w in phrase_words) == len(phrase_words)
            # is_stopword = (phrase_words[0] in STOPWORDS) or (
            #     phrase_words[-1] in STOPWORDS
            # )
            # # Count word overlap
            # repeat = 0
            # for word in preprocessor(phrase):
            #     if word in query_keywords:
            #         repeat += 1
            # is_query_word = len(phrase_words) - repeat < len(phrase_words) / 2
            # if (
            #     (not max_min_str)
            #     or (not max_min_len)
            #     or (not is_alpha)
            #     or is_query_word
            #     or is_stopword
            # ):
            #     return False
            # else:
            #     return True

        phrases = [
            [p if is_reasonable_phrase(p) else "" for p in phrase_list]
            if phrase_list
            else [""]
            for phrase_list in phrases
        ]

        df_file_sent["phrase"] = phrases
        phrase_series = (
            df_file_sent.apply(lambda x: pd.Series(x["phrase"]), axis=1)
            .stack()
            .reset_index(level=1, drop=True)
        )
        phrase_series.name = "phrase"
        df_file_sent = (
            df_file_sent.drop("phrase", axis=1)
            .join(phrase_series)
            .reset_index(drop=True)
        )
        # Encode phrases for clustering
        df_file_sent["embeddings"] = self._encode_phrases(
            [phrase for phrase_list in phrases for phrase in phrase_list],
        ).tolist()
        return df_file_sent

    def _encode_phrases(self, phrase_list):
        """
        Encode sentences/Phrases to a embedding.
        Parameters
        ----------
        phrase_list : array like
            List/Array of all the phrases to be encoded
        Returns
        -------
        embeddings : numpy array - shape - d_emb x num_phrase
            Array of embeddings of the phrases
        """
        return self.client(phrase_list)["embeddings"]

    def _cluster_phrases(self, embeddings_mat):
        """
        Spectral Clustering on embeddings.
        Parameters
        ----------
        embeddings_mat : np.ndarray - d_emb x num_phrase
            List/Array of embeddings of all the phrases to be clustered.
        Returns
        -------
        labels : list
            List of cluster label for each phrase
        """
        embeddings_mat = np.array(embeddings_mat, dtype=np.float32)
        distance = pairwise_distances(
            embeddings_mat,
            metric="cosine",
            n_jobs=self.n_jobs,
        )
        affinity = np.exp(-(distance ** 2))
        clusterer = SpectralClustering(
            n_clusters=self.num_clusters,
            affinity="precomputed",
            n_jobs=self.n_jobs,
        )
        clusterer.fit(affinity)
        return clusterer.labels_

    def _kmeans(self, embeddings_mat, num_subcluster):
        """
        KMeans clustering on embeddings.
        Parameters
        ----------
        embeddings_mat : array like - shape - d_emb x num_phrase
            List/Array of embeddings of all the phrases to be clustered.
        num_subcluster : int
            Number of clusters required for KMeans
        Returns
        -------
        labels : list - len - num_phrase
            List of cluster label for each phrase
        """
        embeddings_mat = np.array(embeddings_mat, dtype=np.float32)
        clusterer = MiniBatchKMeans(n_clusters=num_subcluster, random_state=1337)
        clusterer.fit(embeddings_mat)
        return clusterer.labels_

    def _title_cluster(self, cluster_df):
        """
        Get closest phrase to the cluster center.
        Parameters
        ----------
        cluster_df : dataframe
            Dataframe of the cluster with embeddings in a column 'embeddings' and phrase in a column 'Phrase'.
        Returns
        -------
        phrase : The phrase from the dataframe which is closest in the embedding space to the cluster mean.
        """
        cluster_embed = np.array(
            cluster_df["embeddings"].values.tolist(),
            dtype=np.float32,
        )
        cluster_mean = cluster_embed.mean(0).reshape(-1, 1)
        cluster_mean /= np.linalg.norm(cluster_mean)
        cos_dist_phrase = (
            np.dot(
                cluster_embed,
                cluster_mean.reshape(-1, 1),
            )
            / np.linalg.norm(cluster_embed, axis=1).reshape(-1, 1)
        )
        cos_dist_phrase = cos_dist_phrase.squeeze()
        top_phrase_index = np.argmax(cos_dist_phrase)
        return cluster_df.iloc[top_phrase_index, :]["phrase"]

    def _cluster_top_words(
        self,
        cluster_df,
        keywords,
        seen_words,
        idf_stats,
        num_top_words=3,
    ):
        """
        Get top words denoting the cluster. Currently uses frequency of words to get most frequent n words.
        Parameters
        ----------
        cluster_df : dataframe
            Dataframe of the cluster with phrase in a column 'Phrase'.
        num_top_words : int
            Number of words needed to represent cluster.
        Returns
        -------
        word_list : string
            num_top_words separated by comma as string.
        """
        phrase_list = cluster_df["phrase"].tolist()
        phrase_list = [phrase for phrase in phrase_list if phrase]
        if not phrase_list:
            return ""
        stem_to_unstem = {}
        query_keywords = []
        # We want to avoid adding words from query to top related words
        for keyword in preprocessor(" ".join(keywords)):
            if keyword:
                query_keywords.append(keyword)

        def is_reasonable_word(word):
            # Heuristics to filter out noise
            max_min_len = 2 < len(word) < 10
            is_alpha = word.isalpha()
            is_seen = word in seen_words
            is_query_word = word in query_keywords
            is_stopword = word in STOPWORDS
            if (
                (not max_min_len)
                or (not is_alpha)
                or is_seen
                or is_query_word
                or is_stopword
            ):
                return False
            else:
                return True

        # Every phrase is split into words and count
        word_counter = Counter()
        for phrase in phrase_list:
            phrase = str(phrase)
            for word in phrase.split():
                # Preprocessor returns empty list for filtered tokens like "and"
                stemed_list = preprocessor(word)
                if not stemed_list:
                    continue
                stemed_word = stemed_list[0]
                if is_reasonable_word(word):
                    stem_to_unstem[stemed_word] = word
                    word_counter[stemed_word] += 1

        top_stemed_words = [
            word for word, count in word_counter.most_common(num_top_words)
        ]
        # top_words = [
        #     stem_to_unstem[word]
        #     for word in top_stemed_words
        #     if idf_stats.get(word, 0.0) > 40.0
        # ]

        # @Yi THIS IS THE PARAMETER WHICH YOU SHOULD CHANGE TO IMPROVE CLUSTERING
        top_words = [
            stem_to_unstem[word]
            for word in top_stemed_words
            if idf_stats.get(word, 0.0) > 10.0
        ]
        top_words = sorted(top_words, key=lambda x: idf_stats.get(x, 0.0))[-3:]
        for word in top_stemed_words:
            seen_words.add(word)
        if not top_words:
            top_words = ["Unclustered"]
        return ", ".join(top_words)

    def _merge_dicts(self, record):
        """
        Function to be used with dataframe rows. Merges multiple dictionaries into one.
        Parameters
        ----------
        record: dataframe row
            The DataFrame rows containing dictionaries which are to be merged.
        Returns
        -------
        dict: dictionary
            A combined dictionary for all the rows.
        """
        return {
            key: value
            for dict_obj in record.dropna()
            for key, value in dict_obj.items()
        }

    def fit(self, data, keywords, scores, idf_stats):
        """
        Extract phrases and sentences
        Cluster the phrases and return a json file with cluster and sub-cluster level phrases and words.
        Parameters
        ----------
        data: Dict[str, List[str]]
        Returns
        -------
        json_file: json string
            File contains cluster level phrase, top words, number of sentences and number of subclusters. At subcluster level it
            contains subcluster level phrase, top words, number of sentences and key value ppair of file name and sentences.
        """
        seen_words = set(keywords)

        df_phrases = self._file_phrase_extractor(data, keywords)

        if self.num_clusters is None:
            num_phrase = len(df_phrases)
            self.num_clusters = min(
                max(int(num_phrase / len(data)), self.min_num_clusters),
                self.max_num_clusters,
            )
        df_phrases["cluster_label"] = self._cluster_phrases(
            df_phrases["embeddings"].values.tolist(),
        )
        df_phrases["cluster_phrase"] = "Unclustered"
        df_phrases["cluster_top_words"] = "Unclustered"
        for i in df_phrases["cluster_label"].unique():
            df_phrases_cluster = df_phrases["cluster_label"] == i
            num_subcluster = max(
                min(
                    int(len(df_phrases[df_phrases_cluster]) / self.subcluster_size),
                    self.max_num_subclusters,
                ),
                self.min_num_subclusters,
            )
            # Adding a check for corner case i.e., if cluster has only 1 element and minimum subcluster is 2 or similar situation
            # We will just create one cluster in that case
            if num_subcluster * self.subcluster_size > len(
                df_phrases[df_phrases_cluster],
            ):
                num_subcluster = 1
                df_phrases.loc[df_phrases_cluster, "subcluster_labels"] = 0
            else:
                df_phrases.loc[df_phrases_cluster, "subcluster_labels"] = self._kmeans(
                    df_phrases.loc[df_phrases_cluster, "embeddings"].values.tolist(),
                    num_subcluster,
                )
            # Adding a check for corner case, if a subcluster is not formed then we don't need to run the naming algorithm
            # else it will throw an error
            subcluster_range = df_phrases.loc[
                df_phrases_cluster,
                "subcluster_labels",
            ].unique()
            for j in subcluster_range:
                df_phrases_subcluster = df_phrases_cluster & (
                    df_phrases["subcluster_labels"] == j
                )
                # If number of elements in subclaster is small we disolve the subcluster
                df_phrases.loc[
                    df_phrases_subcluster,
                    "subcluster_phrase",
                ] = self._title_cluster(df_phrases[df_phrases_subcluster])
                top_words = self._cluster_top_words(
                    df_phrases[df_phrases_subcluster],
                    keywords,
                    seen_words,
                    idf_stats,
                )
                df_phrases.loc[
                    df_phrases_subcluster,
                    "subcluster_top_words",
                ] = top_words
            df_phrases.loc[df_phrases_cluster, "cluster_phrase"] = ", ".join(
                [
                    word
                    for word in df_phrases.loc[
                        df_phrases_cluster,
                        "subcluster_phrase",
                    ].unique()
                    if word and isinstance(word, str)
                ],
            )
            df_phrases.loc[df_phrases_cluster, "cluster_top_words"] = ", ".join(
                list(
                    set(
                        (
                            ", ".join(
                                [
                                    word
                                    for word in df_phrases.loc[
                                        df_phrases_cluster,
                                        "subcluster_top_words",
                                    ].unique()
                                    if word and isinstance(word, str)
                                ],
                            )
                        ).split(", "),
                    ),
                ),
            )
            df_phrases.loc[df_phrases_cluster, "num_subcluster"] = num_subcluster
        df_phrases = df_phrases[
            [
                "cluster_phrase",
                "cluster_top_words",
                "subcluster_top_words",
                "subcluster_phrase",
                "filename",
                "sentence",
                "num_subcluster",
            ]
        ].drop_duplicates()
        df_phrases = (
            df_phrases.groupby(
                [
                    "cluster_phrase",
                    "cluster_top_words",
                    "subcluster_top_words",
                    "subcluster_phrase",
                    "filename",
                ],
            )
            .agg({"sentence": list, "num_subcluster": "mean"})
            .reset_index()
        )
        # Scores determine the relevance of overall document
        df_phrases["scores"] = df_phrases.apply(
            lambda x: scores.get(x["filename"], 0),
            axis=1,
        )
        df_phrases = df_phrases.sort_values(by=["scores"], ascending=False)
        # Sort such that all displayed subcluster_members information is sorted by relevance

        df_phrases["subcluster_members"] = df_phrases.apply(
            lambda x: {x["filename"]: x["sentence"]},
            axis=1,
        )
        df_phrases["num_subcluster_members"] = df_phrases["sentence"].str.len()

        df_phrases = (
            df_phrases[
                [
                    "cluster_phrase",
                    "cluster_top_words",
                    "subcluster_top_words",
                    "subcluster_phrase",
                    "subcluster_members",
                    "num_subcluster_members",
                    "num_subcluster",
                ]
            ]
            .groupby(
                [
                    "cluster_phrase",
                    "cluster_top_words",
                    "subcluster_top_words",
                    "subcluster_phrase",
                ],
            )
            .agg(
                {
                    "subcluster_members": self._merge_dicts,
                    "num_subcluster_members": "sum",
                    "num_subcluster": "mean",
                },
            )
            .reset_index()
        )
        clustered_phrases = df_phrases[
            ~df_phrases["cluster_phrase"].str.contains("Unclustered")
            & ~df_phrases["cluster_top_words"].str.contains("Unclustered")
            & (df_phrases["subcluster_top_words"] != "")
            & (df_phrases["subcluster_phrase"] != "")
        ]
        # Edge case for when all entries are "Unclustered" and filtered out
        if clustered_phrases.index.tolist() == []:
            return {}
        df_phrases = clustered_phrases
        df_phrases["children"] = df_phrases.apply(
            lambda x: {
                "cluster_top_words": x["subcluster_top_words"]
                if x["subcluster_top_words"]
                else max(x["subcluster_top_words"].split(", "), key=len) or "Other",
                "cluster_phrase": x["subcluster_phrase"]
                if x["subcluster_phrase"]
                else max(x["subcluster_top_words"].split(", "), key=len) or "Other",
                "num_cluster_members": x["num_subcluster_members"],
                "details": x["subcluster_members"],
            },
            axis=1,
        )
        df_phrases = (
            df_phrases[
                ["cluster_phrase", "cluster_top_words", "num_subcluster", "children"]
            ]
            .groupby(["cluster_phrase", "cluster_top_words"])
            .agg({"num_subcluster": "mean", "children": list})
            .reset_index()
            .rename(columns={"num_subcluster": "num_cluster_members"})
        )
        df_phrases["children_root_level"] = df_phrases.apply(
            lambda x: {
                "cluster_top_words": x["cluster_top_words"],
                "cluster_phrase": x["cluster_phrase"]
                if x["cluster_phrase"]
                else max(x["cluster_top_words"].split(", "), key=len),
                "num_cluster_members": x["num_cluster_members"],
                "children": x["children"],
            },
            axis=1,
        )
        df_phrases["cluster_phrase"] = "root"
        df_phrases["cluster_top_words"] = "root"
        df_phrases["num_members"] = 1
        df_phrases = (
            df_phrases[
                [
                    "cluster_top_words",
                    "cluster_phrase",
                    "children_root_level",
                    "num_members",
                ]
            ]
            .groupby(["cluster_phrase", "cluster_top_words"])
            .agg({"num_members": "sum", "children_root_level": list})
            .reset_index()
            .rename(
                columns={
                    "children_root_level": "children",
                    "num_members": "num_cluster_members",
                },
            )
        )
        return df_phrases.to_json(orient="records")
