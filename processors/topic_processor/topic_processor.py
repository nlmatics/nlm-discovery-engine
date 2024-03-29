import re
from collections import Counter
from collections import defaultdict
from typing import Dict

import numpy as np
from nlm_utils.model_client.encoder import EncoderClient
from sklearn import metrics
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

from de_utils import Rake
from de_utils.rake import STOPWORDS
from discovery_engine.objects import TaskData
from processors.base_processor import BaseProcessor


class TopicProcessor(BaseProcessor):
    processor_type = "agg_post_processor"

    def __init__(self, settings: dict = {}, **kwargs):
        super().__init__(settings)
        self.topn = kwargs.get("topn", 3)
        self.extract_method = kwargs.get("extract_method", "get_combination")
        self.phrase_extractor = Rake(STOPWORDS)
        self.documents = kwargs.get("documents")
        self.sent_encoder = EncoderClient(
            model="sif",
            url=self.settings["MODEL_SERVER_URL"],
        )

    def run(self, task_data: TaskData, *args, **kwargs) -> Dict:
        # Build a sorted dictionary of files with order implied relevancy scores
        json_result, cluster_map, phrases, encs_tsne, encs_pca = self._run_internal(
            task_data,
        )
        task_data.aggs["TopicProcessor"] = json_result
        return json_result

    def _run_internal(self, task_data: TaskData, *args, **kwargs) -> Dict:
        # Build a sorted dictionary of files with order implied relevancy scores
        (texts, text_contexts, answers, answer_contexts) = self.extract_data(task_data)
        has_answers = False
        if len(answers) > 0:
            phrases = answers
            contexts = answer_contexts
            has_answers = True

        else:
            phrases = texts
            contexts = text_contexts

        list_of_phrase_lists = self._get_list_of_phrase_lists(phrases, has_answers)
        phrases = [phrase for phrases in list_of_phrase_lists for phrase in phrases]
        contexts = [
            contexts[i]
            for i, phrase_lists in enumerate(list_of_phrase_lists)
            for _ in range(len(phrase_lists))
        ]
        if len(contexts) > 3:
            clusters, encs_tsne, encs_pca = self._cluster(phrases)
        else:
            # show cluster no matter what << change this
            phrases = []
            contexts = []
            clusters = []
        json_result, cluster_map = self._get_json(phrases, contexts, clusters)
        return json_result, cluster_map, phrases, encs_tsne, encs_pca

    def _pairwise_distance(self, phrases):
        cleaned_phrases = []
        for phrase in phrases:
            # cleaned_phrases.append(" ".join(preprocessor(phrase)))
            cleaned_phrases.append(phrase)

        encs = self.sent_encoder(cleaned_phrases)["embeddings"]
        sims = metrics.pairwise.cosine_similarity(encs)
        dists = np.maximum(1 - sims, 0.0)
        return encs, sims, dists

    def _get_list_of_phrase_lists(self, phrases, has_answers):
        list_of_phrase_lists = self.phrase_extractor.extract_keywords(phrases)
        # remove phrases of length 1
        list_of_phrase_lists = [
            [phrase for phrase in phrases if len(phrase) > 1]
            for phrases in list_of_phrase_lists
        ]
        if has_answers:
            # split long phrases by commas and "and" if phrases are answers
            for idx, phrase in enumerate(phrases):
                phrase_list = []
                if len(phrase.split()) <= 6:
                    phrase_list.append(phrase)
                phrase_sents = phrase.split(", ")
                for phrases in phrase_sents:
                    phrases = re.split(r"\s+(?:but|and)\s+", phrases)
                    for phrase in phrases:
                        # phrase must not be too long or too short
                        if (len(phrase.split()) <= 6) and (len(phrase) > 2):
                            phrase_list.append(phrase)
                list_of_phrase_lists[idx] += phrase_list
        return list_of_phrase_lists

    def _cluster(self, phrases):
        encs, sims, dists = self._pairwise_distance(phrases)

        n_items = len(encs)
        max_components = 50
        n_pca_components = min(max(n_items - 1, 1), max_components)

        if n_pca_components > 3:
            pca = PCA(
                n_components=n_pca_components,
                svd_solver="arpack",
                random_state=0,
            )
            encs_pca = pca.fit_transform(encs)
            tsne = TSNE(n_components=2, perplexity=5, n_iter=1200, random_state=0)
            encs_proj = tsne.fit_transform(encs_pca)
        else:
            pca = PCA(n_components=2, svd_solver="arpack")
            encs_pca = pca.fit_transform(encs)
            encs_proj = encs_pca

        scaler = MinMaxScaler(feature_range=(0, 1))
        encs_proj = scaler.fit_transform(encs_proj)

        # plot_output(encs_tsne, answers)
        alpha = 1.25
        distance_threshold = alpha / np.power(n_items * 2, 1 / 2)
        clusters = Birch(n_clusters=None, threshold=distance_threshold).fit_predict(
            encs_proj,
        )

        # encs_tsne = encs_pca
        # clusters = KMeans(n_clusters=5, random_state=0).fit_predict(encs_tsne)

        # encs_embedded = TSNE(n_components=2).fit_transform(encs)
        # tsne_dists =
        # clusters = DBSCAN(eps=0.01, min_samples=2, metric="cosine").fit_predict
        #     encs_tsne,
        # )
        # clusters = Birch(n_clusters=None).fit_predict(encs_tsne)
        # clusters = DBSCAN(eps=.2, min_samples=2).fit_predict(encs_pca)
        # clusters = DBSCAN(eps=.2, min_samples=2, metric='precomputed').fit_predict(dists)
        # clusters = SpectralClustering(affinity='precomputed').fit_predict(dists)
        # clusters = SpectralClustering().fit_predict(encs_tsne)
        # clusters = AgglomerativeClustering().fit_predict(encs_tsne)

        return clusters, encs_proj, encs_pca

    def _get_json(self, phrases, contexts, clusters):
        cluster_map = {}
        phrase_map = {}
        for idx, cluster in enumerate(clusters):
            if cluster not in cluster_map:
                cluster_map[cluster] = []
                phrase_map[cluster] = []
            # contexts[idx]["answer"] = phrases[idx]
            cluster_map[cluster].append(contexts[idx])
            phrase_map[cluster].append(phrases[idx])

        result = {
            "cluster_phrase": "root",
            "cluster_top_words": "root",
            "num_cluster_members": len(cluster_map),
            "children": [],
        }
        # {phrase : Counter(cluster_id : phrase_count)}
        phrase_2_counter = defaultdict(lambda: Counter())
        cluster_id_2_n_phrases = Counter()
        for cluster_id, cluster_ctx in cluster_map.items():
            for phrase in phrase_map[cluster_id]:
                # text normalization
                phrase = re.sub("[^A-Za-z0-9 ]", "", phrase.lower())
                phrase_2_counter[phrase][cluster_id] += 1
                cluster_id_2_n_phrases[cluster_id] += 1

        repeated_phrases = set()
        for cluster_id, cluster_ctx in cluster_map.items():
            cluster_phrase = ""
            for phrase in phrase_map[cluster_id]:
                phrase = re.sub("[^A-Za-z0-9 ]", "", phrase.lower())
                if phrase not in repeated_phrases:
                    cluster_phrase = phrase
                    repeated_phrases.add(phrase)
                    break
            else:
                phrase = phrase_map[cluster_id][0]
                phrase = re.sub("[^A-Za-z0-9 ]", "", phrase.lower())

                merge_cluster_candidates = [
                    (cnt / (cluster_id_2_n_phrases[idx] + 1), idx)
                    for idx, cnt in phrase_2_counter[phrase].most_common(3)
                    if idx != cluster_id
                ]
                if merge_cluster_candidates != []:
                    merge_cluster_id = max(merge_cluster_candidates)[1]
                    merge_cluster = cluster_map[merge_cluster_id]
                    merge_cluster += cluster_ctx

        repeated_phrases = set()
        for cluster_id, cluster_ctx in cluster_map.items():
            cluster_phrase = ""
            for phrase in phrase_map[cluster_id]:
                phrase = re.sub("[^A-Za-z0-9 ]", "", phrase.lower())
                if phrase not in repeated_phrases:
                    cluster_phrase = phrase
                    repeated_phrases.add(phrase)
                    break
            else:
                # it has already been merged into another cluster in the previous loop
                continue

            child = {
                "cluster_phrase": cluster_phrase,
                "cluster_top_words": ", ".join(phrase_map[cluster_id]),
                "num_cluster_members": len(cluster_ctx),
                "details": {},
            }
            answers = []
            for ctx in cluster_ctx:
                for phrase in phrase_map[cluster_id]:
                    # this parameter will change filename -> file_id
                    child["details"][ctx["file_name"]] = [ctx]
                    if re.sub("[^A-Za-z0-9]", "", phrase):
                        answers.append(phrase)

            cluster_phrases = Counter(answers).most_common(5)
            unique_phrases = []
            has_seen_phrase = False
            for phrase, _ in cluster_phrases:
                for seen_phrase in unique_phrases:
                    if (phrase.lower() in seen_phrase.lower()) or (
                        seen_phrase.lower() in phrase.lower()
                    ):
                        has_seen_phrase = True
                        break
                if not has_seen_phrase:
                    unique_phrases.append(phrase)
            # unique_phrases = sorted(unique_phrases, key=len, reverse=True)
            child["cluster_phrases"] = ", ".join(unique_phrases[:3])
            child["cluster_label"] = unique_phrases[0]
            result["children"].append(child)

        return result, cluster_map

    def extract_data(self, task_data: TaskData):
        texts = []
        text_contexts = []
        answers = []
        answer_contexts = []
        for file_idx, matches in task_data.matches.items():
            # filter out missing files
            if file_idx not in self.documents:
                continue
            for match in matches[0:3]:  # replace with score
                texts.append(match.match_text)
                ctx = {
                    "text": match.match_text,
                    "answer": match.answer,
                    "file_idx": file_idx,
                    "file_name": self.documents[file_idx].file_name,
                }
                text_contexts.append(ctx)
                answer = match.answer
                if answer:
                    answers.append(answer)
                    answer_contexts.append(ctx)
        return (
            texts,
            text_contexts,
            answers,
            answer_contexts,
        )
