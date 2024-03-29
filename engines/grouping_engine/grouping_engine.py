from typing import Dict

import networkx as nx
import numpy as np
from nlm_utils.model_client import EncoderClient
from nltk.tokenize import RegexpTokenizer

from discovery_engine.objects import GroupTaskData
from discovery_engine.objects import TaskData
from engines.base_engine import BaseEngine


# don't merge similar answers on following entities
UNIQUE_ENTITIES = {
    # CARDINAL
    "NUM:count",
    "NUM:code",
    # QUANTITY
    "NUM:volsize",
    "NUM:dist",
    "NUM:temp",
    "NUM:weight",
    # MONEY
    "NUM:money",
    # PERCENT
    "NUM:perc",
    # MEASUREMENT
    "NUM:speed",
    # DATE TIME
    "NUM:date",
}


class GroupingEngine(BaseEngine):
    def __init__(self, settings: dict = {}):
        super().__init__(settings)

        self.client = EncoderClient(
            model="sif",
            url=self.settings["MODEL_SERVER_URL"],
            dummy_number=False,
        )
        # self.phrase_extractor = Rake(stopwords=STOPWORDS, min_length=3)
        self.tokenizer = RegexpTokenizer(r"\w+").tokenize

    def run(self, group_task: GroupTaskData, **kwargs) -> Dict:
        tasks = group_task.tasks

        topics = kwargs["override_topic"]

        for topic in topics:
            task = tasks[topic]
            self.merge_matches_from_multi_criteria(task)

        # calculate question score
        text2emb = {}
        for topic in topics:
            task = tasks[topic]
            for _, matches in task.matches.items():
                for match in matches:
                    # # skip groups when group_flag is disabled
                    if match.criteria.group_flag == "disable":
                        continue
                    if match.formatted_answer:
                        text2emb[match.formatted_answer] = None
                    elif match.answer:
                        text2emb[match.answer] = None
                    if match.qa_text:
                        text2emb[match.qa_text] = None

        self.logger.info(f"Query embeddings for {len(text2emb)} sentences")

        if not len(text2emb):
            return

        embs = self.client(text2emb)["embeddings"]

        for key, emb in zip(text2emb, embs):
            text2emb[key] = emb

        # loop over all matches to group by file_idx
        for topic in topics:
            self.logger.info(f"Processing topic '{topic}'")
            task = tasks[topic]

            # skip groups when enable_grouping is False
            for _, matches in task.matches.items():
                n_matches = len(matches)

                parent_G = nx.DiGraph()
                parent_G.add_nodes_from(range(n_matches))

                location_G = nx.DiGraph()
                location_G.add_nodes_from(range(n_matches))

                same_answer_G = nx.DiGraph()
                same_answer_G.add_nodes_from(range(n_matches))

                similarity_G = nx.DiGraph()
                similarity_G.add_nodes_from(range(n_matches))

                all_nodes_G = nx.DiGraph()
                all_nodes_G.add_nodes_from(range(n_matches))

                match_idx_to_node_map = {}

                for i in range(len(matches)):
                    match_i = matches[i]
                    match_idx_to_node_map[match_i.oid] = match_i

                    if match_i.criteria.group_flag == "disable":
                        continue

                    # table match should be single type
                    answer_i = match_i.formatted_answer or match_i.answer
                    for j in range(i + 1, len(matches)):
                        match_j = matches[j]

                        if match_j.criteria.group_flag == "disable":
                            continue

                        answer_j = match_j.formatted_answer or match_j.answer

                        # parent similarity
                        if match_i.parent_oid == match_j.parent_oid:
                            # Create Edges only for matches with answer in case of Question Answering.
                            if match_i.criteria.is_question:
                                if match_i.has_answer:
                                    parent_G.add_edge(i, j)
                            else:
                                # Else create an edge, there are no questions and we might be looking for header Summary
                                parent_G.add_edge(i, j)

                        # fuzzy similarity
                        # TODO: need to take care numbers

                        if match.criteria.is_bool_question:
                            pass

                        elif answer_i and answer_j:
                            if self.tokenizer(answer_i.lower()) == self.tokenizer(
                                answer_j.lower(),
                            ):
                                same_answer_G.add_edge(i, j)
                            else:
                                similarity_threshold = 0.85
                                # higher similarity threshold for table
                                if (
                                    match_i.group_type == "table"
                                    or match_j.group_type == "table"
                                ):
                                    similarity_threshold = 0.95
                                # check if they are similar answer
                                if (
                                    np.dot(
                                        text2emb[answer_i],
                                        text2emb[answer_j],
                                    )
                                    > similarity_threshold
                                ):
                                    similarity_G.add_edge(i, j)

                            # location similarity
                            # sentences are close and comes from same header
                            if (
                                # answer_i and answer_j and
                                abs(match_i.match_idx - match_j.match_idx) <= 3
                                and match_i.header_text == match_j.header_text
                                and match_i.block_type == match_j.block_type
                                and match_i.qa_text
                                and match_j.qa_text
                                and np.dot(
                                    text2emb[match_i.qa_text],
                                    text2emb[match_j.qa_text],
                                )
                                > 0.5
                            ):
                                location_G.add_edge(i, j)

                gid = 0
                # get groups generated by parents
                for nodes in nx.weakly_connected_components(parent_G):
                    if len(nodes) == 1:
                        continue

                    parent_match = match_idx_to_node_map.get(
                        matches[list(nodes)[0]].parent_oid,
                        None,
                    )
                    if parent_match:
                        group_type = parent_match.group_type
                    else:
                        parent_node_idx = min(
                            (n for n in nodes if parent_G.out_degree(n) > 0),
                            key=lambda node: matches[node].match_idx,
                        )
                        parent_node = matches[parent_node_idx]
                        group_type = parent_node.group_type

                    for node in nodes:
                        matches[node].group = gid
                        matches[node].group_type = group_type
                        location_G.remove_node(node)
                        same_answer_G.remove_node(node)
                        similarity_G.remove_node(node)
                        all_nodes_G.remove_node(node)
                    gid += 1

                # get groups generated by location
                for nodes in nx.weakly_connected_components(location_G):
                    if len(nodes) == 1:
                        continue
                    if any(
                        [
                            matches[node].block_type in {"header", "table"}
                            for node in nodes
                        ],
                    ):
                        continue

                    group_type = "same_location"
                    for node in nodes:
                        matches[node].group = gid
                        matches[node].group_type = group_type
                        same_answer_G.remove_node(node)
                        similarity_G.remove_node(node)
                        all_nodes_G.remove_node(node)
                    gid += 1

                # get groups generated by same answer
                is_boost_by_same_answer = False
                for nodes in nx.weakly_connected_components(same_answer_G):
                    if len(nodes) == 1:
                        continue
                    is_boost_by_same_answer = True
                    for node in nodes:
                        matches[node].group = gid
                        matches[node].group_type = "same_answer"
                        similarity_G.remove_node(node)
                        all_nodes_G.remove_node(node)
                    gid += 1

                # downweight other answers if needed
                if is_boost_by_same_answer:
                    for match in matches:
                        if match.group_type != "same_answer":
                            match.raw_scores["group_score"] = min(
                                match.raw_scores["group_score"],
                                0.9,
                            )

                # get groups generated by similar answer
                for nodes in nx.weakly_connected_components(similarity_G):
                    if len(nodes) == 1:
                        continue
                    for node in nodes:
                        matches[node].group = gid
                        matches[node].group_type = "similar_answer"
                        all_nodes_G.remove_node(node)
                    gid += 1

                # all other nodes without groups
                for node in all_nodes_G.nodes():
                    matches[node].group = gid
                    if matches[node].group_type != "table":
                        matches[node].group_type = "single"
                    gid += 1

    def merge_matches_from_multi_criteria(self, task: TaskData, **kwargs):
        # tmp container to deduplicate matches
        task_matches = {}

        # loop through all criteria
        prev_start_idx = 0
        for c_idx, criteria in enumerate(task.criterias):
            match_len = len(criteria.matches)
            start_idx = prev_start_idx * c_idx
            prev_start_idx = start_idx + match_len
            for file_idx, matches in criteria.matches.items():
                # init container by document
                if file_idx not in task_matches:
                    task_matches[file_idx] = {}

                # group is disable
                for idx, match in enumerate(matches):
                    match.criteria.group_flag = criteria.group_flag
                    # split groups when group_flag is disabled
                    if match.criteria.group_flag == "disable":
                        # assign unique group id to each match
                        # Don't want same idx from different criteria going to the same group and
                        # thereby not showing up as a probable candidate
                        match.group = start_idx + idx
                        # remove parent oid for each match
                        match.parent_oid = match.oid
                        # assign group type
                        match.group_type = "single"

                for match in matches:
                    # match already been retrieved by other criteria
                    existing_match = task_matches[file_idx].get(
                        match.oid,
                        None,
                    )

                    # match not retrieved yet
                    if existing_match is None:
                        task_matches[file_idx][match.oid] = match
                    # replace if current match has answer and old match has no answer
                    elif match.has_answer and not existing_match.has_answer:
                        task_matches[file_idx][match.oid] = match
                    # skip if current match don't have answer and old match has answer
                    elif existing_match.has_answer and not match.has_answer:
                        continue
                    # both matches have answer
                    elif existing_match.has_answer and match.has_answer:
                        # replace if new match has higher rank (smaller rank value)
                        if (
                            match.criteria.criteria_rank
                            < existing_match.criteria.criteria_rank
                        ):
                            task_matches[file_idx][match.oid] = match
                        # Current and existing match from the same match text. Replace only if QA score is more.
                        elif existing_match.oid == match.oid:
                            if (
                                match.raw_scores["squad_score"]
                                > existing_match.raw_scores["squad_score"]
                            ):
                                task_matches[file_idx][match.oid] = match
                            elif (
                                match.raw_scores["squad_score"]
                                == existing_match.raw_scores["squad_score"]
                                == 1
                                and match.raw_scores["boolq_score"]
                                > existing_match.raw_scores["boolq_score"]
                            ):
                                task_matches[file_idx][match.oid] = match
                        # current match has better raw match score, replacing
                        elif (
                            match.raw_scores["raw_match_score"]
                            > existing_match.raw_scores["raw_match_score"]
                        ):
                            task_matches[file_idx][match.oid] = match
                        # current match and existing match share same answer, boost
                        if existing_match.answer == match.answer:
                            task_matches[file_idx][match.oid].raw_scores[
                                "multi_criteria_score"
                            ] *= 1.1
                    # all matches don't have answer, keep old match
                    else:
                        pass

        for file_idx, matches in task_matches.items():
            task.matches[file_idx] = list(matches.values())
