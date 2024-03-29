from collections import defaultdict
from typing import Dict

from discovery_engine.objects import GroupTaskData
from engines.base_engine import BaseEngine


def debug_match(match):
    print("-" * 20)
    # print(
    #     match.match_idx,
    #     match.answer,
    #     match.group,
    #     match.raw_scores["scaled_score"],
    # )
    # print(match.json(indent=2, exclude={"criteria"}))
    print(
        match.json(
            indent=2,
            # exclude={"criteria", "table_data"},
            include={
                "match_idx",
                "match_text",
                "block_type",
                "group",
                "answer",
                "formatted_answer",
                "raw_scores",
                "bbox",
                "has_answer",
                "entity_list",
            },
        ),
    )


class PostRankingEngine(BaseEngine):
    def __init__(self, settings: dict = {}):
        super().__init__(settings)

    def run(self, group_task: GroupTaskData, **kwargs) -> Dict:
        tasks = group_task.tasks

        topics = kwargs["override_topic"]

        # loop over all matches to group by file_idx
        for topic in topics:
            self.logger.info(f"Processing topic '{topic}'")
            task = tasks[topic]

            for matches in task.matches.values():
                if not matches:
                    continue

                # calculate scaled_scores
                for match in matches:
                    # assign answer score, it is not used in ranking rn
                    original_score = max(match.raw_scores["match_score"], 0.2)
                    if self.settings["RUN_CROSS_ENCODER"]:
                        if "cross_encoder_score" in match.raw_scores:
                            original_score = match.raw_scores["cross_encoder_score"]
                        else:
                            original_score = 1.0

                    match.raw_scores["scaled_score"] = (
                        match.raw_scores["scaled_score"]
                        * original_score
                        * match.raw_scores["boolq_score"]
                        * match.raw_scores["squad_score"]
                        * match.raw_scores["qnli_score"]
                        * match.raw_scores["group_score"]
                    )
                    if match.block_type == "table":
                        match.raw_scores["scaled_score"] = (
                            match.raw_scores["scaled_score"]
                            * match.raw_scores["relevancy_score"]
                            * match.raw_scores["table_score"]
                        )

                # build groups
                groups = defaultdict(list)
                same_location_by_answer = defaultdict(set)
                for match in matches:
                    is_duplicate = False
                    if match.group_type == "same_location" and match.answer:
                        if match.answer in same_location_by_answer[match.group]:
                            is_duplicate = True
                        else:
                            same_location_by_answer[match.group].add(match.answer)
                    if not is_duplicate:
                        groups[match.group].append(match)
                groups = list(groups.values())

                # print("before sort")
                # for group_idx, group in enumerate(groups):
                #     # if table been retrieved by parent, remove parent
                #     for x in group:
                #         print(x.has_answer, group_idx)

                # break group if needed
                for group_idx, group in enumerate(groups):
                    # if table been retrieved by parent, remove parent
                    if len(group) == 2 and group[1].block_type == "table":
                        group.pop(0)

                # sort groups
                groups.sort(
                    key=lambda group: (
                        any([x.has_answer for x in group]),
                        max(x.raw_scores["scaled_score"] for x in group if x.has_answer)
                        if any([x.has_answer for x in group])
                        else max(x.raw_scores["scaled_score"] for x in group),
                        # max(x.raw_scores["boolq_score"] for x in group),
                        # max(x.raw_scores["qnli_score"] for x in group),
                    ),
                    reverse=True,
                )
                # print("after sort")
                # for group_idx, group in enumerate(groups):
                #     # if table been retrieved by parent, remove parent
                #     for x in group:
                #         print(x.has_answer, group_idx)

                # scaled_score
                latest_score = 1
                # rank of matches
                rank = {}
                # sort within group
                for group_idx, group in enumerate(groups):
                    group.sort(
                        key=lambda match: (
                            # parent always comes first
                            match.oid == match.parent_oid,
                            -match.match_idx
                            if match.group_type
                            in {"list_item", "table", "header_summary"}
                            else match.has_answer,
                            match.raw_scores["scaled_score"],
                        ),
                        reverse=True,
                    )

                    # if table been retrieved by parent, remove parent
                    if len(group) == 2 and group[1].block_type == "table":
                        group.pop(0)

                    # normalize scaled_score
                    for match in group:
                        if latest_score < match.raw_scores["scaled_score"]:
                            match.raw_scores["scaled_score"] = max(
                                0,
                                latest_score - 0.01,
                            )
                            pass
                        latest_score = match.raw_scores["scaled_score"]
                    latest_score = group[0].raw_scores["scaled_score"]

                    # build answer for parent match
                    if len(group) > 1:
                        parent_match = group[0]
                        if (
                            parent_match.block_type == "list_item"
                            and parent_match.has_answer is False
                        ):
                            answer = []
                            formatted_answer = []
                            # loop future matches
                            for next_match in group[1:]:
                                # reach to next group
                                if next_match.group != parent_match.group:
                                    break
                                # future match has answer
                                if next_match.has_answer:
                                    answer.append(next_match.answer)
                                    if next_match.formatted_answer:
                                        formatted_answer.append(
                                            next_match.formatted_answer,
                                        )
                            # generate answer
                            if answer:
                                parent_match.answer = "; ".join(answer)
                            if formatted_answer:
                                parent_match.formatted_answer = "; ".join(
                                    formatted_answer,
                                )

                    # for table match
                    if len(group) == 2 and group[0].group_type == "table":
                        text_match, table_match = group
                        if text_match.has_answer:
                            for match in group:
                                rank[match.oid] = len(rank)
                        else:
                            rank[table_match.oid] = len(rank)
                    else:
                        for match in group:
                            rank[match.oid] = len(rank)

                def sort_func(match):
                    return (
                        # parent_group_best_score[match.parent_oid],
                        match.has_answer,
                        match.answer != "",
                        match.raw_scores["scaled_score"],
                        # match.raw_scores["boolq_score"],
                    )

                matches.sort(key=sort_func, reverse=True)

                matches.sort(key=lambda x: rank.get(x.oid, float("inf")))

                matches.sort(key=lambda x: x.block_type == "summary", reverse=True)

                if self.settings["DEBUG"]:
                    for idx, match in enumerate(matches):
                        debug_match(match)
