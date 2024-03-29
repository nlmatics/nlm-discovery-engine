from typing import Dict

import pandas as pd

from discovery_engine.objects import GroupTaskData
from engines.base_engine import BaseEngine


class TableExtractionEngine(BaseEngine):
    def __init__(self, settings: dict = {}):
        super().__init__(settings)

    def run(self, group_task: GroupTaskData, **kwargs) -> Dict:
        tasks = group_task.tasks

        topics = kwargs["override_topic"]

        # loop over all matches to group by file_idx
        for topic in topics:
            self.logger.info(f"Processing topic '{topic}'")
            task = tasks[topic]
            for criteria in task.criterias:
                # do not run table search on bool question
                if criteria.is_bool_question:
                    continue

                for _, matches in criteria.matches.items():
                    for match in matches:
                        if match.block_type != "table":
                            continue
                        try:
                            self._run_match(match)
                        except Exception as e:
                            match.has_answer = False
                            match.answer = ""
                            match.raw_scores["match_score"] = 0
                            self.logger.error(
                                f"failed to run {match.match_idx} {e}",
                                exc_info=True,
                            )

    def _run_match(self, match):

        # match does not contain table
        if match.table_data is None or match.table_index is None:
            self.logger.error(f"Table with match_idx {match.match_idx} not avaliable")
            return

        self.logger.debug("start extract on table")

        # get the dataframe
        # df = pickle.loads(match.table_data)
        df = match.table_data
        self.logger.debug(f"Extracting table with shape {df.shape}")

        # sort by original order of the table
        match.table_index.sort(key=lambda x: x["idx"])

        cols = []
        rows = []
        index_cols = []

        for info in match.table_index:
            if info["type"] == "row":
                rows.append(info)
            elif info["type"] == "col":
                cols.append(info)
            else:
                index_cols.append(info)

        # from pprint import pprint as print
        # print("=" * 20)
        # print(match.dict())
        # print("=" * 20)
        # for i in sorted(match.table_index, key=lambda x: x["score"]):
        #     print("-" * 20)
        #     print(i)
        # print('='*20)
        # print(match.json())
        # print("+"*20)
        # print(cols)
        # print("+"*20)
        # print(rows)

        best_match_threshold = (
            1  # if match.raw_scores["relevancy_score"] == 1 else 0.99
        )

        best_col_score = 0
        best_cols = []
        if cols:
            best_col_score = max(cols, key=lambda x: x["score"])["score"]
            best_cols = [
                x["idx"]
                for x in cols
                if x["score"] >= best_col_score * best_match_threshold
            ]
            cols = [x["idx"] for x in cols]

        best_row_score = 0
        best_rows = []
        if rows:
            best_row_score = max(rows, key=lambda x: x["score"])["score"]
            best_rows = [
                x["idx"]
                for x in rows
                if x["score"] >= best_row_score * best_match_threshold
            ]
            rows = [x["idx"] for x in rows]

        best_index_cols_score = 0
        if index_cols:
            best_index_cols_score = max(index_cols, key=lambda x: x["score"])["score"]

        # scale match_score to consider both col and row
        match.raw_scores["match_score"] = min(
            1,
            match.raw_scores["match_score"]
            * (best_row_score + best_col_score + best_index_cols_score)
            * max(best_row_score, best_col_score, best_index_cols_score),
        )

        # build answer
        # see if top 1 match is perfect match
        if best_cols and best_rows:
            best_answer_df = df.iloc[best_rows, best_cols]
        elif best_rows and not best_cols:
            best_answer_df = df.iloc[best_rows]
        elif not best_rows and best_cols:
            best_answer_df = df.iloc[:, best_cols]
        # special logic for index_cols
        elif index_cols:
            best_answer_df = df.reset_index()
            # create es record for index
            assert not isinstance(df.index, pd.RangeIndex)
            names = [x["text"] for x in index_cols]
            best_answer_df = df.reset_index()[names]
        else:
            best_answer_df = df
            # best_answer_df = pd.DataFrame()
            # raise RuntimeError("both rols and cols are not retrived for best_answer_df")

        # build match_text
        # both cols and
        if cols and rows:
            df = df.iloc[rows, cols]
        elif cols and not rows:
            df = df.iloc[:, cols].T
        elif not cols and rows:
            df = df.iloc[rows]
        elif index_cols:
            df = best_answer_df
        # table is retrieved by parent
        else:
            pass

        best_answer_df = self.clean_dataframe(best_answer_df)
        df = self.clean_dataframe(df)

        # build answer
        if df.empty:
            match.table = ""
            match.answer = ""
            match.raw_scores["table_score"] = 0
            # match.raw_scores["match_score"] = 0
            match.has_answer = False
        else:
            match.table = self.convert_df_to_ad_grid_json(df)
            # found table but no best answer, skip this table
            if best_answer_df.empty:
                match.answer = ""
                match.raw_scores["table_score"] = 0.9
                match.has_answer = match.raw_scores["relevancy_score"] > 0.9
                match.table_answer = self.convert_df_to_ad_grid_json(df)
            # found perfect answer in table
            elif best_answer_df.shape[0] * best_answer_df.shape[1] == 1:
                data = self.convert_df_to_ad_grid_json(best_answer_df)
                match.match_text = self.generate_key_value_pairs(data)
                match.answer = best_answer_df.values[0][0]
                match.table_answer = self.convert_df_to_ad_grid_json(best_answer_df)
                match.raw_scores["table_score"] = 1
                match.has_answer = True
            # found some answers in table
            else:
                # raise
                data = self.convert_df_to_ad_grid_json(best_answer_df)
                match.answer = match.match_text = self.generate_key_value_pairs(data)
                match.table_answer = self.convert_df_to_ad_grid_json(best_answer_df)
                # down weight if cell is a long text
                data = []
                for v in best_answer_df.to_dict(orient="list").values():
                    data.extend([len(str(x).split()) for x in v])
                # cell have more than 5 tokens, down-weight
                if data and sum(data) / len(data) >= 5:
                    match.raw_scores["table_score"] = 0.3
                else:
                    match.raw_scores["table_score"] = 0.75
                match.has_answer = True

            match.raw_scores["match_score"] *= match.raw_scores["table_score"]

        self.logger.debug("Table extracted")

    def clean_dataframe(self, df):
        """
        replace internal index name with empty string to better display
        """

        def rename_index(index):
            def _replace(index):
                # internal used string index
                if isinstance(index, str) and (
                    index.startswith("_UNKNOWN_") or index.startswith("_MULTI")
                ):
                    return ""
                # int is pd auto generated index
                if isinstance(index, int):
                    return ""
                # good index
                else:
                    return index

            if isinstance(index, str) or isinstance(index, int):
                return _replace(index)
            elif isinstance(index, list) or isinstance(index, tuple):
                return tuple(_replace(x) for x in index)
            else:
                return index

        # rename columns
        df.columns = df.columns.map(rename_index)
        # rename index
        df.index = df.index.map(rename_index)
        # remove index name
        if df.index.nlevels > 1:
            index_name = [""] * df.index.nlevels
        else:
            index_name = ""
        df.index = df.index.rename(index_name)

        df = df.dropna(axis="index")
        df = df.dropna(axis="columns")
        # # remove duplicated rows
        # df = df.drop_duplicates()
        # # remove duplicated columns
        # df = df.T.drop_duplicates().T

        if df.empty:
            df = pd.DataFrame()

        return df

    def convert_df_to_ad_grid_json(self, df):
        cols = df.columns.get_level_values(-1).tolist()
        rows = []
        index_names = df.index.get_level_values(-1).tolist()
        # clean up empty or inferred index_names
        index_names = [x if x != "" else "" for x in index_names]

        for idx, (_, row) in enumerate(df.iterrows()):
            if index_names:
                rows.append([index_names[idx]] + row.values.tolist())
            else:
                rows.append(row.values.tolist())
        res = {"cols": cols, "rows": rows}
        return res

    def generate_key_value_pairs(self, data):
        rows = data["rows"]
        columns = data["cols"]
        if not rows or not columns:
            return ""
        answer = [f"{rows[0][0]}"]
        for column, row in zip(columns, rows[0][1:]):
            answer.append(f"{column}: {row}")
        return "\n".join(answer).strip()
