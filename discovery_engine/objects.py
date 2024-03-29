from typing import Any
from typing import Dict
from typing import List
from typing import TypeVar

from nlm_utils.utils import query_preprocessing as preprocess
from pydantic import BaseModel
from xxhash import xxh32_hexdigest as hash

from de_utils.utils import is_bool_question
from de_utils.utils import is_question


class MatchDataBase(BaseModel):
    pass


class CriteriaDataBase(BaseModel):
    pass


class DocumentData(BaseModel):
    file_idx: str
    file_name: str
    file_title: str
    file_meta: dict = {}


class RelationData(BaseModel):
    head: str = None
    tail: str = None
    head_prob: float = 00
    tail_prob: float = 0.0


class MatchData(MatchDataBase):
    """
    MatchData is a basic unit of the retreieved information
    """

    # match
    match_idx: int
    match_text: str
    header_text: str
    hierarchy_headers: List[str] = []
    block_text: str
    entity_types: str
    qa_text: str
    parent_text: str
    # attributes
    oid: str = None
    parent_oid: str = None
    page_idx: int = -1
    block_type: str
    block_idx: int = -1
    # scores
    raw_scores: Dict[str, float] = {}
    # grouping
    group: int = 0
    group_type: str = "single"
    # table
    table: Dict[str, Any] = None
    table_answer: Dict[str, Any] = None
    table_data: TypeVar("pandas.core.frame.DataFrame") = None
    table_index: List[Dict[str, Any]] = []
    # criteria
    criteria: CriteriaDataBase = None
    # output
    has_answer: bool = False
    answer: str = ""
    formatted_answer: str = None
    answer_details: Dict[Any, Any] = {}
    boolean: bool = None
    uid: str = None
    explanation: Dict[Any, Any] = {}
    bbox: List[float] = []
    entity_list: List = []
    cross_references: Dict[str, List] = {}
    document_data: DocumentData = None
    relation_data: RelationData = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # init these values to 0
        for key, score in [
            ("match_score", 0),
            ("question_score", 0),
            ("qnli_score", 1),
            ("squad_score", 1),
            ("boolq_score", 1),
            ("file_score", 0),
            ("table_score", 0),
            ("relevancy_score", 0.5),
            ("answer_score", 1),
            ("scaled_score", 1),
            ("group_score", 1),
            ("multi_criteria_score", 1),
        ]:
            if key not in self.raw_scores:
                self.raw_scores[key] = score

        # backwards compatibility
        if self.parent_oid is None:
            self.parent_oid = self.oid

        self.uid = hash(
            f"{self.match_text}:{self.header_text}"  # text info
            f":{self.hierarchy_headers}:{self.block_text}",  # block info
        )

    # class Config:
    #     underscore_attrs_are_private = True
    class Config:
        orm_mode = True
        # feature will be released soon, use json(exclude={}) as temperary solution
        # exclude = {"criteria", "table_data", "table_index"}


MatchData.update_forward_refs()


class CriteriaData(CriteriaDataBase):
    """
    CriteriaData is a basic unit of the given query
    """

    question: str
    question_keywords_text: str = ""
    templates: List[str] = []
    headers: List[str] = []
    group_flag: str = "auto"
    table_flag: str = "auto"
    expected_answer_type: str = "auto"
    page_start: int = -1
    page_end: int = -1
    criteria_rank: int = -1
    is_bool_question: bool = None
    is_question: bool = None
    # matches will be [file_idx, List[MatchDataBase]]
    matches: Dict[str, List[MatchDataBase]] = {}
    uid: str = None
    enable_similar_search: bool = True
    entity_types: List[str] = None
    additional_questions: List[str] = []
    before_context_window: int = 0
    after_context_window: int = 0
    linguistic_keywords: List[str] = []
    linguistic_direct_keywords: List[str] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_question = is_question(self.question)
        self.is_bool_question = is_bool_question(self.question)
        if self.is_bool_question:
            self.expected_answer_type = "bool"

        # cleanup template
        templates = []
        for template in self.templates:
            template = template.strip()
            if template:
                templates.append(template)
        self.templates = templates

        # cleanup headers
        headers = []
        for header in self.headers:
            header = header.strip()
            if header:
                headers.append(header)
        self.headers = headers

        # cleanup question
        self.question = self.question.strip().lower()
        question_keywords = preprocess.filter_keywords(
            preprocess.get_words(self.question),
            preprocess.NLM_STOP_WORDS,
        )
        self.question_keywords_text = " ".join(question_keywords).strip()
        self.uid = hash(
            f"{self.templates}:{self.question}:{self.headers}"
            f":{self.group_flag}:{self.table_flag}:{self.expected_answer_type}"
            f":{self.page_start}{self.page_end}{self.criteria_rank}",
        )
        self.entity_types = self.entity_types or []
        self.additional_questions = self.additional_questions or []


CriteriaData.update_forward_refs()


class TaskData(BaseModel):
    """
    TaskData is a basic unit of the object for pipeline
    """

    topic: str
    topic_idx: str
    file_filter_text: str = ""
    post_processors: List[str] = []
    aggregate_post_processors: List[str] = []
    criterias: List[CriteriaData] = []
    matches: Dict[str, List[MatchData]] = {}
    aggs: Dict[str, Any] = {}
    pagination: Dict[str, Dict[str, int]] = {
        "workspace": {"offset": -1, "total": -1, "result_per_page": -1},
        "document": {"offset": -1, "total": -1, "result_per_page": -1},
    }
    group_by_file: bool = True
    uid: str = None
    search_type: str = "extraction"
    file_filter_struct: Dict[str, Any] = {}
    extractors: List[str] = []
    disable_extraction: bool = False
    abstractive_processors: List[str] = []
    user_acl: List[str] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uid = hash(
            f"{self.topic_idx}:{self.topic}:{self.file_filter_text}:"
            f"{':'.join([x.uid for x in self.criterias])}:"
            f"{self.post_processors}:{self.aggregate_post_processors}",
        )


class GroupTaskData(BaseModel):
    tasks: Dict[str, TaskData] = {}
    documents: Dict[str, DocumentData] = {}
    settings: Dict[str, Any] = {}
