import logging
from typing import Any
from typing import Optional

from pydantic import BaseModel

from de_utils import load_settings
from discovery_engine import DiscoveryEngine
from discovery_engine import VERSION


LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# from engine.services import ServiceFacade

settings = load_settings()


# passing the module into discovery engine instead of instance.
# this will let each worker (process) has it's own services


discovery_engine = DiscoveryEngine(settings)

for engine in [
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
]:
    discovery_engine.add_engine(engine)

discovery_engine.init_workers()


from fastapi import FastAPI

app = FastAPI()


@app.get("/healthz")
async def root():
    return {"message": "Welcome to Discovery Engine Lite services"}


@app.get("/version")
async def version():
    return VERSION


class ApplyTemplateParams(BaseModel):
    workspace_idx: Optional[str] = ""
    file_idx: Optional[str] = ""
    ad_hoc: Optional[bool] = False
    group_by_file: Optional[bool] = True
    search_type: Optional[str] = "extraction"
    field_bundle_idx: Optional[str] = ""
    override_topic: Optional[str] = ""
    criterias: Optional[Any] = None
    post_processors: Optional[list] = []
    aggregate_post_processors: Optional[list] = []
    doc_per_page: Optional[int] = 20
    offset: Optional[int] = 0
    match_per_doc: Optional[int] = 20
    topn: Optional[int] = -1
    batch_idx: Optional[str] = ""
    file_filter_struct: Optional[Any] = None
    extractors: Optional[list] = []
    disable_extraction: Optional[bool] = False
    abstractive_processors: Optional[list] = []
    user_acl: Optional[list] = []


@app.post("/apply_template")
async def apply_template(params: ApplyTemplateParams):
    """
    Apply template on a single file
    """

    data = discovery_engine.apply_template(**params.dict())

    return data
