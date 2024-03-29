import logging
from abc import ABCMeta
from abc import abstractmethod
from time import process_time
from timeit import default_timer

from discovery_engine.objects import TaskData

REGISTERED_PROCESSORS = {
    "post_processor",
    "agg_post_processor",
    "abstractive_processor",
}


class BaseProcessor(metaclass=ABCMeta):
    def __init__(self, settings: dict):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.settings = settings
        if not hasattr(self, "processor_type"):
            raise ValueError("Must specify processor_type")

        if self.processor_type not in REGISTERED_PROCESSORS:
            raise ValueError(
                'Processor mush be either "post_processor" or "agg_post_processor" or "abstractive_processor"',
            )

    def __call__(self, *args, **kwargs):
        wall_time = default_timer()
        clock_time = process_time()
        self.logger.info(f"Executing {self.__class__.__name__}")

        res = self.run(*args, **kwargs)

        wall_time = default_timer() - wall_time
        clock_time = process_time() - clock_time

        self.logger.info(
            f"{self.__class__.__name__} Finished. Wall time: {wall_time:.4f}s, Clock time: {clock_time:.4f}s",
        )
        return res

    @abstractmethod
    def run(self, task: TaskData, **kwargs) -> TaskData:
        """
        Processor
        This method takes a list of matches and returns the processed matches

        Args:
            matches: a list of input MatchData received from the pipeline
            :param task:

        Returns:
            matches: a list of output MatchData returns to the pipeline
        """
        raise NotImplementedError
