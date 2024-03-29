import logging
from abc import ABCMeta
from abc import abstractmethod
from time import process_time
from timeit import default_timer

from discovery_engine.objects import GroupTaskData


class BaseEngine(metaclass=ABCMeta):
    def __init__(self, settings: dict = {}):
        self.logger = logging.getLogger(self.__class__.__name__)
        if settings["DEBUG"]:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        self.settings = settings

    def __call__(self, *args, **kwargs):

        wall_time = default_timer() * 1000
        clock_time = process_time() * 1000

        self.logger.info(f"Executing {self.__class__.__name__} with kwargs {kwargs}")

        res = self.run(*args, **kwargs)

        wall_time = default_timer() * 1000 - wall_time
        clock_time = process_time() * 1000 - clock_time

        self.logger.info(
            f"{self.__class__.__name__} finished. Wall time: {wall_time:.2f}ms, Clock time: {clock_time:.2f}ms",
        )
        return res

    @abstractmethod
    def run(self, task: GroupTaskData, **kwargs) -> GroupTaskData:
        """
        Processor
        This method takes a list of matches and returns the processed matches

        Args:
            matches: a list of input MatchData received from the pipeline

        Returns:
            matches: a list of output MatchData returns to the pipeline
        """
        pass
