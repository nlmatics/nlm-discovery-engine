from discovery_engine.objects import GroupTaskData
from engines.base_engine import BaseEngine

# do not remove below - dynamically loaded
from processors import AnswerPickerProcessor
from processors import CurrencyExtractorProcessor
from processors import EntityExtractionProcessor
from processors import NumberFormatterProcessor
from processors import RelationExtractionProcessor
from processors import SentenceClusteringProcessor
from processors.dynamic_value_processor.dynamic_value_processor import (
    DynamicValueProcessor,
)


REGISTERED_PROCESSORS = {
    "AnswerPickerProcessor",
    "NumberFormatterProcessor",
    "SentenceClusteringProcessor",
    "EntityExtractionProcessor",
    "CurrencyExtractorProcessor",
    "RelationExtractionProcessor",
}


class PostProcessorEngine(BaseEngine):
    def __init__(self, settings: dict = {}):
        super().__init__(settings)

    def run(self, group_task: GroupTaskData, **kwargs):
        for _, task in group_task.tasks.items():
            if task.post_processors:
                for post_processor in task.post_processors:
                    if post_processor and post_processor != [""]:
                        try:
                            if post_processor.endswith("()"):
                                processor_class = post_processor[:-2]
                                processor_args = ""
                            else:
                                # parse processor arguments
                                try:
                                    processor_class, processor_args = post_processor[
                                        :-1
                                    ].split("(", 1)
                                except ValueError:
                                    continue

                            # assign Process suffix if needed
                            if not processor_class.endswith("Processor"):
                                processor_class += "Processor"

                            # check if Processor
                            if processor_class not in REGISTERED_PROCESSORS:
                                self.logger.error(
                                    f"Processor {processor_class} is not registered",
                                )
                                continue

                            # run post processor
                            if processor_args:
                                processor = (
                                    f"{processor_class}({processor_args}, "
                                    f"settings=self.settings, documents=group_task.documents)"
                                )
                            else:
                                processor = f"{processor_class}(settings=self.settings, documents=group_task.documents)"
                            self.logger.info(f"init post processor {processor}")
                            processor = eval(processor)
                            processor(task)
                        except Exception:
                            self.logger.error(
                                "Error during post processing.",
                                exc_info=True,
                            )
            else:
                dynamic_value_processor = DynamicValueProcessor(self.settings)
                dynamic_value_processor.run(task)
