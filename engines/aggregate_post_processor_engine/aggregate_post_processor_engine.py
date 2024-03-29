from discovery_engine.objects import GroupTaskData
from engines.base_engine import BaseEngine
from processors import *  # noqa: F403, F401


REGISTERED_PROCESSORS = {
    "TopicProcessor",
}


class AggregatePostProcessorEngine(BaseEngine):
    def __init__(self, settings: dict = {}):

        super().__init__(settings)

    def run(self, group_task: GroupTaskData, **kwargs):
        for _, task in group_task.tasks.items():
            if task.matches and task.aggregate_post_processors:
                for post_processor in task.aggregate_post_processors:
                    if post_processor is not None and post_processor != [""]:
                        try:
                            if post_processor.endswith("()"):
                                processor_class = post_processor[:-2]
                                processor_args = ""
                            else:
                                try:
                                    # parse processor arguments
                                    processor_class, processor_args = post_processor[
                                        :-1
                                    ].split(
                                        "(",
                                        1,
                                    )
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
