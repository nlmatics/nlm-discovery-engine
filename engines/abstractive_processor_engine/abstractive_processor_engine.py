from discovery_engine.objects import GroupTaskData
from engines.base_engine import BaseEngine

# do not remove below - dynamically loaded
from processors import AbstractiveSummaryProcessor


REGISTERED_ABSTRACTIVE_PROCESSORS = {
    "AbstractiveSummaryProcessor",
}


class AbstractiveProcessorEngine(BaseEngine):
    def __init__(self, settings: dict = {}):
        super().__init__(settings)

    def run(self, group_task: GroupTaskData, **kwargs):

        if not kwargs["ad_hoc"]:
            self.logger.info(
                "AbstractiveProcessorEngine is applied only for Ad-Hoc Search",
            )
            return

        abstractive_processors = []
        enable_workspace_summarization = group_task.settings.get(
            "search_settings",
            {},
        ).get(
            "enable_workspace_summarization",
            False,
        )
        enable_summarization_by_default = group_task.settings.get(
            "search_settings",
            {},
        ).get(
            "enable_summarization_by_default",
            False,
        )

        for _, task in group_task.tasks.items():
            abstractive_processors.extend(task.abstractive_processors)
            if enable_summarization_by_default or enable_workspace_summarization:
                list_contains_summarization = False
                for abs_proc in abstractive_processors:
                    if abs_proc.startswith("AbstractiveSummaryProcessor"):
                        list_contains_summarization = True
                        break
                if not list_contains_summarization:
                    if kwargs["ad_hoc"] and kwargs["file_idx"] is None:
                        if enable_workspace_summarization:
                            abstractive_processors.append(
                                "AbstractiveSummaryProcessor()",
                            )
                    elif (
                        kwargs["ad_hoc"]
                        and kwargs["file_idx"]
                        and isinstance(kwargs["file_idx"], str)
                    ):
                        abstractive_processors.append("AbstractiveSummaryProcessor()")

            self.logger.info(
                f"Applying AbstractiveProcessorEngine '{abstractive_processors}'",
            )
            if abstractive_processors:
                for abstractive_processor in abstractive_processors:
                    if abstractive_processor and abstractive_processor != [""]:
                        try:
                            if abstractive_processor.endswith("()"):
                                processor_class = abstractive_processor[:-2]
                                processor_args = ""
                            else:
                                # parse processor arguments
                                try:
                                    (
                                        processor_class,
                                        processor_args,
                                    ) = abstractive_processor[:-1].split("(", 1)
                                except ValueError:
                                    continue

                            # assign Process suffix if needed
                            if not processor_class.endswith("Processor"):
                                processor_class += "Processor"

                            # check if Processor
                            if processor_class not in REGISTERED_ABSTRACTIVE_PROCESSORS:
                                self.logger.error(
                                    f"Abstractive Processor {processor_class} is not registered",
                                )
                                continue

                            # run abstractive processor
                            if processor_args:
                                processor = (
                                    f"{processor_class}({processor_args}, "
                                    f"settings=self.settings, group_task_settings=group_task.settings)"
                                )
                            else:
                                processor = (
                                    f"{processor_class}(settings=self.settings, "
                                    f"group_task_settings=group_task.settings)"
                                )
                            self.logger.info(f"init abstractive processor {processor}")
                            processor = eval(processor)
                            processor(task, **kwargs)
                        except Exception:
                            self.logger.error(
                                "Error during abstractive processing.",
                                exc_info=True,
                            )
