from nlm_utils.parsers.value_parser import CountParser
from nlm_utils.parsers.value_parser import DateParser
from nlm_utils.parsers.value_parser import MoneyParser
from nlm_utils.parsers.value_parser import PercentParser
from nlm_utils.parsers.value_parser import PeriodParser

from discovery_engine.objects import TaskData
from processors.base_processor import BaseProcessor


class DynamicValueProcessor(BaseProcessor):
    processor_type = "post_processor"

    def __init__(
        self,
        settings: dict = {},
        **kwargs,
    ):
        super().__init__(settings)
        self.answer_type_parser_map = {
            "NUM:money": MoneyParser(),
            "NUM:perc": PercentParser(),
            "NUM:count": CountParser(),
            "NUM:date": DateParser(),
            "NUM:period": PeriodParser(),
        }

    def run(self, task: TaskData, *args, **kwargs):
        for criteria in task.criterias:
            for _, matches in criteria.matches.items():
                for match in matches:
                    if match.answer:
                        match.answer_details = self._run(
                            match.answer,
                            match.match_text,
                            criteria.expected_answer_type,
                        )
                        if match.answer_details:
                            match.formatted_answer = match.answer_details[
                                "formatted_value"
                            ]
                    if not (match.answer and match.answer_details):
                        # Add consistency to all the matches.
                        final_answer = None
                        if match.answer is not None:
                            final_answer = match.answer
                        match.answer_details = {
                            "formatted_value": final_answer,
                            "raw_value": final_answer,
                        }

    def _run(self, string_input, match_text, answer_type):
        if answer_type in self.answer_type_parser_map:
            try:
                parser = self.answer_type_parser_map[answer_type]
                return parser.parse(string_input)
            except Exception:
                pass
