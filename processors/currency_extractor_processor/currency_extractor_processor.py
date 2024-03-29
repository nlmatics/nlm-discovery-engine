import json
import os
import re
from collections import defaultdict
from collections import namedtuple

from discovery_engine.objects import TaskData
from processors.base_processor import BaseProcessor

Currency = namedtuple(
    "Currency",
    [
        "alpha3",  # unicode:       the ISO4217 alpha3 code
        "name",  # unicode:       the currency name
        "symbols",  # List[unicode]: list of possible symbols
    ],
)
CURRENCY_AMOUNT_PATTERN = r"(\d+\s*(?:,\d+)*(?:\.\d+)?)"


class CurrencyExtractorProcessor(BaseProcessor):

    processor_type = "post_processor"

    def __init__(
        self,
        debug=False,
        settings: dict = {},
        **kwargs,
    ):
        super().__init__(settings)
        self.currency_dict = None
        self.currency_symbol_set = None
        self.debug = debug
        self.get_currency_data()
        self.currency_pattern = r"""
            (?P<text>
                (?P<prefix>{currency_prefixes}|[{currency_symbols}])\s*
                (?P<amount>{amount_pattern})
                |
                (?P<amount1>{amount_pattern})\s*
                ((?P<postfix_name>{currency_names})|(?P<postfix_code>{currency_codes})(?:\W|$))
            )
        """.format(
            amount_pattern=CURRENCY_AMOUNT_PATTERN,
            currency_prefixes="|".join(
                re.escape(i) for i in self.get_currency_symbol_set() if len(i) > 1
            ),
            currency_symbols="".join(
                re.escape(i) for i in self.get_currency_symbol_set() if len(i) == 1
            ),
            currency_names="|".join(
                [i.replace(" ", "\\s+") + "[s]*" for i in self.get_currency_names()],
            ),
            currency_codes="|".join(self.get_currency_codes()),
        )

        self.currency_name_pattern = r"""
            {currency_names}
        """.format(
            currency_names="|".join(
                [i.replace(" ", "\\s+") for i in self.get_currency_names()],
            ),
        )

        self.currency_pattern_re = re.compile(
            self.currency_pattern,
            re.IGNORECASE | re.MULTILINE | re.DOTALL | re.VERBOSE,
        )
        self.currency_name_pattern_re = re.compile(
            self.currency_name_pattern,
            re.IGNORECASE | re.MULTILINE | re.DOTALL | re.VERBOSE,
        )

    def get_currency_data(self):
        """
        Reads the hand-curated JSON file (currency.json) which holds all the possible Symbols, prefix / postfix etc.
        Creates a singleton instance of currency_dictionary and Currency Symbol set.
        :return:
            1. Currency Dictionary with
                a) ALPHA 3 codes as the Key and Currency Tuple as Values
                b) Currency Symbol as Key and ALPHA 3 code as value, used for reverse mapping.
                c) Currency Name as Key and ALPHA 3 code as value, again for reverse mapping.
            2. Currency Symbol Set, which contains unique symbols for the currencies.
        """
        if not self.currency_dict:
            self.currency_dict = {}
            abs_path = os.path.realpath(
                os.path.join(os.getcwd(), os.path.dirname(__file__)),
            )
            with open(abs_path + "/currency.json", encoding="utf-8") as f:
                self.currency_dict["alpha3"] = {
                    k: Currency(**v) for k, v in json.load(f).items()
                }
                f.close()

            self.currency_dict["symbol"] = defaultdict(list)
            self.currency_dict["currency_names"] = {}
            currency_symbol_list = []

            for alpha3, curr in self.currency_dict["alpha3"].items():
                for sym in curr.symbols:
                    self.currency_dict["symbol"][sym] += [alpha3]
                    currency_symbol_list.append(sym)
                self.currency_dict["currency_names"][curr.name.lower()] = alpha3

            self.currency_symbol_set = set(currency_symbol_list)

        return self.currency_dict, self.currency_symbol_set

    def get_currency_dict(self):
        """
        Retrieves Currency Dictionary
        :return: Currency Dictionary
        """
        return self.get_currency_data()[0]

    def get_currency_symbol_set(self):
        """
        Retrieves Currency Symbol Set
        :return: Currency Symbol Set
        """
        return self.get_currency_data()[1]

    def get_currency_codes(self):
        """
        Retrieves Currency ALPHA3 Codes
        :return: Currency ALPHA3 Codes
        """
        return self.get_currency_data()[0]["alpha3"].keys()

    def get_currency_names(self):
        """
        Retrieves Currency Names
        :return: Currency Names
        """
        return self.get_currency_data()[0]["currency_names"].keys()

    def get_currency_code_from_symbol(self, sym):
        """
        Retrieves ALPHA3 value that matches the symbol
        :param sym: Currency Symbol. Prefix to the currency amount.
        :return: ALPHA3 value
        """
        return self.get_currency_dict()["symbol"][sym][0]

    def get_currency_code_from_currency_name(self, currency_name):
        """
        Retrieves ALPHA3 value that matches the symbol
        :param currency_name: Name of the currency, a possible value for Post fix to the amount.
        :return: ALPHA3 value
        """
        return self.get_currency_dict()["currency_names"][currency_name.lower()]

    def get_currency(self, string_input):
        """
        Retrieves ALPHA3 value for the currency match on the string input provided
        :param string_input: Input string on which the regular expression is matched upon.
        :return: ALPHA3 value
        """
        for match in self.currency_pattern_re.finditer(string_input):
            match_dict = match.groupdict()
            if self.debug:
                # print(match_dict)
                self.logger.info(
                    f"Currency Match dictionary for {string_input} is {match_dict}",
                )
            if match_dict["prefix"]:
                return self.get_currency_code_from_symbol(match_dict["prefix"])
            elif match_dict["postfix_name"]:
                matched_name_str = match_dict["postfix_name"]
                name_match = self.currency_name_pattern_re.match(matched_name_str)
                return self.get_currency_code_from_currency_name(name_match.group())
            elif match_dict["postfix_code"]:
                return match_dict["postfix_code"]
            else:
                self.logger.info(f"No Currency Match for {string_input}")
                return None

    def run(self, task: TaskData, *args, **kwargs):
        for criteria in task.criterias:
            for _, matches in criteria.matches.items():
                for match in matches:
                    match.formatted_answer = self._run(
                        match.answer,
                        match.match_text,
                    )

    def _run(self, string_input, match_text):
        result = None
        if string_input:
            result = self.get_currency(string_input)
        if not result:
            result = self.get_currency(match_text)
        return result
