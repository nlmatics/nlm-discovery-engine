import re

import pandas as pd

from discovery_engine.objects import TaskData
from processors.base_processor import BaseProcessor

UNITS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
TENS = [
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]
SCALES = ["hundred", "thousand", "million", "billion", "trillion"]


class NumberFormatterProcessor(BaseProcessor):

    processor_type = "post_processor"

    def __init__(
        self,
        answer_type="MONEY",
        only_nums=False,
        settings: dict = {},
        **kwargs,
    ):

        """
        self.answer_type in ["MONEY", "UNITS"]
        """
        super().__init__(settings)
        self.money_scales = re.compile(
            r"(^million$|^millions$|^bn$|^dollars$|^dollar$|^thousand$|^billion$|^billions$|^hundred$)",
        )
        self.acre_scales = re.compile(
            r"(^sqft$|^sq-ft$|^foot$|^square$|^foot$|^feet$|^acres$|^ft$|^acre$|^sft$|^sf$|^sq$|^gfs$|^nsf$)",
        )
        self.story_scales = re.compile(r"(^story$|^stories$|^floor$|^floors$|^height$)")

        self.year_scales = re.compile(r"(^years$|^year$|^yr$|^yrs$)")
        self.unwanted_words = re.compile(r"('gross$|^max$|^net$|^rentable$)")
        self.single_scales = ["m", "bn", "b", "k", "mm", "mil", "m", "$", "€", "¥"]

        self.denomination = ["$", "€", "¥", "£"]
        self.percent = re.compile(r"(^percent$|^%$|^percentage$)")
        self.other_units = re.compile(r"(^shares$)")
        self.sf = re.compile(r"(^nsf$|^sft$|^sf$|^gsf$|^square$|^sq-ft$|^sqft$|^sq$)")

        self.money_scales_rules = re.compile(r"(million|bn|thousand|billion|hundred)")
        self.number_rule = re.compile(r"[0-9,.]+")

        self.answer_type = answer_type
        self.only_nums = only_nums

    def run(self, task: TaskData, *args, **kwargs):
        for criteria in task.criterias:
            for _, matches in criteria.matches.items():
                for match in matches:
                    unit_search_text = f"{match.header_text} {match.qa_text}"
                    if isinstance(match.table_data, pd.DataFrame):
                        unit_search_text += (
                            f"{match.table_data.index.names}"
                            f" {match.table_data.index.name}"
                        )

                    match.formatted_answer = self._run(
                        match.answer,
                        match.match_text,
                        unit_search_text,
                    )

    def number_in_string(self, s):
        number = ""
        pointer = 0
        for i in s:
            if i.isdigit() or i == ".":
                number += i
                if i == ".":
                    pointer += 1
        if number != "":
            return number, pointer
        else:
            return "", pointer

    def check_money_scales(self, s):
        s = s.lower()
        million = ("million", "mil", "mm", "mill", "mln", "m")
        billion = ("billion", "bil", "bn", "B", "bill", "bln")
        trillion = ("trillion", "t", "tril", "trill", "trn", "tln")
        thousand = ("thousand", "k", "thsnd")
        hundred = ("hundred", "hund")

        if s.endswith(million):
            return "million"
        elif s.endswith(billion):
            return "billion"
        elif s.endswith(thousand):
            return "thousand"
        elif s.endswith(hundred):
            return "hundred"
        elif s.endswith(trillion):
            return "trillion"
        else:
            return ""

    def text2int(self, textnum):
        numwords = {}
        if not numwords:
            numwords["and"] = (1, 0)
            for idx, word in enumerate(UNITS):
                numwords[word] = (1, idx)
            for idx, word in enumerate(TENS):
                numwords[word] = (1, idx * 10)
            for idx, word in enumerate(SCALES):
                numwords[word] = (10 ** (idx * 3 or 2), 0)

        current = result = 0
        for word in textnum.split():
            if word not in numwords:
                return None

            scale, increment = numwords[word]
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0

        return result + current

    def remove_punctuation(self, string, remove=""):
        string = string.lower()
        punctuations = r"""!()-[]{};:'"\<>/?@#^&*_~"""
        other_punctuation = ""","""
        for x in string:

            if x in punctuations:
                string = string.replace(x, " ")
            elif x in other_punctuation:
                string = string.replace(x, "")
            if remove == "$":
                string = string.replace("$", "")

        if string.endswith("."):
            string = string.rstrip(".")
        #         for i in range(len(string)):
        #             if (
        #                 string[i] == "."
        #                 and not string[i + 1].isdigit()
        #                 and not string[i - 1].isdigit()
        #             ):
        #                 string = string.replace(string[i], " ")
        return string

    def text2int_replace(self, text):
        string = self.remove_punctuation(text) + " /"

        string1 = ""
        s = []
        for i in string.split():
            if i in UNITS or i in TENS:
                string1 = string1 + i + " "
            elif i not in UNITS or i not in TENS or string1 != "":
                string1 = string1.rstrip()
                s.append(string1)
                string1 = ""
        for i in s:
            if i == "":
                s.remove(i)

        for i in s:
            if i != "":
                string = string.replace(i, str(self.text2int(i)))

        return string

    def add_comma(self, s):
        result = ""
        if "." not in s:
            for i, c in enumerate(s[::-1]):
                if s.isdigit():
                    if (i + 1) % 3 == 0 and i < len(s) - 1:
                        result = "," + c + result
                    else:
                        result = c + result
                else:
                    if (i + 1) % 3 == 0 and i < len(s) - 2:
                        result = "," + c + result
                    else:
                        result = c + result
        else:
            result = s
        return result

    def wordNum(self, string2):
        string2 = string2.lower()
        if string2.endswith(" "):
            string2 = string2[: -len(" ")]
        elif string2.startswith(" "):
            string2 = string2[len(" ") :]
        if string2.endswith("."):
            string2 = string2[: -len(".")]
        if (
            self.number_in_string(string2)[1] > 0
            and self.check_money_scales(string2) != ""
        ):
            final = int(
                float(self.number_in_string(string2)[0])
                * self.text2int(f"one {self.check_money_scales(string2)}"),
            )
        elif (
            self.number_in_string(string2)[1] > 0
            and self.check_money_scales(string2) == ""
        ):
            final = self.number_in_string(string2)[0]
        elif (
            self.number_in_string(string2)[0] != ""
            and self.check_money_scales(string2) == ""
        ):
            final = self.number_in_string(string2)[0]
        elif (
            self.number_in_string(string2)[0] != ""
            and self.check_money_scales(string2) != ""
        ):
            final = int(self.number_in_string(string2)[0]) * int(
                self.text2int(f"one {self.check_money_scales(string2)}"),
            )
        else:
            string2 = re.sub(r"[^\w\s]", "", string2)
            final = self.text2int(string2)
        return str(final)

    def remove_words(self, query):

        querywords = query.split()
        for i in range(len(querywords)):
            if querywords[i] in self.denomination and querywords[i + 1][0].isdigit():
                querywords[i + 1] = querywords[i] + querywords[i + 1]
                querywords[i] = ""
            if (
                len(querywords[i]) > 6
                and querywords[i][1].isdigit()
                and querywords[i].find(".") != -1
            ):
                #                 if querywords[i][querywords[i].find('.')+1]=='0':

                querywords[i] = querywords[i][: querywords[i].find(".")]

            if querywords[i].find("$") != -1:

                querywords[i] = querywords[i][querywords[i].find("$") :]

        resultwords = [
            word for word in querywords if not self.unwanted_words.match(word.lower())
        ]
        result = " ".join(resultwords)
        result = (
            result.replace("single", "1").replace("double", "2").replace("triple", "3")
        )
        return result + " /"

    def num_formatter(self, text, answer_type="money"):

        text = self.text2int_replace(text)
        text = self.remove_words(text)
        t = text.split()

        money = []
        units = []

        for i in range(len(t)):

            if (
                t[i].replace(".", "", 1).isdigit()
                or t[i][0] in self.denomination
                or t[i][-1] in self.denomination
                or t[i].endswith(tuple(self.single_scales))
                and len(t[i]) > 1
                and t[i][0].isdigit()
                or t[i].endswith("%")
                and len(t[i]) > 1
            ):

                # MONEY APPEND
                if i + 1 < len(t):
                    if (
                        self.money_scales.match(t[i + 1])
                        or t[i + 1] in self.single_scales
                        or t[i][0] in self.denomination
                        and len(t[i]) > 1
                        or t[i][-1] in self.denomination
                        and len(t[i]) > 1
                        or t[i].endswith(tuple(self.single_scales))
                        or len(t[i]) > 7
                        and t[i][-1] != "%"
                        and not self.story_scales.match(t[i + 1])
                        and not self.acre_scales.match(t[i + 1])
                        and not self.other_units.match(t[i + 1])
                    ):
                        mil_word = ""

                        if (
                            t[i][0] not in self.denomination
                            and t[i][-1] not in self.denomination
                        ):
                            if (
                                self.money_scales.match(t[i + 1])
                                or t[i + 1] in self.single_scales
                            ):
                                mil_word = t[i] + " " + t[i + 1]
                                money.append(
                                    self.add_comma("$" + self.wordNum(mil_word)),
                                )
                                continue
                            money.append(self.add_comma("$" + self.wordNum(t[i])))
                        elif t[i][0] in self.denomination:
                            if (
                                self.money_scales.match(t[i + 1])
                                or t[i + 1] in self.single_scales
                            ):
                                mil_word = t[i] + " " + t[i + 1]

                                money.append(
                                    self.add_comma(t[i][0] + self.wordNum(mil_word)),
                                )
                                continue
                            money.append(self.add_comma(t[i][0] + self.wordNum(t[i])))
                        elif t[i][-1] in self.denomination:
                            if (
                                self.money_scales.match(t[i + 1])
                                or t[i + 1] in self.single_scales
                            ):
                                mil_word = t[i] + " " + t[i + 1]

                                money.append(
                                    self.add_comma(t[i][-1] + self.wordNum(mil_word)),
                                )
                                continue
                            money.append(self.add_comma(t[i][-1] + self.wordNum(t[i])))
                        # STORY APPEND
                    elif (
                        self.story_scales.match(t[i + 1])
                        and t[i].replace(".", "", 1).isdigit()
                    ):

                        suffix = t[i + 1]

                        story_word = self.add_comma(t[i]) + " " + suffix
                        units.append(story_word)
                        continue
                    elif (
                        self.story_scales.match(t[i - 1])
                        and t[i].replace(".", "", 1).isdigit()
                        and not t[i - 2].replace(".", "", 1).isdigit()
                    ):
                        prefix = t[i - 1]
                        story_word = self.add_comma(t[i]) + " " + prefix

                        units.append(story_word)
                        continue

                    # ACRE APPEND
                    elif (
                        self.acre_scales.match(t[i + 1])
                        and t[i].replace(".", "", 1).isdigit()
                    ):

                        suffix = t[i + 1]
                        if self.sf.match(suffix):
                            suffix = "sf"
                        acre_word = self.add_comma(t[i]) + " " + suffix
                        units.append(acre_word)
                        continue
                    elif (
                        self.acre_scales.match(t[i - 1])
                        and t[i].replace(".", "", 1).isdigit()
                        and not t[i - 2].replace(".", "", 1).isdigit()
                    ):
                        prefix = t[i - 1]
                        acre_word = self.add_comma(t[i]) + " " + prefix
                        units.append(acre_word)
                        continue

                    # PERCENT APPEND
                    elif (
                        t[i].replace(".", "", 1).isdigit()
                        and self.percent.match(t[i + 1])
                        or t[i].endswith("%")
                        and len(t[i]) > 1
                    ):
                        if self.percent.match(t[i + 1]):
                            t[i] += "%"
                        units.append(t[i])
                    # YEARS APPEND
                    elif t[i].isdigit() and self.year_scales.match(t[i + 1]):
                        t[i] += " " + t[i + 1]
                        units.append(t[i])
                        # UNITS APPEND
                    elif t[i].replace(".", "", 1).isdigit():
                        units.append(self.add_comma(t[i]))

        if answer_type == "MONEY":
            return money
        elif answer_type == "UNITS":
            return units
        elif answer_type == "ALL":
            all_nums = []
            all_num = [money, units]
            for i in all_num:
                if i:
                    for j in i:
                        all_nums.append(j)
            return all_nums

    # Run function which takes input_string, match_string and ans_type

    def check_multiplier(self, match_text):
        if not match_text:
            return 1
        elif "in billion" in match_text:
            return 1_000_000_000
        elif "in million" in match_text:
            return 1_000_000
        elif "in thousand" in match_text:
            return 1_000
        elif "in hundred" in match_text:
            return 100
        else:
            return 1

    def _run(self, raw_input_string, text, qa_text=""):
        if raw_input_string == "" or text == "":
            return ""
        newlist = []

        # extracts all values from match string
        mylist = self.num_formatter(text, answer_type=self.answer_type)

        # extract values from input string
        input_string = self.num_formatter(raw_input_string, answer_type="ALL")

        # Matches keyword in the match string
        for i in input_string:

            if i.startswith("$"):
                i = i.replace("$", "")
            if newlist == []:
                r = re.compile(f".*{i}")
                newlist = list(filter(r.match, mylist))

                continue
            else:
                break

        if not newlist:
            return ""

        if self.only_nums and len(newlist[0]) > 1:
            text = newlist[0].split()[0]

        elif not self.only_nums:
            text = newlist[0]
        else:
            return ""

        if self.answer_type == "MONEY" and not self.money_scales_rules.findall(
            raw_input_string,
        ):
            # raw output
            number = self.number_rule.findall(text)[0]

            # get multiplier
            multiplier = self.check_multiplier(qa_text)

            # apply multiplier
            new_number = float(number.replace(",", "")) * multiplier

            if new_number - int(new_number) != 0:
                new_number = f"{new_number:,}"
            else:
                new_number = f"{int(new_number):,d}"

            text = text.replace(number, new_number)

        return text
