import os
import re
import string
from typing import List
from typing import Set

import ahocorasick
import tiktoken
from nlm_utils.utils import ensure_bool
from nlm_utils.utils import ensure_float
from nltk.corpus import stopwords
from server.storage import nosql_db


OPENAI_MAX_NUM_TOKENS = 3000
openai_tokenizer = tiktoken.get_encoding(
    "cl100k_base",
)

NLTK_STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS = {
    "a",
    "again",
    "about",
    "after",
    "afterwards",
    "almost",
    "along",
    "already",
    "also",
    "although",
    "always",
    "am",
    "among",
    "amongst",
    "amoungst",
    "amount",
    "an",
    "and",
    "another",
    "any",
    "anyhow",
    "anyone",
    "anything",
    "anyway",
    "anywhere",
    "are",
    "around",
    "as",
    "at",
    "back",
    "be",
    "became",
    "because",
    "become",
    "becomes",
    "becoming",
    "been",
    "before",
    "beforehand",
    "behind",
    "being",
    "below",
    "beside",
    "besides",
    "between",
    "beyond",
    "bill",
    "both",
    "but",
    "by",
    "call",
    "can",
    "cannot",
    "cant",
    "co",
    "con",
    "could",
    "couldnt",
    "cry",
    "de",
    "describe",
    "detail",
    "did",
    "don't",
    "don'",
    "do",
    "done",
    "down",
    "due",
    "during",
    "each",
    "eg",
    "e.g.",
    "ex",
    "eight",
    "either",
    "eleven",
    "else",
    "elsewhere",
    "empty",
    "enough",
    "etc",
    "even",
    "ever",
    "every",
    "everyone",
    "everything",
    "everywhere",
    "except",
    "few",
    "fifteen",
    "fify",
    "fill",
    "find",
    "fire",
    "first",
    "five",
    "for",
    "former",
    "formerly",
    "forty",
    "found",
    "four",
    "from",
    "front",
    "full",
    "further",
    "get",
    "give",
    "go",
    "had",
    "has",
    "hasnt",
    "have",
    "he",
    "hence",
    "her",
    "here",
    "hereafter",
    "hereby",
    "herein",
    "hereupon",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "however",
    "hundred",
    "i",
    "ie",
    "if",
    "in",
    "inc",
    "indeed",
    "interest",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "keep",
    "last",
    "latter",
    "latterly",
    "least",
    "less",
    "ltd",
    "made",
    "many",
    "may",
    "me",
    "meanwhile",
    "might",
    "mill",
    "mine",
    "more",
    "moreover",
    "most",
    "mostly",
    "move",
    "much",
    "must",
    "my",
    "myself",
    "name",
    "namely",
    "neither",
    "never",
    "nevertheless",
    "next",
    "nine",
    "no",
    "nobody",
    "none",
    "noone",
    "nor",
    "not",
    "nothing",
    "now",
    "nowhere",
    "of",
    "off",
    "often",
    "on",
    "once",
    "one",
    "only",
    "onto",
    "or",
    "other",
    "others",
    "otherwise",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "part",
    "per",
    "perhaps",
    "please",
    "put",
    "rather",
    "re",
    "s",
    "same",
    "see",
    "seem",
    "seemed",
    "seeming",
    "seems",
    "serious",
    "several",
    "she",
    "should",
    "show",
    "side",
    "since",
    "sincere",
    "six",
    "sixty",
    "so",
    "some",
    "somehow",
    "someone",
    "something",
    "sometime",
    "sometimes",
    "somewhere",
    "still",
    "such",
    "system",
    "take",
    "ten",
    "than",
    "that",
    "the",
    "their",
    "them",
    "themselves",
    "then",
    "thence",
    "there",
    "thereafter",
    "thereby",
    "therefore",
    "therein",
    "thereupon",
    "these",
    "they",
    "thick",
    "thin",
    "third",
    "this",
    "those",
    "though",
    "three",
    "through",
    "throughout",
    "thru",
    "thus",
    "to",
    "together",
    "too",
    "top",
    "toward",
    "towards",
    "twelve",
    "twenty",
    "two",
    "un",
    "under",
    "until",
    "up",
    "upon",
    "us",
    "very",
    "via",
    "was",
    "we",
    "well",
    "were",
    "what",
    "whatever",
    "when",
    "whence",
    "whenever",
    "where",
    "whereafter",
    "whereas",
    "whereby",
    "wherein",
    "whereupon",
    "wherever",
    "whether",
    "which",
    "whoever",
    "yet",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "the",
}

QUESTION_WORDS = {
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "which",
    "whose",
    "whom",
}

BOOL_QUESTION_WORDS = {
    "is",
    "isn't",
    "was",
    "were",
    "do",
    "does",
    "will",
    "can",
    "are",
    "has",
    "have",
    "had",
    "did",
    "should",
    "would",
    "could",
    "need",
    "must",
    "shall",
}

PUNCT_TABLE = str.maketrans(dict.fromkeys(string.punctuation))
quotation_pattern = re.compile(r'[”“"‘’\']')


def is_question(sent: str):
    parts = sent.split("|")
    if len(parts) > 1:
        sent = parts[1].strip()
    tokens = sent.lower().split()
    return len(tokens) > 1 and (
        tokens[0] in QUESTION_WORDS or tokens[0] in BOOL_QUESTION_WORDS
    )


def is_bool_question(sent: str):
    tokens = sent.lower().split()
    return len(tokens) > 1 and tokens[0] in BOOL_QUESTION_WORDS


def make_question(sent: str):
    if not sent.endswith("?"):
        return sent + "?"
    else:
        return sent


def remove_question_mark(sent: str):
    if sent.endswith("?"):
        return sent[:-1].strip()
    else:
        return sent


def remove_punctuation(sent: str):
    new_s = sent.translate(PUNCT_TABLE)
    return new_s


def get_words(sent: str):
    sent = remove_punctuation(sent)
    return sent.lower().split()


def filter_keywords(keywords: list, additional_words: Set[str] = {}):
    tokens = [
        token
        for token in keywords
        if (
            token not in QUESTION_WORDS
            and token not in BOOL_QUESTION_WORDS
            and token not in additional_words
        )
    ]
    return tokens


def validate_text(text: str):
    if text is not None:
        # strip the special characters from debug mode
        # text = text.split(">", 1)[-1] # removed to prevent error when question contains ">"
        text = text.strip()
    else:
        text = ""
    return text


def extract_keywords(question: str, remove_words: Set[str] = {}) -> str:
    # if in debug mode, we must remove special characters
    candidates = []
    candidates += re.findall("[\"'“](.*?)[\"'”]", question)
    # if no quotations are given but question is short we treat it as keywords
    if not candidates:
        candidates = question.split()
    keywords = filter_keywords(candidates, remove_words)
    keywords = keywords if len(keywords) < 5 else []
    return keywords


def is_summary_search(question: str, templates: List[str], header: List[str]):
    if header == [question] and (templates == [] or templates == [""]):
        return True
    return False


def resolve_query_params(templates: List[str], question: str, headers: List[str]):
    if question == "" and (templates == [] or templates == [""]) and len(headers):
        question = headers[0]
    # validate text, remove empty string and None
    question = validate_text(question)
    templates = [validate_text(template) for template in templates]

    # remove empty template
    templates = [template for template in templates if template]

    keywords = []
    if question:
        keywords = extract_keywords(remove_question_mark(question))
        if is_question(question):
            question = make_question(question)
    return templates, question, headers, keywords


def load_settings():
    settings = {
        "DEBUG": bool(os.getenv("DEBUG", False)),
        # model server
        "MODEL_SERVER_URL": os.getenv(
            "MODEL_SERVER_URL",
            "https://services.nlmatics.com",
        ),
        "DPR_MODEL_SERVER_URL": os.getenv(
            "DPR_MODEL_SERVER_URL",
            os.getenv(
                "MODEL_SERVER_URL",
                "https://services.nlmatics.com",
            ),
        ),
        "QATYPE_MODEL_SERVER_URL": os.getenv(
            "QATYPE_MODEL_SERVER_URL",
            os.getenv(
                "MODEL_SERVER_URL",
                "https://services.nlmatics.com",
            ),
        ),
        "QA_MODEL_SERVER_URL": os.getenv(
            "QA_MODEL_SERVER_URL",
            os.getenv(
                "MODEL_SERVER_URL",
                "https://services.nlmatics.com",
            ),
        ),
        "BOOLQ_MODEL_SERVER_URL": os.getenv(
            "BOOLQ_MODEL_SERVER_URL",
            os.getenv(
                "MODEL_SERVER_URL",
                "https://services.nlmatics.com",
            ),
        ),
        # simialrity threshold
        "SIF_THRESHOLD": ensure_float(os.getenv("SIF_THRESHOLD", 0.45)),
        "DPR_THRESHOLD": ensure_float(os.getenv("DPR_THRESHOLD", 0.45)),
        # enable DPR
        "USE_DPR": ensure_bool(os.getenv("USE_DPR", False)),
        # enable QATYPE filter
        "USE_QATYPE": ensure_bool(os.getenv("USE_QATYPE", False)),
        "QATYPE_MODEL": os.getenv("QATYPE_MODEL", "roberta"),
        # QNLI settings
        "QNLI_MODEL": os.getenv("QNLI_MODEL", "roberta"),
        "QNLI_THRESHOLD": ensure_float(os.getenv("QNLI_THRESHOLD", 0.5)),
        # SQUAD settings
        "QA_MODEL": os.getenv("QA_MODEL", "roberta"),
        "QA_TASK": os.getenv("QA_TASK", "roberta-qa"),
        "PHRASE_QA_TASK": os.getenv("PHRASE_QA_TASK", "roberta-phraseqa"),
        "RUN_QNLI": ensure_bool(os.getenv("RUN_QNLI", False)),
        "RUN_CROSS_ENCODER": ensure_bool(os.getenv("RUN_CROSS_ENCODER", False)),
        "SQUAD_THRESHOLD": ensure_float(os.getenv("SQUAD_THRESHOLD", 0.3)),
        # BOOLQ settings
        "BOOLQ_MODEL": os.getenv("BOOLQ_MODEL", "roberta"),
        "BOOLQ_THRESHOLD": ensure_float(os.getenv("BOOLQ_THRESHOLD", 0.5)),
        # Search engine
        "ES_URL": os.getenv("ES_URL"),
        "ES_SECRET": os.getenv("ES_SECRET"),
        # others
        "USE_FILE_SCORE": ensure_bool(os.getenv("USE_FILE_SCORE", True)),
        "SUMMARIZATION_MODEL": os.getenv(
            "SUMMARIZATION_MODEL",
            "openai",
        ),
        "SUMMARIZATION_MODEL_SERVER_URL": os.getenv(
            "SUMMARIZATION_MODEL_SERVER_URL",
            os.getenv(
                "MODEL_SERVER_URL",
                "https://services.nlmatics.com",
            ),
        ),
        "SUMMARIZATION_TASK": os.getenv(
            "SUMMARIZATION_TASK",
            "qa_sum",
        ),
        "QUERY_DECOMPOSITION_MODEL_SERVER_URL": os.getenv(
            "QUERY_DECOMPOSITION_MODEL_SERVER_URL",
            os.getenv(
                "MODEL_SERVER_URL",
                "https://services.nlmatics.com",
            ),
        ),
    }

    for k, v in settings.items():
        if not isinstance(v, str):
            continue

        v = v.lower()
        if v in {"true", "false"}:
            settings[k] = v == "true"
            continue

        try:
            settings[k] = int(v)
            continue
        except ValueError:
            pass

        try:
            settings[k] = float(v)
            continue
        except ValueError:
            pass
    return settings


def apply_private_dictionary(
    text,
    dictionary,
    change_to_key=False,
):
    # modify query with synonyms
    orig_text = text
    texts = [orig_text] if not change_to_key else []
    applied_synonyms = {}

    for key, synonyms in dictionary.items():
        # *IMPORTANT*
        # the synonyms is sorted before saving into db.
        # Since most of the synonyms are suffix, sorting will prevents edge cases.
        # e.g.
        # When replacing "AAPL" and "AAPL.O" to "Apple",
        # we must replace "AAPL.O" to "Apple", then replace "AAPL" to "Apple".
        # Otherwise we will have sentence like "Apple.O"
        add_synonym = False
        if not change_to_key:
            for synonym in sorted(synonyms + [key], reverse=True):
                if len(synonym.strip()) > 0:
                    # case insensitive replacement for synonyms
                    # Check for words that are not immediately followed by a letter or number.
                    synonym_regex = re.compile(
                        rf"(?<![\w\d]){synonym}(?![\w\d])",
                        re.IGNORECASE,
                    )
                    if synonym_regex.search(orig_text) is not None:
                        for s in sorted(synonyms + [key], reverse=True):
                            text = synonym_regex.sub(s, orig_text)
                            if text not in texts:
                                texts.append(text)
                                add_synonym = True
                    if add_synonym:
                        break
        else:
            for synonym in sorted(synonyms + [key], reverse=True):
                if len(synonym.strip()) > 0:
                    synonym_regex = re.compile(
                        rf"(?<![\w\d]){synonym}(?![\w\d])",
                        re.IGNORECASE,
                    )
                    text = synonym_regex.sub(key, text)

        if add_synonym:
            applied_synonyms[key] = synonyms
    # If we have to replace with Key, add only the last completely modified text.
    if change_to_key and text not in texts:
        texts.append(text)

    return texts if not change_to_key else texts[0], applied_synonyms


def expand_cross_references(
    file_idx: str,
    context: str,
    discard_blocks: List[int] = None,
    max_tokens_limit=OPENAI_MAX_NUM_TOKENS,
):
    discard_blocks = discard_blocks or []
    ret_str = ""
    ref_definitions = nosql_db.get_document_reference_definitions_by_id(file_idx) or {}
    if ref_definitions:
        automaton = ahocorasick.Automaton()
        # Create the automaton
        for idx, (key, value) in enumerate(ref_definitions.items()):
            # value contains block_idx and block_text
            automaton.add_word(key, (idx, (key, value)))
        automaton.make_automaton()

        matched_refs = []
        match_keys = {}
        matched_blocks = []
        for end_index, (insert_order, original_value) in automaton.iter(context):
            match_key = original_value[0]
            start_index = end_index - len(match_key) + 1
            start_checked = True
            if start_index - 1 > 0:
                start_checked = check_char_is_word_boundary(context[start_index - 1])

            if start_checked and (
                end_index == len(context) - 1
                or check_char_is_word_boundary(context[end_index + 1])
            ):
                found_longer_match = False
                for matched_ref in matched_refs:
                    if start_index >= matched_ref[0] and end_index <= matched_ref[1]:
                        found_longer_match = True
                        break
                if not found_longer_match:
                    if match_key not in match_keys:
                        matched_refs.append(
                            (start_index, end_index, (insert_order, original_value)),
                        )
                        for item in original_value[1]:
                            # Do not add block_idx if already added.
                            block_idx = item.get("block_idx", -1)
                            if (
                                block_idx != -1
                                and block_idx not in matched_blocks
                                and block_idx not in discard_blocks
                            ):
                                if (
                                    nosql_db.escape_mongo_data(match_key)
                                    not in match_keys
                                ):
                                    match_keys[
                                        nosql_db.escape_mongo_data(match_key)
                                    ] = []
                                match_keys[
                                    nosql_db.escape_mongo_data(match_key)
                                ].append(item)
                                matched_blocks.append(block_idx)

        for _, items in match_keys.items():
            for item in items:
                block_text = item.get("block_text", "")
                if block_text and block_text not in ret_str:
                    if not ret_str:
                        ret_str = block_text
                    else:
                        ret_str += "\n" + block_text
        if ret_str:
            # len_words = len(context.split(" ")) + len(ret_str.split(" "))
            tokens = openai_tokenizer.encode(
                context + " " + ret_str,
                disallowed_special=(),
            )
            len_words = len(tokens)
            if len_words > max_tokens_limit:
                # Don't send anything for now.
                ret_str = ""
    return ret_str


def normalize_quotes(text: str):
    return quotation_pattern.sub('"', text)


def remove_quotes(text: str):
    return quotation_pattern.sub("", text)


def check_char_is_word_boundary(c):
    if c.isalnum():
        return False
    if c in ["-", "_"]:
        return False
    return True
