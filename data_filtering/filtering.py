import gzip
import json
from tqdm import tqdm
import ipdb
import regex as re
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import copy
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import Manager
from multiprocessing.synchronize import Lock
import fire

# URLs (http/https/ftp/www, tolerates Unicode domains; excludes closing brackets/quotes)
_P_URL = re.compile(
    r"""(?xi)
    \b(
        (?:https?|ftp)://[^\s<>"'））」》\]\)}]+
        |www\.[^\s<>"'））」》\]\)}]+
    )
    """
)

# Emails (basic but robust; avoids most false positives)
_P_EMAIL = re.compile(
    r"""(?xi)
    \b[A-Z0-9._%+-]+
    @
    [A-Z0-9.-]+\.[A-Z]{2,}\b
    """
)

def get_all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

@dataclass
class ViolationRecord:
    name: str
    count: int = 0
    score: float = -1.0
    doc_threshold: float = 0.0
    stamp_label: Optional[str] = None
    violated: bool = False

@dataclass
class DocRecord:
    id: int
    original_sentence_count: int
    dropped: bool
    violation_records: dict[str, ViolationRecord] = field(default_factory=dict)
    # These fields are now optional and default to None to save memory
    original_doc_json_line: Optional[str] = None
    stamped_doc: Optional[str] = None

    def print_summary(self):
        print("---=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        print(f"Doc ID: {self.id}")
        print(f"Original Sentence Count: {self.original_sentence_count}")
        print(f"Violations and scores: {{}}".format(", ".join(f"{name}: {vr.count} ({vr.score:.4f})" for name, vr in self.violation_records.items())))
        print(f"Dropped: {self.dropped}")
        print("Doc Content:")
        if self.stamped_doc:
            print(self.stamped_doc)
            return self.stamped_doc
        else:
            print("[Text not stored in record to save memory]")
        print("-----")

class SentenceFilter:
    """
    Return True if a sentence violates the filter.
    """
    name: str

    def __init__(self, name: str):
        self.name = name

    def violated(self, sentence: str) -> bool:
        raise NotImplementedError

class DocumentFilter:
    """
    Return True if a document violates the filter.
    """
    name: str

    def __init__(self, name: str):
        self.name = name

    def violated(self, document: str) -> bool:
        raise NotImplementedError

class LambdaSentenceFilter(SentenceFilter):
    """Convenience wrapper to register quick filters."""
    def __init__(self, name: str, fn: Callable[[str], bool]):
        super().__init__(name)
        self._fn = fn

    def violated(self, sentence: str) -> bool:
        return bool(self._fn(sentence))

# Concrete filters equivalent to your original logic
class ContainsAnyCharFromSetFilter(SentenceFilter):
    def __init__(self, name: str, chars_set: set):
        super().__init__(name)
        self.chars_set = chars_set

    def violated(self, sentence: str) -> bool:
        return any(c in self.chars_set for c in sentence)

test_charset_filter = ContainsAnyCharFromSetFilter("test_charset", {'你', '好', '啊'})
assert test_charset_filter.violated("你好") == True
assert test_charset_filter.violated("hello") == False
assert test_charset_filter.violated("hello 你") == True
assert test_charset_filter.violated("") == False
assert test_charset_filter.violated("鉛筆") == False

class HangulFilter(SentenceFilter):
    def __init__(self, name: str):
        super().__init__(name)
        # Compile a regex that matches any Hangul character (all relevant blocks)
        self.hangul_re = re.compile(
            r'[\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\uD7B0-\uD7FF\uFFA0-\uFFDC]'
        )

    def violated(self, sentence: str) -> bool:
        # Use regex search to check for any Hangul character
        return bool(self.hangul_re.search(sentence))

test_hangul_filter = HangulFilter("test_hangul")
assert test_hangul_filter.violated("hello") == False
assert test_hangul_filter.violated("안녕하세요") == True
assert test_hangul_filter.violated("hello 안녕하세요") == True
assert test_hangul_filter.violated("") == False
assert test_hangul_filter.violated("鉛筆") == False


class LatinDominantFilter(SentenceFilter):
    def __init__(self, name: str, min_len: int = 10, ratio: float = 0.5, max_len: int=100, max_consecutive_latin_words: int = 4):
        super().__init__(name)
        self.min_len = min_len # if a sentence only has <min_len Latin chars, ignore it
        self.ratio = ratio # ratio of Latin chars in the sentence to consider it violated
        self.max_len = max_len # if a sentence has more than max_len latin chars, violate it directly
        self.max_consecutive_latin_words = max_consecutive_latin_words
        self.latin_only_re = re.compile(r'[^\p{Latin}\p{M}]+')
        self.latin_any_re = re.compile(r'[\p{Latin}\p{M}]')

    def _max_consecutive_latin_words(self, sentence: str) -> int:
        """
        Count the longest streak of consecutive 'Latin words'.
        A 'Latin word' here is any token that contains at least one Latin character.
        """
        max_run = 0
        current_run = 0

        for token in sentence.split():
            if self.latin_any_re.search(token):
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0

        return max_run


    def violated(self, sentence: str) -> bool:
        if not sentence:
            return False
        
        if self._max_consecutive_latin_words(sentence) > self.max_consecutive_latin_words:
            return True

        latin_only = self.latin_only_re.sub("", sentence)
        if len(latin_only) < self.min_len:
            return False
        if len(latin_only) > self.max_len:
            return True
        n_total = len(sentence.replace(" ", ""))
        return (len(latin_only) / max(1, n_total)) > self.ratio


test_latin_dominant_filter = LatinDominantFilter("test_latin_dominant", min_len=10, ratio=0.5, max_len=30)
assert test_latin_dominant_filter.violated("hello") == False
assert test_latin_dominant_filter.violated("hello world lol lol!") == True
assert test_latin_dominant_filter.violated("hello 안녕하세요") == False
assert test_latin_dominant_filter.violated("hello hello hello hello 안녕하세요") == True
assert test_latin_dominant_filter.violated("hello hello hello hello 안녕하세요 안녕하세요 안녕하세요 안녕하세요 안녕하세요 안녕하세요 안녕하세요") == False
assert test_latin_dominant_filter.violated("안녕하세요안녕하세요안녕하세요") == False
assert test_latin_dominant_filter.violated("") == False
assert test_latin_dominant_filter.violated("hello hello hello hello hello hellooo  안녕하세요 안녕하세요 안녕하세요 안녕하세요 안녕하세요 안녕하세요요요요요") == True

assert test_latin_dominant_filter._max_consecutive_latin_words(" ") == 0
assert test_latin_dominant_filter._max_consecutive_latin_words("I") == 1
assert test_latin_dominant_filter._max_consecutive_latin_words("I am") == 2
assert test_latin_dominant_filter._max_consecutive_latin_words("I am a cat.") == 4
assert test_latin_dominant_filter._max_consecutive_latin_words("引っかからない") == 0
assert test_latin_dominant_filter._max_consecutive_latin_words("引っかかってしまうno") == 1
assert test_latin_dominant_filter._max_consecutive_latin_words("He is coming tomorrow. だが He is not. ") == 4
assert test_latin_dominant_filter._max_consecutive_latin_words("英語で言うと「I love cats.」だ。") == 3
assert test_latin_dominant_filter._max_consecutive_latin_words("「He is a cat」と「I am not a cat」どっちが正しい？") == 8
assert test_latin_dominant_filter._max_consecutive_latin_words("住所は「9A CT House, 2nd floor, Providence, Mahe, Seychelles」です。") == 8
assert test_latin_dominant_filter._max_consecutive_latin_words("Sony社とApple社の製品を使う。") == 1
assert test_latin_dominant_filter._max_consecutive_latin_words("Apple社は昨日、iPhone 88とApple Footballの発売をアナウンスしました。") == 3
assert test_latin_dominant_filter._max_consecutive_latin_words("また、流れる発音は「Speech Recognition API」を利用したものです(Chromeをご利用ください)。") == 3

class UnicodeDocumentFilter(DocumentFilter):
    """
    Returns the percentage ratio of non-specified Unicode characters
    Uses a compiled regex to match allowed characters.
    """
    def __init__(self, name: str, allowed_unicode_ranges_re: re.Pattern):
        super().__init__(name)
        self.allowed_unicode_ranges_re = allowed_unicode_ranges_re

    def violated(self, document: str) -> tuple[int, float]:
        if not document:
            return 0, 0.0  # Empty documents: 0 matched chars, ratio 0.0
        
        cleaned_document = document.replace(" ", "").replace("\n", "")
        total_chars = len(cleaned_document)

        if total_chars == 0:
            return 0, 0.0
        
        matches = self.allowed_unicode_ranges_re.findall(cleaned_document)
        matched_chars = sum(len(match) for match in matches)
        violation_chars = total_chars - matched_chars
        violation_score = 1 - (matched_chars / total_chars)
        return violation_chars, violation_score
    
    def flagged_characters(self, document: str) -> list[str]:
        return [ch for ch in document.replace(" ", "").replace("\n", "") if not self.allowed_unicode_ranges_re.match(ch)]

_RANGE_PATTERN = (
    # --- Japanese scripts ---
    r"\u3040-\u309F"          # Hiragana
    r"\u30A0-\u30FF"          # Katakana
    r"\u31F0-\u31FF"          # Katakana Phonetic Extensions
    # r"\u3400-\u4DBF"          # CJK Unified Ideographs Extension A
    r"\u4E00-\u9FFF"          # CJK Unified Ideographs (basic Kanji)
    r"\u2E80-\u2EFF"          # CJK Radicals Supplement
    # r"\u2F00-\u2FDF"          # Kangxi Radicals

    # CJK punctuation / compatibility
    r"\u3000-\u303F"          # CJK Symbols and Punctuation
    r"\u3200-\u32FF"          # Enclosed CJK Letters and Months
    r"\u3300-\u33FF"          # CJK Compatibility
    r"\uF900-\uFAFF"          # CJK Compatibility Ideographs
    r"\uFE30-\uFE4F"          # CJK Compatibility Forms

    # Halfwidth / fullwidth Japanese
    r"\uFF01-\uFF5E"          # Fullwidth ASCII (except space)
    r"\uFF61-\uFF9F"          # Halfwidth Kana + punctuation (FF61–FF65, FF66–FF9F)

    # some outliers
    r"\uFFE5"                 # FULLWIDTH YEN SIGN ￥
    r"\uFFE6"                 # FULLWIDTH WON SIGN ￦
    r"\u00A1-\u00BF"          # Latin-1 supplement subset: punctuation & symbols, contains ¢ £ ¤ ¥ © ®
    r"\u00D7"                 # MULTIPLICATION SIGN (×)
    r"\u00F7"                 # DIVISION SIGN (÷)

    # --- Basic Latin (ASCII) ---
    r"\u0020"                 # SPACE (if you want to allow spaces)
    r"\u0021-\u002F"          # !"#$%&'()*+,-./
    r"\u0030-\u0039"          # 0–9
    r"\u003A-\u0040"          # :;<=>?@
    r"\u0041-\u005A"          # A–Z
    r"\u005B-\u0060"          # [\]^_`
    r"\u0061-\u007A"          # a–z
    r"\u007B-\u007E"          # {|}~

    # --- General punctuation & common symbol blocks ---
    r"\u2000-\u206F"          # General Punctuation (includes ZWJ etc., see notes)
    r"\u20A0-\u20CF"          # Currency Symbols
    r"\u2100-\u214F"          # Letterlike Symbols
    r"\u2150-\u218F"          # Number Forms
    r"\u2190-\u21FF"          # Arrows
    r"\u2200-\u22FF"          # Mathematical Operators
    r"\u2300-\u23FF"          # Misc Technical
    r"\u2460-\u24FF"          # Enclosed Alphanumerics
    r"\u2500-\u257F"          # Box Drawing
    r"\u25A0-\u25FF"          # Geometric Shapes
    r"\u2600-\u26FF"          # Miscellaneous Symbols
    r"\u2700-\u27BF"          # Dingbats

    # --- Emoji blocks ---
    r"\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
    r"\U0001F600-\U0001F64F"  # Emoticons
    r"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
    r"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    r"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A

    # Emoji flags & variation selectors
    r"\U0001F1E6-\U0001F1FF"  # Regional Indicator Symbols (country flags)
    r"\uFE00-\uFE0F"          # Variation Selectors
    r"\U000E0100-\U000E01EF"  # Variation Selectors Supplement

    # --- OPTIONAL: historical kana (uncomment if you want them) ---
    # r"\U0001B000-\U0001B0FF"  # Kana Supplement
    # r"\U0001B100-\U0001B12F"  # Kana Extended-A
    # r"\U0001B130-\U0001B16F"  # Small Kana Extension
    # r"\U0001AFF0-\U0001AFFF"  # Kana Extended-B

    # --- OPTIONAL & RISKY: Tags block (emoji subdivision flags, but invisible) ---
    # r"\U000E0000-\U000E007F"  # Tags
)


_ALLOWED_RE = re.compile(f"[{_RANGE_PATTERN}]+")

test_japanese_document_filter = UnicodeDocumentFilter(
    "test_japanese",
    allowed_unicode_ranges_re=_ALLOWED_RE
)

assert abs(test_japanese_document_filter.violated("こんにちは、世界！")[1] - 0) < 1e-3  # All Japanese
assert abs(test_japanese_document_filter.violated("Hello世界")[1] - 0) < 1e-3  # Mixed English and Japanese
assert abs(test_japanese_document_filter.violated("富含氮磷钾")[1] - 0) < 1e-3  # CJK characters
assert abs(test_japanese_document_filter.violated("åabc")[1] - (1/4)) < 1e-3  # non-ascii Latin

assert abs(test_japanese_document_filter.violated("\n")[1] - 0) < 1e-3
assert abs(test_japanese_document_filter.violated(" ")[1] - 0) < 1e-3
assert abs(test_japanese_document_filter.violated("　")[1] - 0) < 1e-3
assert abs(test_japanese_document_filter.violated("🇯🇵 🇰🇷 🇮🇹")[1] - 0) < 1e-3
assert abs(test_japanese_document_filter.violated("😂 😊 🚢 🍦 🍎")[1] - 0) < 1e-3
assert abs(test_japanese_document_filter.violated("■━⇒【】○")[1] - 0) < 1e-3


class DocFilteringStatus:
    all_doc_records: Dict[str, List[DocRecord]]
    def __init__(self, file_paths: List[str], process_num: int | list[int], memory_efficient: bool = False):
        self.file_paths = file_paths
        self.memory_efficient = memory_efficient
        if isinstance(process_num, int):
            self.process_num = {file_path: process_num for file_path in file_paths}
        elif isinstance(process_num, list):
            assert len(process_num) == len(file_paths)
            self.process_num = {file_path: num for file_path, num in zip(file_paths, process_num)}
    
    def get_process_num_for_file(self, file_path: str) -> int:
        return self.process_num.get(file_path, 0)
    
    def append_doc_records(self, file_path: str, doc_records: List[DocRecord]):
        if not hasattr(self, 'all_doc_records'):
            self.all_doc_records = {}
        self.all_doc_records[file_path] = doc_records
    
    def append_single_doc_record(self, file_path: str, doc_record: DocRecord):
        if not hasattr(self, 'all_doc_records'):
            self.all_doc_records = {}
        if file_path not in self.all_doc_records:
            self.all_doc_records[file_path] = []
        self.all_doc_records[file_path].append(doc_record)
    
    def get_all_doc_records_in_a_list(self) -> List[DocRecord]:
        if not hasattr(self, 'all_doc_records'):
            return []
        all_doc_records_list = []
        for records in self.all_doc_records.values():
            all_doc_records_list.extend(records)
        return all_doc_records_list
    
    def get_all_doc_records(self) -> Dict[str, List[DocRecord]]:
        if not hasattr(self, 'all_doc_records'):
            return {}
        return self.all_doc_records
    
    def get_doc_records_for_file(self, file_path: str) -> List[DocRecord]:
        if not hasattr(self, 'all_doc_records'):
            return []
        return self.all_doc_records.get(file_path, [])
    
    def sort_each_doc_records_by_score_sum(self):
        if not hasattr(self, 'all_doc_records'):
            return
        for file_path, records in self.all_doc_records.items():
            self.all_doc_records[file_path] = sorted(records, key=lambda r: sum(vr.score for vr in r.violation_records.values()), reverse=True)
    
    def sort_each_doc_records_focusing_on_specific_violation(self, violation_name: str):
        if not hasattr(self, 'all_doc_records'):
            return
        for file_path, records in self.all_doc_records.items():
            # sum scores. but for the specific violation, multiply that score by 100 before summing
            self.all_doc_records[file_path] = sorted(records, key=lambda r: sum((vr.score * 100 if vr.name == violation_name else vr.score) for vr in r.violation_records.values()), reverse=True)
    
    def _print_stats_for_doc_records(self, doc_records: List[DocRecord]):
        total = len(doc_records)
        dropped = sum(1 for r in doc_records if r.dropped)
        print(f"Dropped documents: {dropped} ({dropped/total*100:.2f}%) out of {total}")
        if total == 0:
            return
        for vr_name in doc_records[0].violation_records.keys():
            violated = sum(1 for r in doc_records if r.violation_records[vr_name].violated)
            print(f"Documents violated by {vr_name}: {violated} ({violated/total*100:.2f}%)")
        print("-----")

    def print_final_stats(self):
        # first, print stats for each file
        if not hasattr(self, 'all_doc_records'):
            raise ValueError("No doc records found")
        for file_path, records in self.all_doc_records.items():
            print(f"Stats for file: {file_path}")
            self._print_stats_for_doc_records(records)
        # then, print overall stats
        all_records = self.get_all_doc_records_in_a_list()
        print("Overall stats:")
        self._print_stats_for_doc_records(all_records)


# NEW: A simple, lightweight summary object for each worker to return
@dataclass
class FileStats:
    file_path: str
    total_docs_processed: int = 0
    total_docs_dropped: int = 0
    violation_counts: Dict[str, int] = field(default_factory=dict)
    def update_by_doc_record(self, doc_record: DocRecord):
        self.total_docs_processed += 1
        if doc_record.dropped:
            self.total_docs_dropped += 1
        for vr_name, vr in doc_record.violation_records.items():
            if vr.violated:
                self.violation_counts[vr_name] = self.violation_counts.get(vr_name, 0) + 1
            # ADDED: track those dropped by sentence filters and not by doc filters
            if not vr_name.endswith("_doc"):
                if vr.violated:
                    flagged_by_doc_filters = any(
                        vr_doc.violated for vr_doc in doc_record.violation_records.values() if vr_doc.name.endswith("_doc")
                    )
                    if not flagged_by_doc_filters:
                        self.violation_counts[f"{vr_name}_by_only_sentence"] = self.violation_counts.get(f"{vr_name}_by_only_sentence", 0) + 1

    def print_stats(self):
        total = self.total_docs_processed
        dropped = self.total_docs_dropped
        print(f"Stats for file: {self.file_path}")
        print(f"Dropped documents: {dropped} ({dropped/total*100:.2f}%) out of {total}")
        if total == 0:
            return
        for vr_name, count in self.violation_counts.items():
            print(f"Documents violated by {vr_name}: {count} ({count/total*100:.2f}%)")
        print("-----")


def process_single_file(
    file_path: str,
    filters: List,
    violation_records_template: Dict,
    process_num: Optional[int],
    output_dir: str,
    memory_efficient: bool,
    lock: Lock
) -> Tuple[str, List[DocRecord]]:
    """
    Processes a single file:
    1. Reads documents line by line.
    new. if a line violates a sentence filter, remove it. 
    2. If a document passes filters, writes it immediately to a new .gz file.
    3. Conditionally stores text in the returned DocRecord objects for stats/debugging.
    """

    worker_id = os.getpid()
    base_name = os.path.basename(file_path)
    print(f"Worker {worker_id} started: {base_name}, process_num={process_num}, memory_efficient={memory_efficient}")
    doc_records_for_file = []

    STAMP_LABELS = [vr.stamp_label for vr in violation_records_template.values()]

    # Prepare output path
    output_path = os.path.join(output_dir, f"filtered_{base_name}")

    # --- NEW: Setup for progress reporting ---
    report_interval = 0
    if process_num is not None:
        # Report every 10% if process_num is specified
        if process_num > 10:
            report_interval = process_num // 10
    else:
        # Report every 10,000 docs if processing the whole file
        report_interval = 10000

    with gzip.open(file_path, 'rt', encoding='utf-8') as f_in, \
         gzip.open(output_path, 'wt', encoding='utf-8') as f_out:
        stats = FileStats(file_path=file_path, violation_counts={name: 0 for name in violation_records_template.keys()})
        for i, line in enumerate(f_in):
            if process_num is not None and i >= process_num:
                break

            if report_interval > 0 and (i + 1) % report_interval == 0:
                with lock: # Use the lock to prevent mixed-up print statements
                    if process_num:
                        progress_percent = (i + 1) / process_num * 100
                        print(f"-> Worker {worker_id} on {base_name}: Processed {i + 1}/{process_num} lines ({progress_percent:.0f}%)")
                    else:
                        print(f"-> Worker {worker_id} on {base_name}: Processed {i + 1} lines")

            # The full text of the line is only needed for processing, not necessarily for storing
            text = json.loads(line)["text"]
            
            text = _P_URL.sub("", text)
            text = _P_EMAIL.sub("", text)

            # NEW: dont do strip to keep the structure. this is for line removal.
            # sentences = [s.strip() for s in text.split("\n") if s.strip()]
            sentences = [s for s in text.split("\n")]

            # NEW: only count non-empty lines. 
            original_doc_sentence_count = len([s for s in sentences if s != ""])

            if original_doc_sentence_count == 0:
                continue

            doc_record = DocRecord(
                id=i,
                original_sentence_count=original_doc_sentence_count,
                violation_records=copy.deepcopy(violation_records_template),
                dropped=False # Will be determined shortly
            )
            # Conditionally store original line
            if not memory_efficient:
                doc_record.original_doc_json_line = line

            # doc-level filters
            for f_filter in filters:
                if isinstance(f_filter, DocumentFilter):
                    violation_char_count, violation_score = f_filter.violated(text)
                    doc_record.violation_records[f_filter.name].count = violation_char_count
                    doc_record.violation_records[f_filter.name].score = violation_score

            # sentence-level filters
            stamped_sentences = []
            for sentence in sentences:
                stamps = []
                for f_filter in filters:
                    if isinstance(f_filter, SentenceFilter) and f_filter.violated(sentence):
                        doc_record.violation_records[f_filter.name].count += 1
                        stamps.append(doc_record.violation_records[f_filter.name].stamp_label)
                if stamps:
                    sentence += " " + " ".join(stamps)
                stamped_sentences.append(sentence)

            # Conditionally store the stamped document text
            if not memory_efficient:
                doc_record.stamped_doc = "\n".join(stamped_sentences)

            # Calculate scores for filters
            for vr in doc_record.violation_records.values():
                # if score != -1.0, it means it is a doc-level filter, and score has already been assigned. 
                if vr.score == -1.0:
                    vr.score = vr.count / original_doc_sentence_count
                vr.violated = vr.score > vr.doc_threshold

            # Determine if the document should be dropped
            doc_record.dropped = any(vr.violated for vr in doc_record.violation_records.values())

            if memory_efficient:
                stats.update_by_doc_record(doc_record)
            else:
                doc_records_for_file.append(doc_record)

            # **Immediate Writing**: If the doc passed, write its original JSON line to the output file
            if not doc_record.dropped:
                # NEW: before writing, remove stamped sentences that violated sentence filters
                # reconstruct text without sentences that violated sentence filters
                sentences_to_keep = []
                for sentence in stamped_sentences:
                    if not any(stamp in sentence for stamp in STAMP_LABELS):
                        sentences_to_keep.append(sentence)

                text_to_write = "\n".join(sentences_to_keep)

                original_line_json = json.loads(line)
                original_line_json["text"] = text_to_write

                line_to_write = json.dumps(original_line_json, ensure_ascii=False) + "\n"

                f_out.write(line_to_write)
            
            if memory_efficient:
                del doc_record
                del sentences
                del stamped_sentences
                del text
                del line

    with lock:
        print(f"Worker {worker_id} FINISHED: {base_name}. Filtered output saved to {output_path}")
    return (file_path, doc_records_for_file, stats)



def main(
    data_path: str,
    output_dir: str,
    unihan_chars_path: str = "unihan_bmp_no_japanese.txt",
    num_workers: int = os.cpu_count(),
    process_num_per_file: Optional[int] = None,
    memory_efficient: bool = False
):
    """
    Filters large text datasets in parallel.

    Args:
        data_path: Path to the input directory containing .gz files.
        output_dir: Path to the directory where filtered .gz files will be saved.
        unihan_chars_path: Path to the Unihan characters file for the Chinese filter.
        num_workers: Number of worker processes to use. Defaults to the number of CPU cores.
        process_num_per_file: Number of lines to process from each file. If None, processes all lines.
        store_text_in_records: If True, stores full text in records for debugging. Uses more memory.
    """
    start_time = time.time()
    
    # --- CONFIGURATION & SETUP ---
    print("--- Starting Data Filtering Process ---")
    print(f"Input data path: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Number of workers: {num_workers}")
    print(f"Process num per file: {'All' if process_num_per_file is None else process_num_per_file}")
    print(f"Memory efficient: {memory_efficient}")
    print("-" * 20)

    with open(unihan_chars_path, "r", encoding="utf-8") as f:
        unihan_simplified_chars = set(line.split("\t")[0] for line in f if line.strip())
    print(f"Read {len(unihan_simplified_chars)} Unihan simplified chars.")

    filters = [
        ContainsAnyCharFromSetFilter("zh_char", unihan_simplified_chars),
        # HangulFilter("ko_char"), # the Whitelist filter will handle Korean hanguls now. 
        LatinDominantFilter("latin_dominant", min_len=8, ratio=0.40, max_len=20, max_consecutive_latin_words=4),
        UnicodeDocumentFilter("japanese_doc", allowed_unicode_ranges_re=_ALLOWED_RE)
    ]

    violation_records = {
        "zh_char": ViolationRecord(name="zh_char", doc_threshold=0.001, stamp_label="[zh_char]"), 
        # "ko_char": ViolationRecord(name="ko_char", doc_threshold=0.001, stamp_label="[ko_char]"), 
        "latin_dominant": ViolationRecord(name="latin_dominant", doc_threshold=0.05, stamp_label="[latin_dominant]"),
        "japanese_doc": ViolationRecord(name="japanese_doc", doc_threshold=0.001, stamp_label="[japanese_doc]")
    }

    file_paths = get_all_file_paths(data_path)
    if not file_paths:
        print(f"Error: No files found in directory: {data_path}")
        return

    # --- EXECUTION ---
    os.makedirs(output_dir, exist_ok=True)
    doc_filtering_status = DocFilteringStatus(file_paths, process_num_per_file, memory_efficient)
    
    manager = Manager()
    print_lock = manager.Lock()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        worker_func = partial(
            process_single_file,
            filters=filters,
            violation_records_template=violation_records,
            process_num=process_num_per_file,
            output_dir=output_dir,
            memory_efficient=memory_efficient,
            lock=print_lock
        )
        
        results = list(tqdm(executor.map(worker_func, file_paths), total=len(file_paths), desc="Processing files"))

    if memory_efficient:
        # first, print individual stats
        for file_path, _, stats in results:
            stats.print_stats()
        # then, aggregate overall stats
        all_stats = list(stats for _, _, stats in results)
        overall_stats = FileStats(file_path="Overall")
        overall_stats.total_docs_processed = sum(s.total_docs_processed for s in all_stats)
        overall_stats.total_docs_dropped = sum(s.total_docs_dropped for s in all_stats)
        for s in all_stats:
            for vr_name, count in s.violation_counts.items():
                overall_stats.violation_counts[vr_name] = overall_stats.violation_counts.get(vr_name, 0) + count
        print("\n--- Overall Stats ---")
        overall_stats.print_stats()

        
    else:
        for file_path, doc_records, _ in results:
            doc_filtering_status.append_doc_records(file_path, doc_records)

        print("\n--- All files processed. Aggregating and printing stats. ---\n")
        doc_filtering_status.print_final_stats()

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    fire.Fire(main)

