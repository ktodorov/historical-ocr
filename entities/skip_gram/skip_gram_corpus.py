from typing import List

from entities.skip_gram.skip_gram_entry import SkipGramEntry

class SkipGramCorpus:
    def __init__(self, text_lines: List[List[int]]):
        self._window_size = 1
        self._entries = self._generate_skip_gram_entries(text_lines)

    def cut_data(self, corpus_size: int):
        self._entries = self._entries[:corpus_size]

    def _generate_skip_gram_entries(self, text_lines: List[List[int]]):
        result = []

        for text_line in text_lines:
            for target_idx in range(len(text_line)):

                # if target is not the first token, we add new entry where context token is the previous one
                if target_idx > 0:
                    result.append(
                        SkipGramEntry(
                            target_token=text_line[target_idx],
                            context_token=text_line[target_idx-1]))

                # if target is not the last token, we add new entry where context token is the next one
                if target_idx < len(text_line) - 1:
                    result.append(
                        SkipGramEntry(
                            target_token=text_line[target_idx],
                            context_token=text_line[target_idx+1]))

        return result

    def get_entry(self, idx: int) -> SkipGramEntry:
        return self._entries[idx]

    @property
    def length(self) -> int:
        return len(self._entries)
