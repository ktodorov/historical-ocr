from typing import List

from entities.skip_gram_entry import SkipGramEntry

class CBOWCorpus:
    def __init__(self, text_lines: List[List[int]], window_size: int = 2):
        self._window_size = window_size
        self._entries = self._generate_skip_gram_entries(text_lines)

    def cut_data(self, corpus_size: int):
        self._entries = self._entries[:corpus_size]

    def _generate_skip_gram_entries(self, text_lines: List[List[int]]):
        result = []

        for text_line in text_lines:
            for target_idx in range(len(text_line)):
                window_start = max(0, target_idx-self._window_size)
                window_end = min(len(text_line), target_idx+self._window_size+1)
                context_tokens = text_line[window_start:target_idx] + text_line[target_idx+1:window_end]
                result.append(SkipGramEntry(
                    target_token=text_line[target_idx],
                    context_tokens=context_tokens))

        return result

    def get_entry(self, idx: int) -> SkipGramEntry:
        return self._entries[idx]

    @property
    def length(self) -> int:
        return len(self._entries)
