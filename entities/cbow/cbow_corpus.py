from typing import List

from entities.cbow.cbow_entry import CBOWEntry

class CBOWCorpus:
    def __init__(self, text_lines: List[List[int]], window_size: int = 2):
        self._window_size = window_size
        self._entries = self._generate_skip_gram_entries(text_lines)
        self._total_documents = len(text_lines)

    def cut_data(self, corpus_size: int):
        self._entries = self._entries[:corpus_size]

    def _generate_skip_gram_entries(self, text_lines: List[List[int]]) -> List[CBOWEntry]:
        result = []

        for i, text_line in enumerate(text_lines):
            for target_idx in range(len(text_line)):
                window_start = max(0, target_idx-self._window_size)
                window_end = min(len(text_line), target_idx+self._window_size+1)
                context_tokens = text_line[window_start:target_idx] + text_line[target_idx+1:window_end]
                result.append(CBOWEntry(
                    document_idx=i,
                    target_token=text_line[target_idx],
                    context_tokens=context_tokens))

        return result

    def get_entry(self, idx: int) -> CBOWEntry:
        return self._entries[idx]

    def get_total_documents(self) -> int:
        return self._total_documents

    def get_document_indices(self, document_idx: int) -> List[int]:
        document_indices = len([i for i, entry in enumerate(self._entries) if entry.document_index == document_idx])
        return document_indices

    @property
    def length(self) -> int:
        return len(self._entries)
