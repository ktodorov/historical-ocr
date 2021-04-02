from typing import Dict, List

from entities.cbow.cbow_entry import CBOWEntry

class CBOWCorpus:
    def __init__(self, text_lines: List[List[int]], window_size: int = 2):
        self._window_size = window_size
        self._entries = self._generate_cbow_entries(text_lines)
        self._total_documents = len(set([x.document_index for x in self._entries]))

    def cut_data(self, corpus_size: int):
        self._entries = self._entries[:corpus_size]
        self._total_documents = len(set([x.document_index for x in self._entries]))

    def _generate_cbow_entries(self, text_lines: List[List[int]]) -> List[CBOWEntry]:
        result = []

        for i, text_line in enumerate(text_lines):
            if len(text_line) <= 1:
                continue

            for target_idx in range(len(text_line)):
                window_start = max(0, target_idx-self._window_size)
                window_end = min(len(text_line), target_idx+self._window_size+1)
                context_tokens = text_line[window_start:target_idx] + text_line[target_idx+1:window_end]
                result.append(CBOWEntry(
                    document_idx=i,
                    target_token=text_line[target_idx],
                    context_tokens=context_tokens))

        return result

    def get_entries(self, ids: List[int]) -> List[CBOWEntry]:
        return [self._entries[idx] for idx in ids]

    def get_indices_per_document(self) -> Dict[int, List[int]]:
        result = {
            i: []
            for i in set([x.document_index for x in self._entries])
        }

        for i, entry in enumerate(self._entries):
            result[entry.document_index].append(i)

        return result


    @property
    def length(self) -> int:
        return len(self._entries)
