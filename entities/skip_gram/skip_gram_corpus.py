from typing import Dict, List

from entities.skip_gram.skip_gram_entry import SkipGramEntry
from itertools import takewhile


class SkipGramCorpus:
    def __init__(self, text_lines: List[List[int]]):
        self._window_size = 1
        self._total_documents = len(text_lines)
        self._entries = self._generate_skip_gram_entries(text_lines)

    def cut_data(self, corpus_size: int):
        self._entries = self._entries[:corpus_size]
        self._total_documents = len(set([x.document_index for x in self._entries]))

    def _generate_skip_gram_entries(self, text_lines: List[List[int]]) -> List[SkipGramEntry]:
        result = []

        for i, text_line in enumerate(text_lines):
            for target_idx in range(len(text_line)):

                # if target is not the first token, we add new entry where context token is the previous one
                if target_idx > 0:
                    result.append(
                        SkipGramEntry(
                            document_idx=i,
                            target_token=text_line[target_idx],
                            context_token=text_line[target_idx-1]))

                # if target is not the last token, we add new entry where context token is the next one
                if target_idx < len(text_line) - 1:
                    result.append(
                        SkipGramEntry(
                            document_idx=i,
                            target_token=text_line[target_idx],
                            context_token=text_line[target_idx+1]))

        return result

    def get_entries(self, ids: List[int]) -> List[SkipGramEntry]:
        return [self._entries[idx] for idx in ids]

    def get_indices_per_document(self) -> Dict[int, List[int]]:
        result = {
            i: []
            for i in range(self._total_documents)
        }

        for i, entry in enumerate(self._entries):
            result[entry.document_index].append(i)

        return result

    @property
    def length(self) -> int:
        return len(self._entries)
