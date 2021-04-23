from typing import Dict, List

from entities.word2vec.word2vec_entry import Word2VecEntry


class Word2VecCorpus:
    def __init__(self, text_lines: List[List[int]], window_size: int):
        self._window_size = window_size
        self._entries = self._generate_skip_gram_entries(text_lines)
        self._total_documents = len(
            set([x.document_index for x in self._entries]))

    def cut_data(self, corpus_size: int):
        self._entries = self._entries[:corpus_size]
        self._total_documents = len(
            set([x.document_index for x in self._entries]))

    def _generate_skip_gram_entries(self, text_lines: List[List[int]]) -> List[Word2VecEntry]:
        result = []

        for i, text_line in enumerate(text_lines):
            if len(text_line) <= 1:
                continue

            for target_idx in range(len(text_line)):
                window_start = max(0, target_idx-self._window_size)
                window_end = min(len(text_line), target_idx +
                                 self._window_size+1)
                context_tokens = text_line[window_start:target_idx] + \
                    text_line[target_idx+1:window_end]
                result.append(Word2VecEntry(
                    document_idx=i,
                    target_token=text_line[target_idx],
                    context_tokens=context_tokens))

        return result

    def get_entries(self, ids: List[int]) -> List[Word2VecEntry]:
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
