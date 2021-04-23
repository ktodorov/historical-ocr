from enums.overlap_type import OverlapType
from typing import List

class WordEvaluation:
    def __init__(self, word: str, embeddings_list: List[List[List[float]]] = None):
        self._word = word
        self._embeddings = None

        if embeddings_list is not None:
            self._embeddings = embeddings_list
            self._known_tokens = [x is not None for x in embeddings_list]

    def add_embeddings(self, embeddings: List[int], idx: int):
        if self._embeddings is None:
            self._embeddings = []

        # make sure the list is big enough
        while len(self._embeddings) <= idx:
            self._embeddings.append(None)

        if embeddings is not None:
            self._embeddings[idx] = embeddings

        self._known_tokens = [x is not None for x in self._embeddings]

    def get_embeddings(self, idx: int) -> list:
        if idx > len(self._embeddings):
            raise Exception('Invalid embeddings index')

        return self._embeddings[idx]

    def get_embeddings_size(self) -> int:
        filled_embeddings = list(filter(lambda x: x is not None, self._embeddings))
        if len(filled_embeddings) == 0:
            return None

        return len(filled_embeddings[0])

    @property
    def word(self) -> str:
        return self._word

    def token_is_known(self, idx: int) -> bool:
        return self._known_tokens[idx]

    def contains_all_embeddings(self, overlap_type: OverlapType) -> bool:
        result = self._embeddings[0] is not None

        if overlap_type == OverlapType.GTvsRaw:
            return (result and self._embeddings[1] is not None)
        elif overlap_type == OverlapType.GTvsBase:
            return (result and self._embeddings[2] is not None)
        elif overlap_type == OverlapType.GTvsOriginal:
            return (result and self._embeddings[3] is not None)

        raise NotImplementedError(f'Overlap type {overlap_type.value} is not implemented')

    def __str__(self):
        return self._word
