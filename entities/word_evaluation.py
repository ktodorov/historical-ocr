class WordEvaluation:
    def __init__(self, word: str, embeddings_1: list, embeddings_2: list):
        self._word = word
        self._embeddings_1 = embeddings_1
        self._embeddings_2 = embeddings_2

    def get_embeddings(self, idx: int) -> list:
        if idx == 0:
            return self.embeddings_1
        elif idx == 1:
            return self.embeddings_2

        raise Exception('Invalid embeddings index')

    @property
    def word(self) -> str:
        return self._word

    @property
    def embeddings_1(self) -> list:
        return self._embeddings_1

    @property
    def embeddings_2(self) -> list:
        return self._embeddings_2

    def __str__(self):
        return self._word
