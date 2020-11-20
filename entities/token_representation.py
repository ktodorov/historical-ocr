
class TokenRepresentation:
    def __init__(self, word: str, vocabulary_id: int):
        self._word = word
        self._vocabulary_id = vocabulary_id

    @property
    def word(self) -> str:
        return self._word

    @property
    def vocabulary_id(self) -> int:
        return self._vocabulary_id