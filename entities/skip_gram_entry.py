from typing import List

class SkipGramEntry:
    def __init__(self, target_token: int, context_tokens: List[int]):
        self._target_token = target_token
        self._context_tokens = context_tokens

    def __repr__(self):
        context_words_string = "','".join(self._context_tokens)
        return f'target: \'{self._target_token}\' | context: [{context_words_string}]'

    @property
    def target_token(self) -> int:
        return self._target_token

    @property
    def context_tokens(self) -> List[int]:
        return self._context_tokens