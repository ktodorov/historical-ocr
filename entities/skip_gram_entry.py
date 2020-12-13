from typing import List

class SkipGramEntry:
    def __init__(self, target_token: int, context_token: int):
        self._target_token = target_token
        self._context_token = context_token

    def __repr__(self):
        return f'target: \'{self._target_token}\' | context: \'{self._context_token}\''

    @property
    def target_token(self) -> int:
        return self._target_token

    @property
    def context_token(self) -> int:
        return self._context_token