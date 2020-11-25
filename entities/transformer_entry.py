from typing import List

class TransformerEntry:
    def __init__(self, token_ids: List[int], mask_ids: List[int]):
        self._token_ids = token_ids
        self._mask_ids = mask_ids

    @property
    def token_ids(self) -> List[int]:
        return self._token_ids

    @property
    def mask_ids(self) -> List[int]:
        return self._mask_ids