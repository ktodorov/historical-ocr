from typing import Dict, List

from overrides.overrides import overrides
from datasets.dataset_base import DatasetBase


class DocumentDatasetBase(DatasetBase):
    def __init__(self):
        super().__init__()

    def get_indices_per_document(self) -> Dict[int, List[int]]:
        return {}

    def use_collate_function(self) -> bool:
        return True

    def collate_function(self, sequences):
        return sequences[0]
