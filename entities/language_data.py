import pickle

from copy import deepcopy
from typing import Dict, List, Tuple
import math

from services.tokenize.base_tokenize_service import BaseTokenizeService


class LanguageData:
    def __init__(self):
        self._ocr_inputs: List[int] = []  # ocr_inputs
        self._ocr_aligned: List[int] = []
        self._gs_aligned: List[int] = []

        self._ocr_offsets: List[int] = []
        self._gs_offsets: List[int] = []

        self._ocr_texts: List[int] = []
        self._gs_texts: List[int] = []

        self._ocr_masks: List[int] = []
        self._gs_masks: List[int] = []

    @classmethod
    def from_pairs(
            cls,
            tokenize_service: BaseTokenizeService,
            pairs: list):
        new_obj = cls()

        for pair in pairs:
            new_obj.add_entry(
                pair[0], # OCR
                pair[1], # GT
                tokenize_service)

        return new_obj

    def add_entry(
            self,
            ocr_text: str,
            gs_text: str,
            tokenize_service: BaseTokenizeService):

        ocr_ids, _, ocr_offsets, ocr_mask = tokenize_service.encode_sequence(ocr_text)
        gs_ids, _, gs_offsets, gs_mask = tokenize_service.encode_sequence(gs_text)

        self._ocr_aligned.append(ocr_ids)
        self._gs_aligned.append(gs_ids)

        self._ocr_offsets.append(ocr_offsets)
        self._gs_offsets.append(gs_offsets)

        self._ocr_masks.append(ocr_mask)
        self._gs_masks.append(gs_mask)

    def get_entry(self, index: int):
        if index > self.length:
            raise Exception(
                'Index given is higher than the total items in the language data')

        result = (
            self._ocr_aligned[index],
            self._gs_aligned[index],
            None,
            None,
            # self._ocr_texts[index],
            # self._gs_texts[index],
            self._ocr_offsets[index],
            self._ocr_masks[index],
            self._gs_masks[index],
        )

        return result

    def cut_data(self, length: int):
        if length > self.length:
            raise Exception(
                'Length given is greater than the total items in the language data')

        self._ocr_aligned = deepcopy(self._ocr_aligned[:length])
        self._gs_aligned = deepcopy(self._gs_aligned[:length])
        self._ocr_texts = deepcopy(self._ocr_texts[:length])
        self._gs_texts = deepcopy(self._gs_texts[:length])
        self._ocr_offsets = deepcopy(self._ocr_offsets[:length])
        self._gs_offsets = deepcopy(self._gs_offsets[:length])
        self._ocr_masks = deepcopy(self._ocr_masks[:length])
        self._gs_masks = deepcopy(self._gs_masks[:length])

    def load_data(self, filepath: str):
        with open(filepath, 'rb') as data_file:
            language_data: LanguageData = pickle.load(data_file)

        if not language_data:
            return

        items_length = language_data.length

        self._ocr_aligned = language_data._ocr_aligned if hasattr(
            language_data, '_ocr_aligned') else [None] * items_length
        self._gs_aligned = language_data._gs_aligned if hasattr(
            language_data, '_gs_aligned') else [None] * items_length
        self._ocr_texts = language_data._ocr_texts if hasattr(
            language_data, '_ocr_texts') else [None] * items_length
        self._gs_texts = language_data._gs_texts if hasattr(
            language_data, '_gs_texts') else [None] * items_length
        self._ocr_offsets = language_data._ocr_offsets if hasattr(
            language_data, '_ocr_offsets') else [None] * items_length
        self._gs_offsets = language_data._gs_offsets if hasattr(
            language_data, '_gs_offsets') else [None] * items_length
        self._ocr_masks = language_data._ocr_masks if hasattr(
            language_data, '_ocr_masks') else [None] * items_length
        self._gs_masks = language_data._gs_masks if hasattr(
            language_data, '_gs_masks') else [None] * items_length

    @property
    def length(self) -> int:
        result = min(
            len(self._ocr_aligned),
            len(self._gs_aligned)
        )

        return result
