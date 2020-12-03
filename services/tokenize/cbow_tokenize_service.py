import re
from services.vocabulary_service import VocabularyService
import string
from typing import List, Tuple
from nltk.tokenize import RegexpTokenizer


from services.tokenize.base_tokenize_service import BaseTokenizeService

class CBOWTokenizeService(BaseTokenizeService):
    def __init__(
        self,
        vocabulary_service: VocabularyService):
        super().__init__()

        self._vocabulary_service = vocabulary_service
        self._tokenizer = RegexpTokenizer(r'\w+')

    def encode_tokens(self, tokens: List[str]) -> List[int]:
        pass

    def decode_tokens(self, character_ids: List[int]) -> List[str]:
        result = self._vocabulary_service.ids_to_strings(character_ids)
        return result

    def decode_string(self, character_ids: List[int]) -> List[str]:
        pass

    def id_to_token(self, character_id: int) -> str:
        pass

    def encode_sequence(self, sequence: str) -> Tuple[List[int], List[str], List[Tuple[int,int]], List[int]]:
        pass

    def encode_sequences(self, sequences: List[str]) -> List[Tuple[List[int], List[str], List[Tuple[int,int]], List[int]]]:
        result = [([self._vocabulary_service.string_to_id(x)], None, None, None) for x in sequences]
        return result

    def tokenize_sequences(self, sequences: List[str]) -> List[List[str]]:
        result = [self._tokenizer.tokenize(self._clean_text(sequence)) for sequence in sequences]
        return result

    def _clean_text(self, text):
        # remove numbers
        text_nonum = re.sub(r'\d+', '', text)
        # remove punctuations and convert characters to lower case
        text_nopunct = "".join([char.lower() for char in text_nonum if char not in string.punctuation])
        # substitute multiple whitespace with single whitespace
        # Also, removes leading and trailing whitespaces
        text_no_doublespace = re.sub('\s+', ' ', text_nopunct).strip()
        return text_no_doublespace