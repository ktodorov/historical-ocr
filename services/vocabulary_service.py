import os
from typing import List, Dict, Tuple

import nltk
from nltk.corpus import wordnet as wn

from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.data_service import DataService
from services.file_service import FileService
from services.cache_service import CacheService


class VocabularyService:
    def __init__(
            self,
            data_service: DataService,
            file_service: FileService,
            cache_service: CacheService):

        self._data_service = data_service
        self._file_service = file_service
        self._cache_service = cache_service

        self._vocabulary_cache_key = 'vocab'

        self._id2token: Dict[int, str] = {}
        self._token2idx: Dict[str, int] = {}
        cached_vocabulary = self._load_cached_vocabulary()
        if cached_vocabulary is not None:
            (self._token2idx, self._id2token) = cached_vocabulary

    def initialize_vocabulary_data(self, vocabulary_data):
        if vocabulary_data is None:
            return

        self._id2token: Dict[int, str] = vocabulary_data['id2token']
        self._token2idx: Dict[str, int] = vocabulary_data['token2id']

    def string_to_ids(self, input: List[str]) -> List[int]:
        result = [self.string_to_id(x) for x in input]
        return result

    def string_to_id(self, input: str) -> int:
        if input in self._token2idx.keys():
            return self._token2idx[input]

        return self.unk_token

    def ids_to_string(
        self,
        input: List[int],
        exclude_special_tokens: bool = True,
        join_str: str = '',
        cut_after_end_token: bool = False) -> str:
        if join_str is None:
            raise Exception('`join_str` must be a valid string')

        result = join_str.join([self._id2token[x] for x in input])

        if cut_after_end_token:
            try:
                eos_index = result.index('[EOS]')
                result = result[:eos_index]
            except ValueError:
                pass

        if exclude_special_tokens:
            result = result.replace('[PAD]', '')
            result = result.replace('[EOS]', '')
            result = result.replace('[CLS]', '')

        return result

    def ids_to_strings(
        self,
        input: List[int],
        exclude_pad_tokens: bool = True) -> List[str]:
        result = [self._id2token[x] for x in input]

        if exclude_pad_tokens:
            result = list(filter(lambda x: x != '[PAD]', result))

        return result

    def vocabulary_size(self) -> int:
        return len(self._id2token)

    def get_all_english_nouns(self, limit_amount: int = None) -> List[str]:
        pickles_path = self._file_service.get_pickles_path()
        english_nouns_path = os.path.join(pickles_path, 'english')
        filename = f'english_nouns'
        words = self._data_service.load_python_obj(
            english_nouns_path, filename)
        if words is not None:
            return words

        filepath = os.path.join(english_nouns_path, f'{filename}.txt')
        if not os.path.exists(filepath):
            return None

        with open(filepath, 'r', encoding='utf-8') as noun_file:
            words = [x.replace('\n', '') for x in noun_file.readlines()]

        if limit_amount:
            words = words[:limit_amount]

        words = list(set(words))

        self._data_service.save_python_obj(words, english_nouns_path, filename)

        return words

    def get_vocabulary_tokens(self) -> List[Tuple[int, str]]:
        for index, token in self._id2token.items():
            yield (index, token)

    def initialize_vocabulary_from_corpus(self, tokenized_corpus: List[List[str]]):
        if len(self._token2idx) > 0 and len(self._id2token) > 0:
            return

        unique_tokens = list(set([token for sentence in tokenized_corpus for token in sentence]))
        unique_tokens = list(sorted(unique_tokens))

        vocabulary = [
            '[PAD]',
            '[CLS]',
            '[EOS]',
            '[UNK]'
        ]

        vocabulary.extend(unique_tokens)

        self._token2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
        self._id2token = {idx: w for (idx, w) in enumerate(vocabulary)}

        self._cache_vocabulary()

    def _cache_vocabulary(self):
        self._cache_service.cache_item(
            self._vocabulary_cache_key,
            [
                self._token2idx,
                self._id2token
            ],
            overwrite=False)

    def _load_cached_vocabulary(self):
        return self._cache_service.get_item_from_cache(self._vocabulary_cache_key)

    @property
    def cls_token(self) -> int:
        return self._token2idx['[CLS]']

    @property
    def eos_token(self) -> int:
        return self._token2idx['[EOS]']

    @property
    def unk_token(self) -> int:
        return self._token2idx['[UNK]']

    @property
    def pad_token(self) -> int:
        return self._token2idx['[PAD]']
