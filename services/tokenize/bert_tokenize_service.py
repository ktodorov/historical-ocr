import os

from typing import Tuple, List

from overrides import overrides

from tokenizers import BertWordPieceTokenizer

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


import sentencepiece as spm

from enums.configuration import Configuration
from services.arguments.pretrained_arguments_service import PretrainedArgumentsService

from services.tokenize.base_tokenize_service import BaseTokenizeService
from services.file_service import FileService
from services.download.ocr_download_service import OCRDownloadService

class BERTTokenizeService(BaseTokenizeService):
    def __init__(
            self,
            arguments_service: PretrainedArgumentsService,
            file_service: FileService,
            ocr_download_service: OCRDownloadService):
        super().__init__()

        self._arguments_service = arguments_service
        self._file_service = file_service
        self._ocr_download_service = ocr_download_service

        pretrained_weights = self._arguments_service.pretrained_weights

        pretrained_weights = arguments_service.pretrained_weights
        self._tokenizer = self._load_tokenizer(pretrained_weights)
        if self._tokenizer is None:
            self._tokenizer = self._train_tokenizer(pretrained_weights)

        self._tokenizer._tokenizer.post_processor = BertProcessing(
            ("[SEP]", self._tokenizer.token_to_id("[SEP]")),
            ("[CLS]", self._tokenizer.token_to_id("[CLS]")),
        )

        # self._arguments_service = arguments_service
        # vocabulary_path = os.path.join(arguments_service.data_folder, 'vocabularies', f'{pretrained_weights}-vocab.txt')
        # if not os.path.exists(vocabulary_path):
        #     raise Exception(f'Vocabulary not found in {vocabulary_path}')

        # self._tokenizer: BertWordPieceTokenizer = BertWordPieceTokenizer(
        #     vocabulary_path, lowercase=False)

    @overrides
    def encode_tokens(self, tokens: List[str]) -> List[int]:
        result = [self._tokenizer.token_to_id(x) for x in tokens]
        return result

    @overrides
    def decode_tokens(self, character_ids: List[int]) -> List[str]:
        result = [self._tokenizer.id_to_token(
            character_id) for character_id in character_ids]
        return result

    @overrides
    def decode_string(self, character_ids: List[int]) -> List[str]:
        result = self._tokenizer.decode(character_ids)
        return result

    @overrides
    def id_to_token(self, character_id: int) -> str:
        result = self._tokenizer.id_to_token(character_id)
        return result

    @overrides
    def encode_sequence(self, sequence: str, add_special_tokens: bool = True) -> Tuple[List[int], List[str], List[Tuple[int,int]], List[int]]:
        encoded_representation = self._tokenizer.encode(sequence, add_special_tokens=add_special_tokens)
        return (
            encoded_representation.ids,
            encoded_representation.tokens,
            encoded_representation.offsets,
            encoded_representation.special_tokens_mask)

    @overrides
    def encode_sequences(self, sequences: List[str]) -> List[Tuple[List[int], List[str], List[Tuple[int,int]], List[int]]]:
        encoded_representations = self._tokenizer.encode_batch(sequences)
        return [(x.ids, x.tokens, x.offsets, x.special_tokens_mask) for x in encoded_representations]

    @property
    @overrides
    def vocabulary_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def _load_tokenizer(self, pretrained_weights: str) -> ByteLevelBPETokenizer:
        result = None
        vocabulary_path = os.path.join(
            self._arguments_service.data_folder, 'vocabularies', self._arguments_service.language.value, f'{pretrained_weights}-vocab.json')
        merges_path = os.path.join(
            self._arguments_service.data_folder, 'vocabularies', self._arguments_service.language.value, f'{pretrained_weights}-merges.txt')

        if os.path.exists(vocabulary_path) and os.path.exists(merges_path):
            result = ByteLevelBPETokenizer(vocabulary_path, merges_path)

        return result

    def _train_tokenizer(self, pretrained_weights: str) -> ByteLevelBPETokenizer:
        file_paths = self._ocr_download_service.get_downloaded_file_paths(self._arguments_service.language)
        tokenizer = ByteLevelBPETokenizer()

        tokenizer.train(
            files=file_paths,
            min_frequency=2,
            special_tokens=[
                "[PAD]",
                "[CLS]",
                "[SEP]",
                "[UNK]",
                "[MASK]",
            ])

        save_path = self._file_service.combine_path(self._arguments_service.data_folder, 'vocabularies', self._arguments_service.language.value, create_if_missing=True)
        tokenizer.save_model(save_path, pretrained_weights)

        return tokenizer