from enums.language import Language
from entities.cache.cache_options import CacheOptions
import enum
from services.tokenize.base_tokenize_service import BaseTokenizeService
from services.cache_service import CacheService
from services.file_service import FileService
from services.log_service import LogService
from services.process.process_service_base import ProcessServiceBase
from models.simple.ppmi import PPMI
from entities.word_evaluation import WordEvaluation
from models.simple.skip_gram import SkipGram
from torch.utils import data
from enums.configuration import Configuration
import os
import numpy as np
from overrides.overrides import overrides
import torch
from typing import List
import torch
from copy import deepcopy

from enums.ocr_output_type import OCROutputType

from entities.models.model_checkpoint import ModelCheckpoint
from entities.batch_representation import BatchRepresentation

from models.model_base import ModelBase
from models.transformers.bert import BERT
from models.simple.cbow import CBOW

from services.arguments.ocr_evaluation_arguments_service import OCREvaluationArgumentsService
from services.data_service import DataService
from services.vocabulary_service import VocabularyService


class EvaluationModel(ModelBase):
    def __init__(
            self,
            arguments_service: OCREvaluationArgumentsService,
            data_service: DataService,
            vocabulary_service: VocabularyService,
            process_service: ProcessServiceBase,
            log_service: LogService,
            file_service: FileService,
            cache_service: CacheService,
            tokenize_service: BaseTokenizeService):
        super().__init__(data_service, arguments_service, log_service)

        self._arguments_service = arguments_service
        self._data_service = data_service
        self._vocabulary_service = vocabulary_service
        self._process_service = process_service
        self._log_service = log_service
        self._tokenize_service = tokenize_service

        self._ocr_output_types = [OCROutputType.Raw, OCROutputType.GroundTruth]

        self._inner_models: List[ModelBase] = torch.nn.ModuleList([
            self._create_model(self._arguments_service.configuration, ocr_output_type) for ocr_output_type in self._ocr_output_types])

        self._inner_models.append(
            # BASE
            SkipGram(
                arguments_service=self._arguments_service,
                vocabulary_service=VocabularyService(
                    self._data_service,
                    file_service,
                    cache_service,
                    log_service,
                    overwrite_configuration=Configuration.SkipGram),
                data_service=self._data_service,
                log_service=self._log_service,
                ocr_output_type=OCROutputType.GroundTruth)
        )

        if self._arguments_service.configuration in [Configuration.SkipGram, Configuration.CBOW, Configuration.BERT]:
            pretrained_matrix = None
            if self._arguments_service.configuration in [Configuration.SkipGram, Configuration.CBOW]:
                pretrained_matrix = cache_service.get_item_from_cache(
                    CacheOptions(
                        f'word-matrix-{self._arguments_service.get_dataset_string()}-{OCROutputType.GroundTruth.value}',
                        seed_specific=True))

                pretrained_matrix = pretrained_matrix.to(self._arguments_service.device)

            self._inner_models.append(
                self._create_model(
                    self._arguments_service.configuration,
                    OCROutputType.GroundTruth,
                    pretrained_matrix=pretrained_matrix,
                    overwrite_initialization=True))

    @overrides
    def forward(self, tokens: torch.Tensor):
        self._log_service.log_warning('Joint model currently does not have a forward pass implemented properly. Please use `get_embeddings` instead')
        raise NotImplementedError()

    @overrides
    def get_embeddings(self, tokens: List[str], skip_unknown: bool = False) -> torch.Tensor:
        word_evaluation_sets = []
        for model in self._inner_models:
            embeddings_list = model.get_embeddings(tokens, skip_unknown=skip_unknown)
            word_evaluation_sets.append(embeddings_list)

        result = self._combine_word_evaluations(tokens, word_evaluation_sets)
        return result

    def _create_model(
        self,
        configuration: Configuration,
        ocr_output_type: OCROutputType,
        pretrained_matrix = None,
        overwrite_initialization: bool = False):
        result = None
        if configuration == Configuration.BERT:
            result = BERT(
                arguments_service=self._arguments_service,
                data_service=self._data_service,
                log_service=self._log_service,
                tokenize_service=self._tokenize_service,
                overwrite_initialization=overwrite_initialization)
        elif configuration == Configuration.CBOW:
            result = CBOW(
                arguments_service=self._arguments_service,
                vocabulary_service=deepcopy(self._vocabulary_service),
                data_service=self._data_service,
                log_service=self._log_service,
                pretrained_matrix=pretrained_matrix,
                ocr_output_type=ocr_output_type)
        elif configuration == Configuration.SkipGram:
            result = SkipGram(
                arguments_service=self._arguments_service,
                vocabulary_service=deepcopy(self._vocabulary_service),
                data_service=self._data_service,
                log_service=self._log_service,
                pretrained_matrix=pretrained_matrix,
                ocr_output_type=ocr_output_type)
        elif configuration == Configuration.PPMI:
            result = PPMI(
                arguments_service=self._arguments_service,
                vocabulary_service=deepcopy(self._vocabulary_service),
                data_service=self._data_service,
                log_service=self._log_service,
                process_service=self._process_service,
                ocr_output_type=ocr_output_type)

        if result is None:
            self._log_service.log_error('Joint model inner type is not implemented')
            raise NotImplementedError()

        return result

    @overrides
    def load(
            self,
            path: str,
            name_prefix: str = None,
            name_suffix: str = None,
            load_model_dict: bool = True,
            use_checkpoint_name: bool = True,
            checkpoint_name: str = None) -> ModelCheckpoint:
        self._log_service.log_debug('Loading joint models..')

        for (ocr_output_type, model) in zip(self._ocr_output_types, self._inner_models[:2]):
            ocr_output_type_str = 'grt' if ocr_output_type == OCROutputType.GroundTruth else ocr_output_type.value
            model.load(
                path=path,
                name_prefix=name_prefix,
                name_suffix=f'-{ocr_output_type_str}',
                load_model_dict=load_model_dict,
                use_checkpoint_name=use_checkpoint_name,
                checkpoint_name=checkpoint_name)

        skip_gram_model = self._inner_models[2]
        skip_gram_overwrite_args = {
            'initialize_randomly': True,
            'configuration': Configuration.SkipGram.value,
            'learning_rate': 1e-3,# if self._arguments_service.language == Language.English else 1e-2,
            'minimal_occurrence_limit': 5
        }

        skip_gram_model.load(
            path=path.replace(self._arguments_service.configuration.value, Configuration.SkipGram.value),
            name_prefix=name_prefix,
            name_suffix=f'-{ocr_output_type_str}',
            load_model_dict=load_model_dict,
            use_checkpoint_name=use_checkpoint_name,
            checkpoint_name=checkpoint_name,
            overwrite_args=skip_gram_overwrite_args)

        self._log_service.log_debug('Loading joint models succeeded')
        return None


    def _combine_word_evaluations(self, tokens: List[str], embeddings_list: List[List[List[float]]]) -> List[WordEvaluation]:
        # unique_tokens = set([word_evaluation.word for word_evaluations in word_evaluations_sets for word_evaluation in word_evaluations])

        result = []

        for i, token in enumerate(tokens):
            we = WordEvaluation(token, embeddings_list=[
                x[i] for x in embeddings_list
            ])

            result.append(we)

        # we_dict = {}
        # for unique_token in unique_tokens:
        #     new_word_evaluation = WordEvaluation(unique_token)

        #     for i, word_evaluations in enumerate(word_evaluations_sets):
        #         for word_evaluation in word_evaluations:
        #             if word_evaluation.word != unique_token:
        #                 continue

        #             new_word_evaluation.add_embeddings(word_evaluation.get_embeddings(0), idx=i)

        #     we_dict[unique_token] = new_word_evaluation

        # result = list(we_dict.values())
        return result