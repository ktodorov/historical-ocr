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

from entities.model_checkpoint import ModelCheckpoint
from entities.batch_representation import BatchRepresentation

from models.model_base import ModelBase
from models.transformers.bert import BERT
from models.simple.cbow import CBOW

from services.arguments.ocr_evaluation_arguments_service import OCREvaluationArgumentsService
from services.data_service import DataService
from services.vocabulary_service import VocabularyService


class JointModel(ModelBase):
    def __init__(
            self,
            arguments_service: OCREvaluationArgumentsService,
            data_service: DataService,
            vocabulary_service: VocabularyService):
        super().__init__(data_service, arguments_service)

        self._arguments_service = arguments_service
        self._data_service = data_service
        self._vocabulary_service = vocabulary_service

        self._ocr_output_types = [OCROutputType.Raw, OCROutputType.GroundTruth]

        self._inner_models: List[ModelBase] = torch.nn.ModuleList([
            self._create_model(self._arguments_service.configuration, ocr_output_type) for ocr_output_type in self._ocr_output_types])

    @overrides
    def forward(self, tokens: torch.Tensor):
        return self.get_embeddings(tokens)

    @overrides
    def get_embeddings(self, tokens: List[str], vocab_ids: List[torch.Tensor], skip_unknown: bool = False) -> torch.Tensor:
        word_evaluation_sets = []
        for i, model in enumerate(self._inner_models):
            current_vocab_ids = vocab_ids[i] if vocab_ids is not None else None
            word_evaluations = model.get_embeddings(tokens, current_vocab_ids, skip_unknown=skip_unknown)
            word_evaluation_sets.append(word_evaluations)

        result = self._combine_word_evaluations(word_evaluation_sets)
        return result

    def _create_model(self, configuration: Configuration, ocr_output_type: OCROutputType):
        result = None
        if configuration == Configuration.BERT:
            result = BERT(
                arguments_service=self._arguments_service,
                data_service=self._data_service)
        elif configuration == Configuration.CBOW:
            result = CBOW(
                arguments_service=self._arguments_service,
                vocabulary_service=deepcopy(self._vocabulary_service),
                data_service=self._data_service,
                ocr_output_type=ocr_output_type)
        elif configuration == Configuration.SkipGram:
            result = SkipGram(
                arguments_service=self._arguments_service,
                vocabulary_service=deepcopy(self._vocabulary_service),
                data_service=self._data_service,
                ocr_output_type=ocr_output_type)

        if result is None:
            raise NotImplementedError()

        return result

    @overrides
    def load(
            self,
            path: str,
            name_prefix: str = None,
            name_suffix: str = None,
            load_model_dict: bool = True,
            load_model_only: bool = False,
            use_checkpoint_name: bool = True,
            checkpoint_name: str = None) -> ModelCheckpoint:

        for (ocr_output_type, model) in zip(self._ocr_output_types, self._inner_models):
            ocr_output_type_str = 'grt' if ocr_output_type == OCROutputType.GroundTruth else ocr_output_type.value
            model.load(
                path=path,
                name_prefix=name_prefix,
                name_suffix=f'-{ocr_output_type_str}',
                load_model_dict=load_model_dict,
                load_model_only=load_model_only,
                use_checkpoint_name=use_checkpoint_name,
                checkpoint_name=checkpoint_name)

        return None


    def _combine_word_evaluations(self, word_evaluations_sets: List[List[WordEvaluation]]) -> List[WordEvaluation]:
        unique_tokens = set([word_evaluation.word for word_evaluations in word_evaluations_sets for word_evaluation in word_evaluations])

        we_dict = {}
        for unique_token in unique_tokens:
            new_word_evaluation = WordEvaluation(unique_token)

            for i, word_evaluations in enumerate(word_evaluations_sets):
                for word_evaluation in word_evaluations:
                    if word_evaluation.word != unique_token:
                        continue

                    new_word_evaluation.add_embeddings(word_evaluation.get_embeddings(0), idx=i)

            we_dict[unique_token] = new_word_evaluation

        result = list(we_dict.values())
        return result