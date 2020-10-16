import os
from typing import List, Tuple

import numpy as np
import torch
# import sklearn.manifold.t_sne
from MulticoreTSNE import MulticoreTSNE as TSNE

from enums.experiment_type import ExperimentType

from entities.batch_representation import BatchRepresentation

from models.model_base import ModelBase

from services.arguments.pretrained_arguments_service import PretrainedArgumentsService
from services.metrics_service import MetricsService
from services.file_service import FileService
from services.tokenize.base_tokenize_service import BaseTokenizeService
from services.vocabulary_service import VocabularyService
from services.plot_service import PlotService
from services.data_service import DataService
from services.cache_service import CacheService


class ExperimentService:
    def __init__(
            self,
            arguments_service: PretrainedArgumentsService,
            metrics_service: MetricsService,
            file_service: FileService,
            tokenize_service: BaseTokenizeService,
            vocabulary_service: VocabularyService,
            plot_service: PlotService,
            data_service: DataService,
            cache_service: CacheService,
            model: ModelBase):

        self._arguments_service = arguments_service
        self._metrics_service = metrics_service
        self._file_service = file_service
        self._tokenize_service = tokenize_service
        self._vocabulary_service = vocabulary_service
        self._plot_service = plot_service
        self._data_service = data_service
        self._cache_service = cache_service

        self._model = model.to(arguments_service.device)

    def execute_experiments(self, experiment_types: List[ExperimentType]):
        checkpoints_path = self._file_service.get_checkpoints_path()
        self._model.load(checkpoints_path, 'BEST')
        self._model.eval()

        pass # TODO

