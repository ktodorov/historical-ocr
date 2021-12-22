import os
from typing import List, Tuple

import numpy as np
import torch
# import sklearn.manifold.t_sne

from enums.experiment_type import ExperimentType

from entities.batch_representation import BatchRepresentation

from models.evaluation_model import EvaluationModel

from services.arguments.pretrained_arguments_service import PretrainedArgumentsService
from services.dataloader_service import DataLoaderService
from services.file_service import FileService

class ExperimentServiceBase:
    def __init__(
            self,
            arguments_service: PretrainedArgumentsService,
            dataloader_service: DataLoaderService,
            file_service: FileService,
            model: EvaluationModel):

        self._model = model.to(arguments_service.device)
        self._dataloader_service = dataloader_service
        self._file_service = file_service

        self._dataloader = self._dataloader_service.get_test_dataloader()

        checkpoints_path = self._file_service.get_checkpoints_path()
        self._model.load(checkpoints_path, 'BEST')
        self._model.eval()

    def execute_experiments(self, experiment_types: List[ExperimentType]):
        pass

