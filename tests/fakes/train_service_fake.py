from entities.metric import Metric
from enums.metric_type import MetricType
from typing import Dict, List, Tuple
from overrides import overrides
import torch

from services.train_service import TrainService


class TrainServiceFake(TrainService):
    def __init__(
            self,
            arguments_service,
            dataloader_service,
            loss_function,
            optimizer,
            log_service,
            file_service,
            model):
            super().__init__(
                arguments_service,
                dataloader_service,
                loss_function,
                optimizer,
                log_service,
                file_service,
                model)


    def _perform_batch_iteration(
            self,
            batch: torch.Tensor,
            train_mode: bool = True,
            output_characters: bool = False) -> Tuple[float, Dict[MetricType, float], List[str]]:
        return (0, {}, None)
