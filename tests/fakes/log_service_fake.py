from datetime import datetime, timedelta
from services.log_service import LogService
import torch
import numpy as np

from entities.metric import Metric
from entities.data_output_log import DataOutputLog


class LogServiceFake(LogService):
    def __init__(self):
        pass

    def log_progress(
            self,
            current_step: int,
            all_steps: int,
            epoch_num: int = None,
            evaluation: bool = False):

        pass

    def initialize_evaluation(self):
        pass

    def log_evaluation(
            self,
            train_metric: Metric,
            validation_metric: Metric,
            batches_done: int,
            epoch: int,
            iteration: int,
            iterations: int,
            new_best: bool,
            metric_log_key: str = None):
        """
        logs progress to user through tensorboard and terminal
        """

        pass

    def log_summary(self, key: str, value: object):
        pass

    def log_batch_results(self, data_output_log: DataOutputLog):
        pass

    def log_incremental_metric(self, metric_key: str, metric_value: object):
        pass

    def log_heatmap(
            self,
            heatmap_title: str,
            matrix_values: np.array,
            x_labels: list,
            y_labels: list,
            show_text_inside: bool = False):
        pass

    def start_logging_model(self, model: torch.nn.Module, criterion: torch.nn.Module = None):
        pass

    def get_time_passed(self) -> timedelta:
        result = datetime.now()
        return result

    def _get_current_step(self) -> int:
        return 0
