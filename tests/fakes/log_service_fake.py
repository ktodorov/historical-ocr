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
            epoch: int,
            iteration: int,
            iterations: int,
            new_best: bool,
            metric_log_key: str = None):
        """
        logs progress to user through tensorboard and terminal
        """

        pass

    def log_info(self, message: str):
        print(message)

    def log_debug(self, message: str):
        print(message)

    def log_error(self, message: str):
        print(message)

    def log_exception(self, message: str, exception: Exception):
        log_message = f'Exception occurred. Message: {message}\nOriginal exception: {exception}'
        print(log_message)

    def log_warning(self, message: str):
        print(message)

    def log_summary(self, key: str, value: object):
        pass

    def log_batch_results(self, data_output_log: DataOutputLog):
        pass

    def log_incremental_metric(self, metric_key: str, metric_value: object):
        pass

    def log_arguments(self):
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
        result = timedelta(minutes=60)
        return result

    def _get_current_step(self) -> int:
        return 0
