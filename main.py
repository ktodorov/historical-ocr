from services.log_service import LogService
import numpy as np
import torch
import random

from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.data_service import DataService
from services.train_service import TrainService
from services.test_service import TestService
from services.experiments.experiment_service_base import ExperimentServiceBase

def initialize_seed(seed: int, device: str):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if device == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)


def main(
        arguments_service: ArgumentsServiceBase,
        train_service: TrainService,
        test_service: TestService,
        experiment_service: ExperimentServiceBase,
        log_service: LogService):

    log_service.log_arguments()
    initialize_seed(arguments_service.seed, arguments_service.device)

    try:
        if arguments_service.evaluate:
            log_service.log_debug('Starting TEST run')
            test_service.test()
        elif not arguments_service.run_experiments:
            log_service.log_debug('Starting TRAIN run')
            train_service.train()
        else:
            log_service.log_debug('Starting EXPERIMENT run')
            experiment_service.execute_experiments(arguments_service.experiment_types)
    except Exception as exception:
        log_service.log_exception('Stopping program execution', exception)
        raise exception