from tests.fakes.dataset_service_fake import DatasetServiceFake
from tests.fakes.train_service_fake import TrainServiceFake
from tests.fakes.log_service_fake import LogServiceFake
from tests.fakes.non_context_service_fake import NonContextServiceFake
from tests.fakes.model_fake import ModelFake

import os
from dependency_injection.ioc_container import IocContainer
import dependency_injector.providers as providers
import unittest

from enums.language import Language
from enums.configuration import Configuration
from enums.challenge import Challenge
from enums.ocr_output_type import OCROutputType

from losses.loss_base import LossBase
from optimizers.optimizer_base import OptimizerBase


def initialize_container(
    ocr_output_type: OCROutputType = None, 
    override_args: dict = None) -> IocContainer:
    custom_args = {
        'challenge': Challenge.OCREvaluation,
        'configuration': Configuration.SkipGram,
        'language': Language.English,
        'output_folder': os.path.join('tests', 'results'),
        'experiments_folder': os.path.join('tests', 'experiments'),
        'cache_folder': os.path.join('tests', '.cache'),
        'ocr_output_type': ocr_output_type,
        'checkpoint_name': 'local-test',
        'minimal_occurrence_limit': 5,
        'initialize_randomly': False,
        'patience': 10000,
        'consider_equal_results_as_worse': True
    }

    if override_args is not None:
        for key, value in override_args.items():
            custom_args[key] = value

    container = IocContainer()

    container.arguments_service.override(
        providers.Factory(
            NonContextServiceFake,
            custom_args))

    container.log_service.override(providers.Factory(LogServiceFake))
    container.train_service.override(
        providers.Singleton(
            TrainServiceFake,
            arguments_service=container.arguments_service,
            dataloader_service=container.dataloader_service,
            loss_function=container.loss_function,
            optimizer=container.optimizer,
            log_service=container.log_service,
            file_service=container.file_service,
            model=container.model))

    container.model.override(
        providers.Factory(
            ModelFake,
            arguments_service=container.arguments_service,
            log_service=container.log_service,
            data_service=container.data_service))

    container.optimizer.override(
        providers.Factory(
            OptimizerBase,
            arguments_service=container.arguments_service,
            model=container.model))

    container.loss_function.override(
        providers.Factory(
            LossBase))

    container.dataset_service.override(
        providers.Factory(
            DatasetServiceFake,
            arguments_service=container.arguments_service,
            cache_service=container.cache_service,
            process_service=container.process_service,
            log_service=container.log_service))

    return container

def distinct(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

class TestTraining(unittest.TestCase):

    def test_batch_order_different_corpora_same_seed(self):
        pass
        # # Raw model
        container_1 = initialize_container(ocr_output_type=OCROutputType.Raw)
        container_1.main()

        train_service_1 = container_1.train_service()
        ids_1 = train_service_1.data_loader_train.dataset._ids
        unique_ids_1 = distinct(ids_1)

        # Ground truth model
        container_2 = initialize_container(
            ocr_output_type=OCROutputType.GroundTruth)
        container_2.main()

        train_service_2 = container_2.train_service()
        ids_2 = train_service_2.data_loader_train.dataset._ids
        unique_ids_2 = distinct(ids_2)

        self.assertListEqual(unique_ids_1, unique_ids_2)

    def test_batch_order_different_corpora_different_seed(self):
        pass
        # # Raw model
        container_1 = initialize_container(ocr_output_type=OCROutputType.Raw)
        container_1.main()

        train_service_1 = container_1.train_service()
        ids_1 = train_service_1.data_loader_train.dataset._ids
        ids_1_string = ''.join([str(x) for x in distinct(ids_1)])

        # Ground truth model
        container_2 = initialize_container(
            ocr_output_type=OCROutputType.GroundTruth,
            override_args={
                'seed': 7
            })
        container_2.main()

        train_service_2 = container_2.train_service()
        ids_2 = train_service_2.data_loader_train.dataset._ids
        ids_2_string = ''.join([str(x) for x in distinct(ids_2)])

        self.assertNotEqual(ids_1_string, ids_2_string)

if __name__ == '__main__':
    unittest.main()
