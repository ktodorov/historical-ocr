from tests.fakes.train_service_fake import TrainServiceFake
from tests.fakes.log_service_fake import LogServiceFake
from enums.language import Language
from enums.configuration import Configuration
from enums.challenge import Challenge
from enums.ocr_output_type import OCROutputType
from enums.experiment_type import ExperimentType
import os
from tests.fakes.non_context_service_fake import NonContextServiceFake
from tests.fakes.evaluation_service_fake import EvaluationServiceFake
from dependency_injection.ioc_container import IocContainer
import dependency_injector.providers as providers
import torch
import unittest
from shutil import copyfile


def initialize_container(
    ocr_output_type: OCROutputType = None, 
    override_args: dict = None,
    evaluation: bool = False) -> IocContainer:
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
        'patience': 1,
        'consider_equal_results_as_worse': True
    }

    if override_args is not None:
        for key, value in override_args.items():
            custom_args[key] = value

    container = IocContainer()

    if evaluation:
        container.arguments_service.override(
            providers.Factory(
                EvaluationServiceFake,
                custom_args))
    else:
        container.arguments_service.override(
            providers.Factory(
                NonContextServiceFake,
                custom_args))

    container.log_service.override(providers.Factory(LogServiceFake))
    container.train_service.override(
        providers.Factory(
            TrainServiceFake,
            arguments_service=container.arguments_service,
            dataloader_service=container.dataloader_service,
            loss_function=container.loss_function,
            optimizer=container.optimizer,
            log_service=container.log_service,
            file_service=container.file_service,
            model=container.model))

    return container


class TestNeighbourhoodPlots(unittest.TestCase):

    def test_neighbourhood_plot_of_new_model(self):
        language = 'english'
        preferred_tokens_path = os.path.join('experiments', f'preferred-tokens-{language}.txt')
        tests_preferred_tokens_path = os.path.join('tests', preferred_tokens_path)
        if os.path.exists(preferred_tokens_path) and not os.path.exists(tests_preferred_tokens_path):
            copyfile(preferred_tokens_path, tests_preferred_tokens_path)

        raw_checkpoint_filepath = os.path.join('tests', 'results', 'ocr-evaluation', 'skip-gram', 'english', 'BEST_en-skip-gram-raw-lim-5-local-test.pickle')
        if os.path.exists(raw_checkpoint_filepath):
            os.remove(raw_checkpoint_filepath)

        grt_checkpoint_filepath = os.path.join('tests', 'results', 'ocr-evaluation', 'skip-gram', 'english', 'BEST_en-skip-gram-grt-lim-5-local-test.pickle')
        if os.path.exists(grt_checkpoint_filepath):
            os.remove(grt_checkpoint_filepath)

        # Raw model
        container_1 = initialize_container(ocr_output_type=OCROutputType.Raw)
        container_1.main()

        assert os.path.exists(raw_checkpoint_filepath)

        # Ground truth model
        container_2 = initialize_container(
            ocr_output_type=OCROutputType.GroundTruth)
        container_2.main()

        assert os.path.exists(grt_checkpoint_filepath)

        experiments_container = initialize_container(
            override_args={
                'separate_neighbourhood_vocabularies': True,
                'run_experiments': True,
                'experiment_types': [ExperimentType.CosineSimilarity, ExperimentType.CosineDistance],
                'batch_size': 128,
                'joint_model': True
            },
            evaluation=True)

        experiments_container.main()

        assert len(os.listdir(os.path.join('tests', 'experiments', 'neighbourhoods', language, 'skip-gram'))) > 0

if __name__ == '__main__':
    unittest.main()
