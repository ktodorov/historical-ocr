from models.simple.skip_gram import SkipGram
from tests.fakes.log_service_fake import LogServiceFake
from enums.language import Language
from enums.configuration import Configuration
from enums.challenge import Challenge
from enums.ocr_output_type import OCROutputType
from entities.cache.cache_options import CacheOptions
import os
from tests.fakes.non_context_service_fake import NonContextServiceFake
from dependency_injection.ioc_container import IocContainer
import dependency_injector.providers as providers
import torch
import unittest


def initialize_container(
        ocr_output_type: OCROutputType = None,
        override_args: dict = None) -> IocContainer:
    custom_args = {
        'learning_rate': 1e-3,
        'data_folder': os.path.join('tests', 'data'),
        'challenge': Challenge.OCREvaluation,
        'configuration': Configuration.SkipGram,
        'language': Language.English,
        'output_folder': os.path.join('tests', 'results'),
        'ocr_output_type': ocr_output_type,
        'seed': 42
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

    return container


class TestBaselineSkipGram(unittest.TestCase):

    # def test_baseline_convergence(self):
    #     # Skip Gram GT
    #     container_1 = initialize_container(ocr_output_type=OCROutputType.GroundTruth)

    #     # Create

    #     skip_gram_gt = self._create_model(
    #         container_1.arguments_service(),
    #         container_1.vocabulary_service(),
    #         container_1.data_service(),
    #         container_1.log_service(),
    #         ocr_output_type=OCROutputType.GroundTruth)

    #     # Load

    #     skip_gram_gt.load(
    #         path=os.path.join('results', 'ocr-evaluation', 'skip-gram', 'english'),
    #         name_prefix='BEST',
    #         name_suffix=None,
    #         load_model_dict=True,
    #         use_checkpoint_name=True,
    #         checkpoint_name=None)

    #     # Skip Gram OG
    #     container_2 = initialize_container(ocr_output_type=OCROutputType.GroundTruth)
    #     cache_service_2 = container_2.cache_service()
    #     arguments_service_2 = container_2.arguments_service()

    #     pretrained_matrix = cache_service_2.get_item_from_cache(
    #         CacheOptions(
    #             f'word-matrix-{arguments_service_2.get_dataset_string()}-{OCROutputType.GroundTruth.value}',
    #             seed_specific=True,
    #             seed=13))

    #     pretrained_matrix = pretrained_matrix.to('cuda')

    #     skip_gram_og = self._create_model(
    #         container_2.arguments_service(),
    #         container_2.vocabulary_service(),
    #         container_2.data_service(),
    #         container_2.log_service(),
    #         ocr_output_type=OCROutputType.GroundTruth,
    #         pretrained_matrix=pretrained_matrix)

    #     # Skip Gram BASE
    #     container_3 = initialize_container(ocr_output_type=OCROutputType.GroundTruth)

    #     skip_gram_base = self._create_model(
    #         container_3.arguments_service(),
    #         container_3.vocabulary_service(),
    #         container_3.data_service(),
    #         container_3.log_service(),
    #         ocr_output_type=OCROutputType.GroundTruth)

    #     skip_gram_base.load(
    #         path=os.path.join('results', 'ocr-evaluation', 'skip-gram', 'english'),
    #         name_prefix='BEST',
    #         name_suffix=None,
    #         load_model_dict=True,
    #         use_checkpoint_name=True,
    #         checkpoint_name=None,
    #         overwrite_args={
    #             'initialize_randomly': True,
    #             'configuration': Configuration.SkipGram.value,
    #             'learning_rate': 1e-3,
    #             'minimal_occurrence_limit': 5
    #         })

    #     print(f'GT Context mean: {skip_gram_gt._embeddings_context.weight.mean()}')
    #     print(f'GT Input mean: {skip_gram_gt._embeddings_input.weight.mean()}')

    #     print(f'OG Context mean: {skip_gram_og._embeddings_context.weight.mean()}')
    #     print(f'OG Input mean: {skip_gram_og._embeddings_input.weight.mean()}')

    #     print(f'Base Context mean: {skip_gram_base._embeddings_context.weight.mean()}')
    #     print(f'Base Input mean: {skip_gram_base._embeddings_input.weight.mean()}')

    #     a = 0

    def test_copy_vs_deepcopy(self):

        # Skip Gram BASE
        container_no_copy = initialize_container(ocr_output_type=OCROutputType.GroundTruth)

        skip_gram_no_copy = self._create_model(
            container_no_copy.arguments_service(),
            container_no_copy.vocabulary_service(),
            container_no_copy.data_service(),
            container_no_copy.log_service(),
            ocr_output_type=OCROutputType.GroundTruth)

        skip_gram_no_copy.load(
            path=os.path.join('results', 'ocr-evaluation', 'skip-gram', 'english'),
            name_prefix='BEST',
            name_suffix=None,
            load_model_dict=True,
            use_checkpoint_name=True,
            checkpoint_name=None,
            overwrite_args={
                'initialize_randomly': True,
                'configuration': Configuration.SkipGram.value,
                'learning_rate': 1e-3,
                'minimal_occurrence_limit': 5,
                'checkpoint_name': 'local-no-deep-copy',
            })

        # Skip Gram DEEPCOPY
        container_copy = initialize_container(ocr_output_type=OCROutputType.GroundTruth)

        skip_gram_deepcopy = self._create_model(
            container_copy.arguments_service(),
            container_copy.vocabulary_service(),
            container_copy.data_service(),
            container_copy.log_service(),
            ocr_output_type=OCROutputType.GroundTruth)

        skip_gram_deepcopy.load(
            path=os.path.join('results', 'ocr-evaluation', 'skip-gram', 'english'),
            name_prefix='BEST',
            name_suffix=None,
            load_model_dict=True,
            use_checkpoint_name=True,
            checkpoint_name=None,
            overwrite_args={
                'initialize_randomly': True,
                'configuration': Configuration.SkipGram.value,
                'learning_rate': 1e-3,
                'minimal_occurrence_limit': 5,
                'checkpoint_name': 'local',
            })
            
        # Skip Gram DEEPCOPY
        container_uniform = initialize_container(ocr_output_type=OCROutputType.GroundTruth)

        skip_gram_uniform = self._create_model(
            container_uniform.arguments_service(),
            container_uniform.vocabulary_service(),
            container_uniform.data_service(),
            container_uniform.log_service(),
            ocr_output_type=OCROutputType.GroundTruth)

        skip_gram_uniform.load(
            path=os.path.join('results', 'ocr-evaluation', 'skip-gram', 'english'),
            name_prefix='BEST',
            name_suffix=None,
            load_model_dict=True,
            use_checkpoint_name=True,
            checkpoint_name=None,
            overwrite_args={
                'initialize_randomly': True,
                'configuration': Configuration.SkipGram.value,
                'learning_rate': 1e-3,
                'minimal_occurrence_limit': 5,
                'checkpoint_name': 'local-deep-copy-uniform',
            })

        print(f'No deepcopy Context mean: {skip_gram_no_copy._embeddings_context.weight.mean()}')
        print(f'No deepcopy Input mean:   {skip_gram_no_copy._embeddings_input.weight.mean()}')

        print(f'Deepcopy Context mean: {skip_gram_deepcopy._embeddings_context.weight.mean()}')
        print(f'Deepcopy Input mean:   {skip_gram_deepcopy._embeddings_input.weight.mean()}')

        print(f'Deepcopy + uniform Context mean: {skip_gram_uniform._embeddings_context.weight.mean()}')
        print(f'Deepcopy + uniform Input mean:   {skip_gram_uniform._embeddings_input.weight.mean()}')

        a = 0

    def _create_model(
        self,
        arguments_service,
        vocabulary_service,
        data_service,
        log_service,
        ocr_output_type: OCROutputType,
        pretrained_matrix = None):
        result = SkipGram(
            arguments_service=arguments_service,
            vocabulary_service=vocabulary_service,
            data_service=data_service,
            log_service=log_service,
            pretrained_matrix=pretrained_matrix,
            ocr_output_type=ocr_output_type)

        return result

if __name__ == '__main__':
    unittest.main()
