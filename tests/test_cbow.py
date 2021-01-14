from tests.fakes.log_service_fake import LogServiceFake
from enums.language import Language
from enums.configuration import Configuration
from enums.challenge import Challenge
from enums.ocr_output_type import OCROutputType
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
        'data_folder': os.path.join('tests', 'data'),
        'challenge': Challenge.OCREvaluation,
        'configuration': Configuration.CBOW,
        'language': Language.English,
        'output_folder': os.path.join('tests', 'results'),
        'ocr_output_type': ocr_output_type
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


class TestCBOW(unittest.TestCase):

    def test_embedding_matrix_english_initialization(self):
        main_container = initialize_container()
        metrics_service = main_container.metrics_service()

        # Raw model
        container_1 = initialize_container(ocr_output_type=OCROutputType.Raw)
        vocabulary_service_1 = container_1.vocabulary_service()

        tokens_1 = vocabulary_service_1.get_vocabulary_tokens()

        ids_1 = [id for id, _ in tokens_1]
        ids_tensor_1 = torch.Tensor(ids_1).long()

        model_1 = container_1.model()
        embeddings_1 = model_1.get_embeddings(ids_tensor_1)

        # Ground truth model
        container_2 = initialize_container(
            ocr_output_type=OCROutputType.GroundTruth)
        vocabulary_service_2 = container_2.vocabulary_service()

        tokens_2 = vocabulary_service_2.get_vocabulary_tokens()

        ids_2 = [id for id, _ in tokens_2]
        ids_tensor_2 = torch.Tensor(ids_2).long()

        model_2 = container_2.model()
        embeddings_2 = model_2.get_embeddings(ids_tensor_2)

        # Assert
        for embedding_1, embedding_2 in zip(embeddings_1, embeddings_2):
            self.assertEqual(embedding_1, embedding_2)
            self.assertEqual(metrics_service.calculate_cosine_distance(
                embedding_1, embedding_2), 0.0)

    def test_embedding_matrix_dutch_initialization(self):
        main_container = initialize_container(
            override_args={'language': Language.Dutch})

        metrics_service = main_container.metrics_service()

        # Raw model
        container_1 = initialize_container(
            ocr_output_type=OCROutputType.Raw,
            override_args={'language': Language.Dutch})

        vocabulary_service_1 = container_1.vocabulary_service()

        tokens_1 = vocabulary_service_1.get_vocabulary_tokens()

        ids_1 = [id for id, _ in tokens_1]
        ids_tensor_1 = torch.Tensor(ids_1).long()

        model_1 = container_1.model()
        embeddings_1 = model_1.get_embeddings(ids_tensor_1)

        # Ground truth model
        container_2 = initialize_container(
            ocr_output_type=OCROutputType.GroundTruth,
            override_args={'language': Language.Dutch})

        vocabulary_service_2 = container_2.vocabulary_service()

        tokens_2 = vocabulary_service_2.get_vocabulary_tokens()

        ids_2 = [id for id, _ in tokens_2]
        ids_tensor_2 = torch.Tensor(ids_2).long()

        model_2 = container_2.model()
        embeddings_2 = model_2.get_embeddings(ids_tensor_2)

        # Assert
        for embedding_1, embedding_2 in zip(embeddings_1, embeddings_2):
            self.assertEqual(embedding_1, embedding_2)
            self.assertEqual(metrics_service.calculate_cosine_distance(
                embedding_1, embedding_2), 0.0)

    def test_embedding_matrix_same_different_seeds(self):
        main_container = initialize_container()
        metrics_service = main_container.metrics_service()

        # Raw model
        container_1 = initialize_container(
            ocr_output_type=OCROutputType.Raw,
            override_args={
                'seed': 13
            })

        vocabulary_service_1 = container_1.vocabulary_service()

        tokens_1 = vocabulary_service_1.get_vocabulary_tokens()

        ids_1 = [id for id, _ in tokens_1]
        ids_tensor_1 = torch.Tensor(ids_1).long()

        model_1 = container_1.model()
        embeddings_1 = model_1.get_embeddings(ids_tensor_1)

        # Ground truth model
        container_2 = initialize_container(
            ocr_output_type=OCROutputType.Raw,
            override_args={
                'seed': 42
            })

        vocabulary_service_2 = container_2.vocabulary_service()

        tokens_2 = vocabulary_service_2.get_vocabulary_tokens()

        ids_2 = [id for id, _ in tokens_2]
        ids_tensor_2 = torch.Tensor(ids_2).long()

        model_2 = container_2.model()
        embeddings_2 = model_2.get_embeddings(ids_tensor_2)

        # Assert
        for embedding_1, embedding_2 in zip(embeddings_1, embeddings_2):
            self.assertEqual(embedding_1, embedding_2)
            self.assertEqual(metrics_service.calculate_cosine_distance(
                embedding_1, embedding_2), 0.0)


if __name__ == '__main__':
    unittest.main()
