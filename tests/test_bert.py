from tests.fakes.log_service_fake import LogServiceFake
from enums.language import Language
from enums.configuration import Configuration
from enums.challenge import Challenge
from enums.ocr_output_type import OCROutputType
from enums.pretrained_model import PretrainedModel
import os
from tests.fakes.non_context_service_fake import NonContextServiceFake
from dependency_injection.ioc_container import IocContainer
import dependency_injector.providers as providers
import torch
import unittest


def initialize_container(ocr_output_type: OCROutputType = None, override_args: dict = None) -> IocContainer:
    custom_args = {
        'data_folder': 'data',
        'challenge': Challenge.OCREvaluation,
        'configuration': Configuration.BERT,
        'language': Language.English,
        'output_folder': os.path.join('tests', 'results'),
        'ocr_output_type': ocr_output_type,
        'include_pretrained_model': True,
        'pretrained_weights': 'bert-base-cased',
        'pretrained_model_size': 768,
        'pretrained_max_length': 512,
        'pretrained_model': PretrainedModel.BERT,
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


class TestBERT(unittest.TestCase):

    def test_embedding_matrix_english_initialization(self):
        tokens = ['test', 'token', 'bert', 'vocabulary', 'units', 'python']

        main_container = initialize_container()
        metrics_service = main_container.metrics_service()

        # Raw model
        container_1 = initialize_container(ocr_output_type=OCROutputType.Raw)

        tokenize_service_1 = container_1.tokenize_service()
        encoded_sequences_1 = [
            tokenize_service_1.encode_sequence(token) for token in tokens]
        ids_1 = [torch.Tensor(ids) for ids, _, _, _ in encoded_sequences_1]
        ids_tensor_1 = torch.nn.utils.rnn.pad_sequence(
            ids_1, batch_first=True, padding_value=0).long()

        model_1 = container_1.model()
        word_evaluations_1 = model_1.get_embeddings(tokens, ids_tensor_1)

        # Ground truth model
        container_2 = initialize_container(
            ocr_output_type=OCROutputType.GroundTruth)

        tokenize_service_2 = container_2.tokenize_service()
        encoded_sequences_2 = [
            tokenize_service_2.encode_sequence(token) for token in tokens]
        ids_2 = [torch.Tensor(ids) for ids, _, _, _ in encoded_sequences_2]
        ids_tensor_2 = torch.nn.utils.rnn.pad_sequence(
            ids_2, batch_first=True, padding_value=0).long()

        model_2 = container_2.model()
        word_evaluations_2 = model_2.get_embeddings(tokens, ids_tensor_2)

        # Assert
        for word_evaluation_1, word_evaluation_2 in zip(word_evaluations_1, word_evaluations_2):
            self.assertEqual(word_evaluation_1.get_embeddings(
                0), word_evaluation_2.get_embeddings(0))
            self.assertEqual(metrics_service.calculate_cosine_distance(
                word_evaluation_1.get_embeddings(0), word_evaluation_2.get_embeddings(0)), 0.0)

    def test_embedding_matrix_dutch_initialization(self):
        override_args = {
            'language': Language.Dutch,
            'pretrained_weights': 'wietsedv/bert-base-dutch-cased'
        }

        tokens = ['test', 'token', 'bert', 'vocabulary', 'units', 'python']
        main_container = initialize_container(
            override_args=override_args)

        metrics_service = main_container.metrics_service()

        # Raw model
        container_1 = initialize_container(
            ocr_output_type=OCROutputType.Raw,
            override_args=override_args)

        tokenize_service_1 = container_1.tokenize_service()
        encoded_sequences_1 = [
            tokenize_service_1.encode_sequence(token) for token in tokens]
        ids_1 = [torch.Tensor(ids) for ids, _, _, _ in encoded_sequences_1]
        ids_tensor_1 = torch.nn.utils.rnn.pad_sequence(
            ids_1, batch_first=True, padding_value=0).long()

        model_1 = container_1.model()
        word_evaluations_1 = model_1.get_embeddings(tokens, ids_tensor_1)

        # Ground truth model
        container_2 = initialize_container(
            ocr_output_type=OCROutputType.GroundTruth,
            override_args=override_args)

        tokenize_service_2 = container_2.tokenize_service()
        encoded_sequences_2 = [
            tokenize_service_2.encode_sequence(token) for token in tokens]
        ids_2 = [torch.Tensor(ids) for ids, _, _, _ in encoded_sequences_2]
        ids_tensor_2 = torch.nn.utils.rnn.pad_sequence(
            ids_2, batch_first=True, padding_value=0).long()

        model_2 = container_2.model()
        word_evaluations_2 = model_2.get_embeddings(tokens, ids_tensor_2)

        # Assert
        for word_evaluation_1, word_evaluation_2 in zip(word_evaluations_1, word_evaluations_2):
            self.assertEqual(word_evaluation_1.get_embeddings(
                0), word_evaluation_2.get_embeddings(0))
            self.assertEqual(metrics_service.calculate_cosine_distance(
                word_evaluation_1.get_embeddings(0), word_evaluation_2.get_embeddings(0)), 0.0)

    def test_embedding_matrix_same_different_seed(self):
        tokens = ['test', 'token', 'bert', 'vocabulary', 'units', 'python']
        main_container = initialize_container()
        metrics_service = main_container.metrics_service()

        # Raw model
        container_1 = initialize_container(
            ocr_output_type=OCROutputType.Raw,
            override_args={
                'seed': 13
            })

        tokenize_service_1 = container_1.tokenize_service()
        encoded_sequences_1 = [
            tokenize_service_1.encode_sequence(token) for token in tokens]
        ids_1 = [torch.Tensor(ids) for ids, _, _, _ in encoded_sequences_1]
        ids_tensor_1 = torch.nn.utils.rnn.pad_sequence(
            ids_1, batch_first=True, padding_value=0).long()

        model_1 = container_1.model()
        word_evaluations_1 = model_1.get_embeddings(tokens, ids_tensor_1)

        # Ground truth model
        container_2 = initialize_container(
            ocr_output_type=OCROutputType.Raw,
            override_args={
                'seed': 42
            })

        tokenize_service_2 = container_2.tokenize_service()
        encoded_sequences_2 = [
            tokenize_service_2.encode_sequence(token) for token in tokens]
        ids_2 = [torch.Tensor(ids) for ids, _, _, _ in encoded_sequences_2]
        ids_tensor_2 = torch.nn.utils.rnn.pad_sequence(
            ids_2, batch_first=True, padding_value=0).long()

        model_2 = container_2.model()
        word_evaluations_2 = model_2.get_embeddings(tokens, ids_tensor_2)

        # Assert
        for word_evaluation_1, word_evaluation_2 in zip(word_evaluations_1, word_evaluations_2):
            self.assertEqual(
                word_evaluation_1.get_embeddings(0),
                word_evaluation_2.get_embeddings(0))

            self.assertEqual(
                metrics_service.calculate_cosine_distance(
                    word_evaluation_1.get_embeddings(0),
                    word_evaluation_2.get_embeddings(0)),
                0.0)


if __name__ == '__main__':
    unittest.main()
