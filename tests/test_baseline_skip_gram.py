from models.transformers.bert import BERT
from tests.entities.embedding_configuration import EmbeddingConfiguration
from typing import List
from models.model_base import ModelBase
from services.vocabulary_service import VocabularyService
import numpy as np
import pickle
from models.simple.skip_gram import SkipGram
from models.simple.cbow import CBOW
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
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from scipy import spatial
import csv
import pandas as pd

import tests.constants.embedding_models as embedding_models

def initialize_container(
    configuration: Configuration,
    ocr_output_type: OCROutputType,
    language: Language,
    seed: int = 13,
    override_args: dict = None) -> IocContainer:
    custom_args = {
        'learning_rate': 1e-3,
        'data_folder': os.path.join('tests', 'data'),
        'challenge': Challenge.OCREvaluation,
        'configuration': configuration,
        'language': language,
        'output_folder': os.path.join('tests', 'results'),
        'ocr_output_type': ocr_output_type,
        'seed': seed
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

def calculate_context_words(
    configuration: Configuration,
    vocabulary_service: VocabularyService,
    model: ModelBase,
    target_word: str,
    neighbourhood_set_size: int = 50) -> List[str]:
    target_id = vocabulary_service.string_to_id(target_word)

    target_embeddings = model.get_embeddings([target_word])

    all_embeddings = None
    if configuration == Configuration.SkipGram:
        all_embeddings = list(model._embedding_layer._embeddings_target.parameters())[0].detach().cpu().tolist()
    elif configuration == Configuration.CBOW:
        all_embeddings = list(model._embeddings.parameters())[0].detach().cpu().tolist()

    similarities = []
    for j in range(len(all_embeddings)):
        print(f'Processing {target_word} - {j}/{len(all_embeddings)}            \r', end='')
        if j == target_id:
            assert all_embeddings[j] == target_embeddings

        # similarity = cosine_similarity(target_embeddings, all_embeddings[j])
        similarity = 1 - spatial.distance.cosine(target_embeddings, all_embeddings[j])
        similarities.append(similarity)

    indices = np.argsort(similarities)[::-1]
    sorted_similarities = [similarities[x] for x in indices]

    assert sorted_similarities[-2] < sorted_similarities[1]
    sorted_words = vocabulary_service.ids_to_strings(indices)

    return sorted_words[:neighbourhood_set_size]

def initialize_model(
    arguments_service,
    vocabulary_service,
    data_service,
    log_service,
    tokenize_service,
    ocr_output_type: OCROutputType,
    language: Language,
    configuration: Configuration,
    initialize_randomly: bool,
    learning_rate: float):

    model = create_model(
        configuration,
        arguments_service,
        vocabulary_service,
        data_service,
        log_service,
        tokenize_service,
        ocr_output_type=ocr_output_type)

    model.load(
        path=os.path.join('results', 'ocr-evaluation', configuration.value, language.value),
        name_prefix='BEST',
        name_suffix=None,
        load_model_dict=True,
        use_checkpoint_name=True,
        checkpoint_name=None,
        overwrite_args={
            'initialize_randomly': initialize_randomly,
            'configuration': configuration.value,
            'learning_rate': learning_rate,
            'minimal_occurrence_limit': 5 if configuration != Configuration.BERT else None,
            # 'checkpoint_name': 'local-test-pre',
        })

    return model

def create_model(
    configuration: Configuration,
    arguments_service,
    vocabulary_service,
    data_service,
    log_service,
    tokenize_service,
    ocr_output_type: OCROutputType,
    pretrained_matrix = None):
    if configuration == Configuration.SkipGram:
        result = SkipGram(
            arguments_service=arguments_service,
            vocabulary_service=vocabulary_service,
            data_service=data_service,
            log_service=log_service,
            pretrained_matrix=pretrained_matrix,
            ocr_output_type=ocr_output_type)
    elif configuration == Configuration.CBOW:
        result = CBOW(
            arguments_service=arguments_service,
            vocabulary_service=vocabulary_service,
            data_service=data_service,
            log_service=log_service,
            pretrained_matrix=pretrained_matrix,
            ocr_output_type=ocr_output_type)
    elif configuration == Configuration.CBOW:
        result = BERT(
            arguments_service=arguments_service,
            data_service=data_service,
            log_service=log_service,
            tokenize_service=tokenize_service,
            overwrite_initialization=False)

    return result

target_words = {
    Language.English: ['man', 'new', 'time', 'day', 'good', 'old', 'little', 'one', 'two', 'three'],
    Language.Dutch: ['man', 'jaar', 'tijd', 'mensen', 'dag', 'huis', 'dier', 'afbeelding', 'werk', 'naam', 'groot', 'kleine', 'twee', 'drie', 'vier', 'vijf']
}

def log_neighbourhoods(
    vocabulary_service: VocabularyService,
    model: ModelBase,
    embedding_configuration: EmbeddingConfiguration,
    output_folder: str):
    for target_word in target_words[embedding_configuration.language]:
        context_words = calculate_context_words(embedding_configuration.configuration, vocabulary_service, model, target_word)
        save_context_words('skip-gram-base', target_word, context_words, output_folder, embedding_configuration)


def save_context_words(
    model_name: str,
    target_word: str,
    context_words: List[str],
    output_folder: str,
    embedding_configuration: EmbeddingConfiguration):
    csv_fieldnames = ['language', 'configuration', 'randomly_initialized', 'ocr_type', 'learning_rate', 'target_word', 'context_words']
    file_path = os.path.join(output_folder, 'context-words.csv')

    init_header = not os.path.exists(file_path)

    with open(file_path, 'a', encoding='utf-8') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        if init_header:
            csv_writer.writeheader()

        csv_writer.writerow({
            'language': embedding_configuration.language,
            'configuration': embedding_configuration.configuration,
            'randomly_initialized': embedding_configuration.initialize_randomly,
            'learning_rate': str(embedding_configuration.lr),
            # 'configuration': embedding_configuration.lr,
            'ocr_type': embedding_configuration.ocr_output_type,
            'target_word': target_word,
            'context_words': ', '.join(context_words)
        })

def log_embedding_layers(model):
    print(f'Base Context mean: {model._embedding_layer._embeddings_context.weight.mean()}')
    print(f'Base Input mean: {model._embedding_layer._embeddings_target.weight.mean()}')


class TestBaselineSkipGram(unittest.TestCase):
    def test_baseline_convergence(self):
        output_folder = os.path.join('tests', 'results')
        file_path = os.path.join(output_folder, 'context-words.csv')
        if os.path.exists(file_path):
            os.remove(file_path)

        for language, configurations in embedding_models.configurations.items():
            for configuration, lrs in configurations.items():
                for lr, initialize_randomly_to_output_types in lrs.items():
                    for initialize_randomly, output_types in initialize_randomly_to_output_types.items():
                        for ocr_output_type in output_types:
                            # if configuration != Configuration.SkipGram:
                            #     continue

                            container_base = initialize_container(configuration, ocr_output_type, language)

                            vocabulary_service = container_base.vocabulary_service()
                            arguments_service = container_base.arguments_service()
                            skip_gram_base = initialize_model(
                                arguments_service,
                                vocabulary_service,
                                container_base.data_service(),
                                container_base.log_service(),
                                container_base.tokenize_service(),
                                ocr_output_type=ocr_output_type,
                                language=language,
                                configuration=configuration,
                                initialize_randomly=initialize_randomly,
                                learning_rate=lr)

                            # log_embedding_layers(skip_gram_base)

                            embedding_configuration = EmbeddingConfiguration(language, configuration, lr, initialize_randomly, ocr_output_type)
                            log_neighbourhoods(vocabulary_service, skip_gram_base, embedding_configuration, output_folder=output_folder)

if __name__ == '__main__':
    unittest.main()

