from services.word_neighbourhood_service import WordNeighbourhoodService
from gensim.models.keyedvectors import Vocab
from gensim.utils import tokenize
from torch.utils import data
from models.joint_model import JointModel
import torch
import numpy as np
import random

import dependency_injector.containers as containers
import dependency_injector.providers as providers

import main

from enums.configuration import Configuration
from enums.challenge import Challenge
from enums.pretrained_model import PretrainedModel

from losses.loss_base import LossBase
from losses.joint_loss import JointLoss
from losses.transformer_loss_base import TransformerLossBase
from losses.cross_entropy_loss import CrossEntropyLoss

from models.model_base import ModelBase
from models.transformers.bert import BERT
from models.transformers.xlnet import XLNet
from models.transformers.bart import BART
from models.simple.cbow import CBOW

from optimizers.optimizer_base import OptimizerBase
from optimizers.adam_optimizer import AdamOptimizer
from optimizers.adamw_optimizer import AdamWOptimizer
from optimizers.sgd_optimizer import SGDOptimizer
from optimizers.adamw_transformer_optimizer import AdamWTransformerOptimizer
from optimizers.joint_adamw_transformer_optimizer import JointAdamWTransformerOptimizer

from services.arguments.ocr_quality_arguments_service import OCRQualityArgumentsService
from services.arguments.ocr_evaluation_arguments_service import OCREvaluationArgumentsService
from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.arguments.pretrained_arguments_service import PretrainedArgumentsService

from services.download.ocr_download_service import OCRDownloadService

from services.process.process_service_base import ProcessServiceBase
from services.process.transformer_process_service import TransformerProcessService
from services.process.word2vec_process_service import Word2VecProcessService
from services.process.evaluation_process_service import EvaluationProcessService

from services.evaluation.base_evaluation_service import BaseEvaluationService

from services.data_service import DataService
from services.dataloader_service import DataLoaderService
from services.dataset_service import DatasetService
from services.file_service import FileService
from services.log_service import LogService
from services.mask_service import MaskService
from services.metrics_service import MetricsService
from services.model_service import ModelService
from services.test_service import TestService

from services.tokenize.base_tokenize_service import BaseTokenizeService
from services.tokenize.bert_tokenize_service import BERTTokenizeService
from services.tokenize.xlnet_tokenize_service import XLNetTokenizeService
from services.tokenize.bart_tokenize_service import BARTTokenizeService
from services.tokenize.camembert_tokenize_service import CamembertTokenizeService
from services.tokenize.cbow_tokenize_service import CBOWTokenizeService

from services.train_service import TrainService
from services.vocabulary_service import VocabularyService
from services.plot_service import PlotService
from services.experiments.experiment_service_base import ExperimentServiceBase
from services.experiments.ocr_quality_experiment_service import OCRQualityExperimentService
from services.cache_service import CacheService
from services.string_process_service import StringProcessService

import logging


def get_arguments_service(arguments_service: ArgumentsServiceBase):
    result = 'base'
    challenge = arguments_service.challenge
    run_experiments = arguments_service.run_experiments

    if challenge == Challenge.OCREvaluation:
        if run_experiments:
            result = 'evaluation'
        else:
            result = 'ocr_quality'

    return result


def get_optimizer(arguments_service: ArgumentsServiceBase):
    if arguments_service.evaluate or arguments_service.run_experiments:
        return None

    result = None
    challenge = arguments_service.challenge
    configuration = arguments_service.configuration
    if challenge == Challenge.OCREvaluation:
        if configuration == Configuration.CBOW:
            result = 'cbow'
        else:
            result = 'transformer'

    return result


def get_loss_function(arguments_service: ArgumentsServiceBase):
    loss_function = None
    challenge = arguments_service.challenge
    configuration = arguments_service.configuration

    if challenge == Challenge.OCREvaluation:
        if configuration == Configuration.CBOW:
            return 'cbow'
        else:
            return 'transformer'

    return loss_function


# def register_evaluation_service(
#         arguments_service: ArgumentsServiceBase,
#         file_service: FileService,
#         plot_service: PlotService,
#         metrics_service: MetricsService,
#         process_service: ProcessServiceBase,
#         vocabulary_service: VocabularyService,
#         data_service: DataService,
#         joint_model: bool,
#         configuration: Configuration):
#     evaluation_service = None

#     return evaluation_service


def get_model_type(arguments_service: ArgumentsServiceBase):

    run_experiments = arguments_service.run_experiments
    configuration = arguments_service.configuration

    model = None

    if run_experiments:
        model = 'joint'
    else:
        model = configuration.value

    return model


def get_process_service(arguments_service: ArgumentsServiceBase):
    result = None

    challenge = arguments_service.challenge
    run_experiments = arguments_service.run_experiments
    configuration = arguments_service.configuration

    if challenge == Challenge.OCREvaluation:
        if run_experiments:
            result = 'evaluation'
        elif configuration == Configuration.CBOW:
            result = 'cbow'
        else:
            result = 'transformer'

    return result


def get_tokenize_service(arguments_service: ArgumentsServiceBase) -> str:
    pretrained_model_type = None
    if isinstance(arguments_service, PretrainedArgumentsService):
        pretrained_model_type = arguments_service.pretrained_model

    if pretrained_model_type is None:
        configuration = arguments_service.configuration
        return configuration.value

    return pretrained_model_type.value


def get_experiment_service(arguments_service: ArgumentsServiceBase):

    run_experiments = arguments_service.run_experiments

    if not run_experiments:
        return 'base'

    return 'ocr_quality'


class IocContainer(containers.DeclarativeContainer):
    """Application IoC container."""

    logger = providers.Singleton(logging.Logger, name='example')

    # Services

    arguments_service_base = providers.Singleton(
        ArgumentsServiceBase,
        raise_errors_on_invalid_args=False)

    argument_service_selector = providers.Callable(
        get_arguments_service,
        arguments_service=arguments_service_base)

    arguments_service: providers.Provider[ArgumentsServiceBase] = providers.Selector(
        argument_service_selector,
        base=providers.Singleton(ArgumentsServiceBase),
        evaluation=providers.Singleton(OCREvaluationArgumentsService),
        ocr_quality=providers.Singleton(OCRQualityArgumentsService))

    log_service = providers.Singleton(
        LogService,
        arguments_service=arguments_service,
        external_logging_enabled=False  # external_logging_enabled
    )

    data_service = providers.Factory(DataService)

    file_service = providers.Factory(
        FileService,
        arguments_service=arguments_service
    )

    cache_service = providers.Singleton(
        CacheService,
        arguments_service=arguments_service,
        file_service=file_service,
        data_service=data_service)

    plot_service = providers.Factory(
        PlotService,
        data_service=data_service
    )

    vocabulary_service: providers.Provider[VocabularyService] = providers.Singleton(
        VocabularyService,
        data_service=data_service,
        file_service=file_service,
        cache_service=cache_service
    )

    tokenize_service_selector = providers.Callable(
        get_tokenize_service,
        arguments_service=arguments_service)

    tokenize_service: providers.Provider[BaseTokenizeService] = providers.Selector(
        tokenize_service_selector,
        bert=providers.Singleton(
            BERTTokenizeService,
            arguments_service=arguments_service),
        xlnet=providers.Singleton(
            XLNetTokenizeService,
            arguments_service=arguments_service),
        bart=providers.Singleton(
            BARTTokenizeService,
            arguments_service=arguments_service),
        camembert=providers.Singleton(
            CamembertTokenizeService,
            arguments_service=arguments_service),
        cbow=providers.Singleton(
            CBOWTokenizeService,
            vocabulary_service=vocabulary_service))

    mask_service = providers.Factory(
        MaskService,
        tokenize_service=tokenize_service,
        arguments_service=arguments_service
    )

    metrics_service = providers.Factory(MetricsService)

    string_process_service = providers.Factory(StringProcessService)

    ocr_download_service = providers.Factory(
        OCRDownloadService,
        data_service=data_service,
        string_process_service=string_process_service,
        cache_service=cache_service)

    process_service_selector = providers.Callable(
        get_process_service,
        arguments_service=arguments_service)

    process_service: providers.Provider[ProcessServiceBase] = providers.Selector(
        process_service_selector,
        evaluation=providers.Singleton(
            EvaluationProcessService,
            arguments_service=arguments_service,
            cache_service=cache_service,
            log_service=log_service,
            vocabulary_service=vocabulary_service,
            tokenize_service=tokenize_service),
        cbow=providers.Singleton(
            Word2VecProcessService,
            arguments_service=arguments_service,
            ocr_download_service=ocr_download_service,
            cache_service=cache_service,
            log_service=log_service,
            vocabulary_service=vocabulary_service,
            file_service=file_service,
            tokenize_service=tokenize_service),
        transformer=providers.Singleton(
            TransformerProcessService,
            arguments_service=arguments_service,
            ocr_download_service=ocr_download_service,
            tokenize_service=tokenize_service,
            cache_service=cache_service,
            log_service=log_service))

    dataset_service = providers.Factory(
        DatasetService,
        arguments_service=arguments_service,
        mask_service=mask_service,
        tokenize_service=tokenize_service,
        file_service=file_service,
        log_service=log_service,
        vocabulary_service=vocabulary_service,
        metrics_service=metrics_service,
        data_service=data_service,
        process_service=process_service,
    )

    dataloader_service = providers.Factory(
        DataLoaderService,
        arguments_service=arguments_service,
        dataset_service=dataset_service)

    model_service = providers.Factory(
        ModelService,
        arguments_service=arguments_service,
        data_service=data_service,
        vocabulary_service=vocabulary_service,
        process_service=process_service,
        file_service=file_service)

    model_selector = providers.Callable(
        get_model_type,
        arguments_service=arguments_service)

    model: providers.Provider[ModelBase] = providers.Selector(
        model_selector,
        joint=providers.Singleton(
            JointModel,
            arguments_service=arguments_service,
            data_service=data_service,
            vocabulary_service=vocabulary_service),
        bert=providers.Singleton(
            BERT,
            arguments_service=arguments_service,
            data_service=data_service),
        xlnet=providers.Singleton(
            XLNet,
            arguments_service=arguments_service,
            data_service=data_service),
        bart=providers.Singleton(
            BART,
            arguments_service=arguments_service,
            data_service=data_service),
        cbow=providers.Singleton(
            CBOW,
            arguments_service=arguments_service,
            process_service=process_service,
            data_service=data_service,
            vocabulary_service=vocabulary_service))

    loss_selector = providers.Callable(
        get_loss_function,
        arguments_service=arguments_service)

    loss_function: providers.Provider[LossBase] = providers.Selector(
        loss_selector,
        cbow=providers.Singleton(CrossEntropyLoss),
        transformer=providers.Singleton(TransformerLossBase))

    optimizer_selector = providers.Callable(
        get_optimizer,
        arguments_service=arguments_service)

    optimizer: providers.Provider[OptimizerBase] = providers.Selector(
        optimizer_selector,
        cbow=providers.Singleton(
            SGDOptimizer,
            arguments_service=arguments_service,
            model=model),
        transformer=providers.Singleton(
            AdamWTransformerOptimizer,
            arguments_service=arguments_service,
            model=model))

    evaluation_service = None
    # evaluation_service = register_evaluation_service(
    #     arguments_service=arguments_service,
    #     file_service=file_service,
    #     plot_service=plot_service,
    #     metrics_service=metrics_service,
    #     process_service=process_service,
    #     vocabulary_service=vocabulary_service,
    #     data_service=data_service,
    #     joint_model=joint_model,
    #     configuration=configuration)

    word_neighbourhood_service = providers.Factory(
        WordNeighbourhoodService,
        arguments_service=arguments_service,
        metrics_service=metrics_service,
        plot_service=plot_service,
        file_service=file_service)

    experiment_service_selector = providers.Callable(
        get_experiment_service,
        arguments_service=arguments_service)

    experiment_service = providers.Selector(
        experiment_service_selector,
        ocr_quality=providers.Factory(
            OCRQualityExperimentService,
            arguments_service=arguments_service,
            dataloader_service=dataloader_service,
            file_service=file_service,
            metrics_service=metrics_service,
            plot_service=plot_service,
            cache_service=cache_service,
            word_neighbourhood_service=word_neighbourhood_service,
            model=model),
        base=providers.Factory(
            ExperimentServiceBase,
            arguments_service=arguments_service,
            dataloader_service=dataloader_service,
            file_service=file_service,
            model=model))

    test_service = providers.Factory(
        TestService,
        arguments_service=arguments_service,
        dataloader_service=dataloader_service,
        evaluation_service=evaluation_service,
        file_service=file_service,
        model=model
    )

    train_service = providers.Factory(
        TrainService,
        arguments_service=arguments_service,
        dataloader_service=dataloader_service,
        loss_function=loss_function,
        optimizer=optimizer,
        log_service=log_service,
        model=model,
        file_service=file_service
    )

    # Misc

    main = providers.Callable(
        main.main,
        data_service=data_service,
        arguments_service=arguments_service,
        train_service=train_service,
        test_service=test_service,
        experiment_service=experiment_service
    )
