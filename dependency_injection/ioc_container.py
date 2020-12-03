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


def initialize_seed(seed: int, device: str):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if device == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)


def get_argument_service_type(
        challenge: Challenge,
        configuration: Configuration,
        run_experiments: bool):
    argument_service_type = None

    if challenge == Challenge.OCREvaluation:
        if run_experiments:
            argument_service_type = OCREvaluationArgumentsService
        else:
            argument_service_type = OCRQualityArgumentsService
    else:
        raise Exception('Challenge not supported')

    return argument_service_type


def register_optimizer(
        joint_model: bool,
        evaluate: bool,
        run_experiments: bool,
        challenge: Challenge,
        configuration: Configuration,
        model: ModelBase,
        arguments_service: ArgumentsServiceBase):
    if evaluate or run_experiments:
        return None

    optimizer = None
    if challenge == Challenge.OCREvaluation:
        if configuration == Configuration.CBOW:
            optimizer = providers.Singleton(
                SGDOptimizer,
                arguments_service=arguments_service,
                model=model)
        else:
            optimizer = providers.Singleton(
                AdamWTransformerOptimizer,
                arguments_service=arguments_service,
                model=model)

    return optimizer


def register_loss(
        joint_model: bool,
        configuration: Configuration,
        challenge: Challenge,
        arguments_service: ArgumentsServiceBase):
    loss_function = None

    if challenge == Challenge.OCREvaluation:
        if configuration == Configuration.CBOW:
            loss_function = providers.Singleton(
                CrossEntropyLoss
            )
        else:
            loss_function = providers.Singleton(
                TransformerLossBase
            )

    return loss_function


def register_evaluation_service(
        arguments_service: ArgumentsServiceBase,
        file_service: FileService,
        plot_service: PlotService,
        metrics_service: MetricsService,
        process_service: ProcessServiceBase,
        vocabulary_service: VocabularyService,
        data_service: DataService,
        joint_model: bool,
        configuration: Configuration):
    evaluation_service = None

    return evaluation_service


def register_model(
        arguments_service: ArgumentsServiceBase,
        file_service: FileService,
        plot_service: PlotService,
        metrics_service: MetricsService,
        data_service: DataService,
        tokenize_service: BaseTokenizeService,
        log_service: LogService,
        vocabulary_service: VocabularyService,
        model_service: ModelService,
        process_service: ProcessServiceBase,
        joint_model: bool,
        configuration: Configuration,
        run_experiments: bool):
    model = None

    if run_experiments:
        model = providers.Singleton(
            JointModel,
            arguments_service=arguments_service,
            data_service=data_service,
            vocabulary_service=vocabulary_service)
    elif configuration == Configuration.BERT:
        model = providers.Singleton(
            BERT,
            arguments_service=arguments_service,
            data_service=data_service)
    elif configuration == Configuration.XLNet:
        model = providers.Singleton(
            XLNet,
            arguments_service=arguments_service,
            data_service=data_service)
    elif configuration == Configuration.BART:
        model = providers.Singleton(
            BART,
            arguments_service=arguments_service,
            data_service=data_service)
    elif configuration == Configuration.CBOW:
        model = providers.Singleton(
            CBOW,
            arguments_service=arguments_service,
            process_service=process_service,
            data_service=data_service,
            vocabulary_service=vocabulary_service)

    return model


def register_process_service(
        challenge: Challenge,
        configuration: Configuration,
        arguments_service: ArgumentsServiceBase,
        file_service: FileService,
        tokenize_service: BaseTokenizeService,
        vocabulary_service: VocabularyService,
        data_service: DataService,
        metrics_service: MetricsService,
        log_service: LogService,
        cache_service: CacheService,
        ocr_download_service: OCRDownloadService,
        string_process_service: StringProcessService,
        run_experiments: bool):
    process_service = None

    if challenge == Challenge.OCREvaluation:
        if run_experiments:
            process_service = providers.Singleton(
                EvaluationProcessService,
                arguments_service=arguments_service,
                cache_service=cache_service,
                log_service=log_service,
                vocabulary_service=vocabulary_service,
                tokenize_service=tokenize_service)
        elif configuration == Configuration.CBOW:
            process_service = providers.Singleton(
                Word2VecProcessService,
                arguments_service=arguments_service,
                ocr_download_service=ocr_download_service,
                cache_service=cache_service,
                log_service=log_service,
                vocabulary_service=vocabulary_service,
                file_service=file_service,
                tokenize_service=tokenize_service)
        else:
            process_service = providers.Singleton(
                TransformerProcessService,
                arguments_service=arguments_service,
                ocr_download_service=ocr_download_service,
                tokenize_service=tokenize_service,
                cache_service=cache_service,
                log_service=log_service)

    return process_service


def register_tokenize_service(
        arguments_service: ArgumentsServiceBase,
        vocabulary_service: VocabularyService,
        configuration: Configuration,
        pretrained_model_type: PretrainedModel):
    tokenize_service = None
    if pretrained_model_type == PretrainedModel.BERT:
        tokenize_service = providers.Singleton(
            BERTTokenizeService,
            arguments_service=arguments_service)
    if pretrained_model_type == PretrainedModel.XLNet:
        tokenize_service = providers.Singleton(
            XLNetTokenizeService,
            arguments_service=arguments_service)
    if pretrained_model_type == PretrainedModel.BART:
        tokenize_service = providers.Singleton(
            BARTTokenizeService,
            arguments_service=arguments_service)
    elif pretrained_model_type == PretrainedModel.CamemBERT:
        tokenize_service = providers.Singleton(
            CamembertTokenizeService,
            arguments_service=arguments_service)
    elif configuration == Configuration.CBOW:
        tokenize_service = providers.Singleton(
            CBOWTokenizeService,
            vocabulary_service=vocabulary_service)

    return tokenize_service


def register_experiment_service(
        arguments_service: ArgumentsServiceBase,
        dataloader_service: DataLoaderService,
        file_service: FileService,
        metrics_service: MetricsService,
        plot_service: PlotService,
        cache_service: CacheService,
        model: ModelBase,
        run_experiments: bool):

    if not run_experiments:
        return None

    experiment_service = providers.Factory(
        OCRQualityExperimentService,
        arguments_service=arguments_service,
        dataloader_service=dataloader_service,
        file_service=file_service,
        metrics_service=metrics_service,
        plot_service=plot_service,
        cache_service=cache_service,
        model=model
    )

    return experiment_service


class IocContainer(containers.DeclarativeContainer):
    """Application IoC container."""

    logger = providers.Singleton(logging.Logger, name='example')

    # Services

    arguments_service_base = PretrainedArgumentsService(
        raise_errors_on_invalid_args=False)

    challenge = arguments_service_base.challenge
    seed = arguments_service_base.seed
    device = arguments_service_base.device
    configuration = arguments_service_base.configuration
    joint_model = arguments_service_base.joint_model
    evaluate = arguments_service_base.evaluate
    run_experiments = arguments_service_base.run_experiments
    external_logging_enabled = arguments_service_base.enable_external_logging
    pretrained_model_type = arguments_service_base.pretrained_model

    argument_service_type = get_argument_service_type(
        challenge, configuration, run_experiments)
    arguments_service = providers.Singleton(
        argument_service_type
    )

    initialize_seed(seed, device)

    log_service = providers.Singleton(
        LogService,
        arguments_service=arguments_service,
        external_logging_enabled=external_logging_enabled
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

    vocabulary_service = providers.Singleton(
        VocabularyService,
        data_service=data_service,
        file_service=file_service,
        cache_service=cache_service
    )

    tokenize_service = register_tokenize_service(
        arguments_service=arguments_service,
        vocabulary_service=vocabulary_service,
        configuration=configuration,
        pretrained_model_type=pretrained_model_type)

    mask_service = providers.Factory(
        MaskService,
        tokenize_service=tokenize_service,
        arguments_service=arguments_service
    )

    metrics_service = providers.Factory(
        MetricsService
    )

    string_process_service = providers.Factory(
        StringProcessService
    )

    ocr_download_service = providers.Factory(
        OCRDownloadService,
        data_service=data_service,
        string_process_service=string_process_service,
        cache_service=cache_service)

    process_service = register_process_service(
        challenge,
        configuration,
        arguments_service=arguments_service,
        file_service=file_service,
        tokenize_service=tokenize_service,
        vocabulary_service=vocabulary_service,
        data_service=data_service,
        metrics_service=metrics_service,
        log_service=log_service,
        cache_service=cache_service,
        ocr_download_service=ocr_download_service,
        string_process_service=string_process_service,
        run_experiments=run_experiments)

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

    model = register_model(
        arguments_service=arguments_service,
        file_service=file_service,
        plot_service=plot_service,
        metrics_service=metrics_service,
        data_service=data_service,
        tokenize_service=tokenize_service,
        log_service=log_service,
        vocabulary_service=vocabulary_service,
        model_service=model_service,
        process_service=process_service,
        joint_model=joint_model,
        configuration=configuration,
        run_experiments=run_experiments)

    loss_function = register_loss(
        joint_model=joint_model,
        configuration=configuration,
        challenge=challenge,
        arguments_service=arguments_service)

    optimizer = register_optimizer(
        joint_model,
        evaluate,
        run_experiments,
        challenge,
        configuration,
        model,
        arguments_service
    )

    evaluation_service = register_evaluation_service(
        arguments_service=arguments_service,
        file_service=file_service,
        plot_service=plot_service,
        metrics_service=metrics_service,
        process_service=process_service,
        vocabulary_service=vocabulary_service,
        data_service=data_service,
        joint_model=joint_model,
        configuration=configuration)

    experiment_service = register_experiment_service(
        arguments_service=arguments_service,
        dataloader_service=dataloader_service,
        file_service=file_service,
        metrics_service=metrics_service,
        plot_service=plot_service,
        cache_service=cache_service,
        model=model,
        run_experiments=run_experiments)

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
