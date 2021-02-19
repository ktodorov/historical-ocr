from services.fit_transformation_service import FitTransformationService
from services.tagging_service import TaggingService
from datasets.evaluation_dataset import EvaluationDataset
from datasets.ppmi_dataset import PPMIDataset
from datasets.transformer_lm_dataset import TransformerLMDataset
from datasets.skip_gram_dataset import SkipGramDataset
from datasets.word2vec_dataset import Word2VecDataset
from datasets.dataset_base import DatasetBase
from services.process.ppmi_process_service import PPMIProcessService
from losses.simple_loss import SimpleLoss
from services.experiments.process.word_neighbourhood_service import WordNeighbourhoodService
from services.experiments.process.metrics_process_service import MetricsProcessService
from models.joint_model import JointModel

import dependency_injector.containers as containers
import dependency_injector.providers as providers

from dependency_injection.selector_utils import *

import main


from losses.loss_base import LossBase
from losses.transformer_loss_base import TransformerLossBase
from losses.cross_entropy_loss import CrossEntropyLoss

from models.model_base import ModelBase
from models.transformers.bert import BERT
from models.transformers.xlnet import XLNet
from models.transformers.bart import BART
from models.simple.cbow import CBOW
from models.simple.skip_gram import SkipGram
from models.simple.ppmi import PPMI

from optimizers.optimizer_base import OptimizerBase
from optimizers.sgd_optimizer import SGDOptimizer
from optimizers.adamw_transformer_optimizer import AdamWTransformerOptimizer

from services.arguments.ocr_quality_arguments_service import OCRQualityArgumentsService
from services.arguments.ocr_quality_non_context_arguments_service import OCRQualityNonContextArgumentsService
from services.arguments.ocr_evaluation_arguments_service import OCREvaluationArgumentsService
from services.arguments.arguments_service_base import ArgumentsServiceBase

from services.download.ocr_download_service import OCRDownloadService

from services.process.process_service_base import ProcessServiceBase
from services.process.transformer_process_service import TransformerProcessService
from services.process.word2vec_process_service import Word2VecProcessService
from services.process.skip_gram_process_service import SkipGramProcessService
from services.process.evaluation_process_service import EvaluationProcessService

from services.data_service import DataService
from services.dataloader_service import DataLoaderService
from services.dataset_service import DatasetService
from services.file_service import FileService
from services.log_service import LogService
from services.mask_service import MaskService
from services.metrics_service import MetricsService
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

from services.experiments.process.neighbourhood_overlap_process_service import NeighbourhoodOverlapProcessService
from services.experiments.process.neighbourhood_similarity_process_service import NeighbourhoodSimilarityProcessService

import logging


class IocContainer(containers.DeclarativeContainer):
    """Application IoC container."""

    logger = providers.Singleton(
        logging.Logger,
        name='historical-ocr logger')

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
        ocr_quality=providers.Singleton(OCRQualityArgumentsService),
        ocr_quality_non_context=providers.Singleton(OCRQualityNonContextArgumentsService))

    log_service = providers.Singleton(
        LogService,
        arguments_service=arguments_service,
        logger=logger)

    data_service = providers.Factory(
        DataService,
        log_service=log_service)

    file_service = providers.Factory(
        FileService,
        arguments_service=arguments_service
    )

    cache_service = providers.Singleton(
        CacheService,
        arguments_service=arguments_service,
        file_service=file_service,
        data_service=data_service,
        log_service=log_service)

    plot_service = providers.Factory(
        PlotService,
        data_service=data_service
    )

    vocabulary_service: providers.Provider[VocabularyService] = providers.Singleton(
        VocabularyService,
        data_service=data_service,
        file_service=file_service,
        cache_service=cache_service,
        log_service=log_service
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
            vocabulary_service=vocabulary_service),
        skip_gram=providers.Singleton(
            CBOWTokenizeService,
            vocabulary_service=vocabulary_service),
        ppmi=providers.Singleton(
            CBOWTokenizeService,
            vocabulary_service=vocabulary_service))

    mask_service = providers.Factory(
        MaskService,
        tokenize_service=tokenize_service,
        arguments_service=arguments_service
    )

    metrics_service = providers.Factory(MetricsService)

    string_process_service = providers.Factory(StringProcessService)

    tagging_service = providers.Factory(TaggingService)

    ocr_download_service = providers.Factory(
        OCRDownloadService,
        arguments_service=arguments_service,
        data_service=data_service,
        string_process_service=string_process_service,
        cache_service=cache_service,
        log_service=log_service)

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
            tokenize_service=tokenize_service,
            tagging_service=tagging_service),
        cbow=providers.Singleton(
            Word2VecProcessService,
            arguments_service=arguments_service,
            ocr_download_service=ocr_download_service,
            cache_service=cache_service,
            log_service=log_service,
            vocabulary_service=vocabulary_service,
            file_service=file_service,
            tokenize_service=tokenize_service),
        skip_gram=providers.Singleton(
            SkipGramProcessService,
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
            log_service=log_service),
        ppmi=providers.Singleton(
            PPMIProcessService,
            ocr_download_service=ocr_download_service,
            arguments_service=arguments_service,
            cache_service=cache_service,
            vocabulary_service=vocabulary_service,
            tokenize_service=tokenize_service,
            log_service=log_service))

    dataset_service = providers.Factory(
        DatasetService,
        arguments_service=arguments_service,
        mask_service=mask_service,
        process_service=process_service,
        log_service=log_service)

    dataloader_service = providers.Factory(
        DataLoaderService,
        arguments_service=arguments_service,
        dataset_service=dataset_service,
        log_service=log_service)

    model_selector = providers.Callable(
        get_model_type,
        arguments_service=arguments_service)

    model: providers.Provider[ModelBase] = providers.Selector(
        model_selector,
        joint=providers.Singleton(
            JointModel,
            arguments_service=arguments_service,
            data_service=data_service,
            vocabulary_service=vocabulary_service,
            process_service=process_service,
            log_service=log_service),
        bert=providers.Singleton(
            BERT,
            arguments_service=arguments_service,
            data_service=data_service,
            log_service=log_service),
        xlnet=providers.Singleton(
            XLNet,
            arguments_service=arguments_service,
            data_service=data_service,
            log_service=log_service),
        bart=providers.Singleton(
            BART,
            arguments_service=arguments_service,
            data_service=data_service,
            log_service=log_service),
        cbow=providers.Singleton(
            CBOW,
            arguments_service=arguments_service,
            process_service=process_service,
            data_service=data_service,
            vocabulary_service=vocabulary_service,
            log_service=log_service),
        skip_gram=providers.Singleton(
            SkipGram,
            arguments_service=arguments_service,
            process_service=process_service,
            data_service=data_service,
            vocabulary_service=vocabulary_service,
            log_service=log_service),
        ppmi=providers.Singleton(
            PPMI,
            arguments_service=arguments_service,
            data_service=data_service,
            vocabulary_service=vocabulary_service,
            log_service=log_service))

    loss_selector = providers.Callable(
        get_loss_function,
        arguments_service=arguments_service)

    loss_function: providers.Provider[LossBase] = providers.Selector(
        loss_selector,
        base=providers.Singleton(LossBase),
        cross_entropy=providers.Singleton(CrossEntropyLoss),
        simple=providers.Singleton(SimpleLoss),
        transformer=providers.Singleton(TransformerLossBase))

    optimizer_selector = providers.Callable(
        get_optimizer,
        arguments_service=arguments_service)

    optimizer: providers.Provider[OptimizerBase] = providers.Selector(
        optimizer_selector,
        base=providers.Singleton(
            OptimizerBase,
            arguments_service=arguments_service,
            model=model),
        sgd=providers.Singleton(
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

    fit_transformation_service = providers.Factory(
        FitTransformationService)

    neighbourhood_similarity_process_service = providers.Factory(
        NeighbourhoodSimilarityProcessService,
        arguments_service=arguments_service,
        file_service=file_service,
        log_service=log_service,
        tagging_service=tagging_service)

    word_neighbourhood_service = providers.Factory(
        WordNeighbourhoodService,
        arguments_service=arguments_service,
        metrics_service=metrics_service,
        plot_service=plot_service,
        file_service=file_service,
        log_service=log_service,
        fit_transformation_service=fit_transformation_service,
        cache_service=cache_service,
        neighbourhood_similarity_process_service=neighbourhood_similarity_process_service)

    neighbourhood_overlap_process_service = providers.Factory(
        NeighbourhoodOverlapProcessService,
        arguments_service=arguments_service,
        cache_service=cache_service)

    metrics_process_service = providers.Factory(
        MetricsProcessService,
        metrics_service=metrics_service)

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
            vocabulary_service=vocabulary_service,
            word_neighbourhood_service=word_neighbourhood_service,
            log_service=log_service,
            tagging_service=tagging_service,
            neighbourhood_overlap_process_service=neighbourhood_overlap_process_service,
            metrics_process_service=metrics_process_service,
            model=model),
        none=providers.Object(None))

    test_service = providers.Factory(
        TestService,
        arguments_service=arguments_service,
        dataloader_service=dataloader_service,
        evaluation_service=evaluation_service,
        file_service=file_service,
        model=model
    )

    train_service_selector = providers.Callable(
        include_train_service,
        arguments_service=arguments_service)

    train_service: providers.Provider[TrainService] = providers.Selector(
        train_service_selector,
        include=providers.Factory(
            TrainService,
            arguments_service=arguments_service,
            dataloader_service=dataloader_service,
            loss_function=loss_function,
            optimizer=optimizer,
            log_service=log_service,
            model=model,
            file_service=file_service),
        exclude=providers.Object(None))

    # Misc

    main = providers.Callable(
        main.main,
        arguments_service=arguments_service,
        train_service=train_service,
        test_service=test_service,
        experiment_service=experiment_service,
        log_service=log_service)
