from services.metrics_service import MetricsService
from services.vocabulary_service import VocabularyService
from tests.fakes.log_service_fake import LogServiceFake
from services.cache_service import CacheService
from services.string_process_service import StringProcessService
from services.data_service import DataService
from services.download.ocr_download_service import OCRDownloadService
from services.process.word2vec_process_service import Word2VecProcessService
from services.tokenize.cbow_tokenize_service import CBOWTokenizeService
from models.simple.cbow import CBOW
from services.file_service import FileService
from enums.challenge import Challenge
from enums.language import Language
from enums.configuration import Configuration
import os
from tests.fakes.argument_service_fake import ArgumentServiceFake
import dependency_injector.containers as containers
import dependency_injector.providers as providers


class TestContainer(containers.DeclarativeContainer):

    arguments_service = providers.Factory(
        ArgumentServiceFake,
        custom_values={
            'data_folder': os.path.join('tests', 'data'),
            'challenge': Challenge.OCREvaluation,
            'configuration': Configuration.CBOW,
            'language': Language.English,
            'output_folder': os.path.join('tests', 'results')
        })

    file_service = providers.Factory(
        FileService,
        arguments_service=arguments_service)

    log_service = providers.Factory(LogServiceFake)

    data_service = providers.Factory(DataService)

    string_process_service = providers.Factory(StringProcessService)

    cache_service = providers.Singleton(
        CacheService,
        arguments_service=arguments_service,
        file_service=file_service,
        data_service=data_service)

    ocr_download_service = providers.Factory(
        OCRDownloadService,
        data_service=data_service,
        string_process_service=string_process_service,
        cache_service=cache_service)

    vocabulary_service = providers.Singleton(
        VocabularyService,
        data_service=data_service,
        file_service=file_service,
        cache_service=cache_service
    )

    metrics_service = providers.Factory(
        MetricsService
    )

    tokenize_service = providers.Singleton(
        CBOWTokenizeService,
        vocabulary_service=vocabulary_service)

    process_service = providers.Singleton(
        Word2VecProcessService,
        arguments_service=arguments_service,
        ocr_download_service=ocr_download_service,
        cache_service=cache_service,
        log_service=log_service,
        vocabulary_service=vocabulary_service,
        file_service=file_service,
        tokenize_service=tokenize_service)

    model = providers.Singleton(
        CBOW,
        arguments_service=arguments_service,
        process_service=process_service,
        data_service=data_service,
        vocabulary_service=vocabulary_service)
