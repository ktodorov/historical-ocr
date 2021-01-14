from tests.fakes.log_service_fake import LogServiceFake
from enums.language import Language
from enums.configuration import Configuration
from enums.challenge import Challenge
import os
from tests.fakes.non_context_service_fake import NonContextServiceFake
from dependency_injection.ioc_container import IocContainer
import dependency_injector.providers as providers
import unittest

def initialize_container() -> IocContainer:
    container = IocContainer()
    container.arguments_service.override(
        providers.Factory(NonContextServiceFake,
            custom_values={
                'data_folder': os.path.join('tests', 'data'),
                'challenge': Challenge.OCREvaluation,
                'configuration': Configuration.CBOW,
                'language': Language.English,
                'output_folder': os.path.join('tests', 'results')
            }))

    container.log_service.override(providers.Factory(LogServiceFake))
    return container


class TestFileService(unittest.TestCase):
    def test_combine_path_missing(self):
        container = initialize_container()
        file_service = container.file_service()

        path_to_test = os.path.join('tests', 'results', 'temp')

        if os.path.exists(path_to_test):
            os.rmdir(path_to_test)

        self.assertRaises(Exception, lambda: file_service.combine_path('tests', 'results', 'temp', create_if_missing=False))
        self.assertFalse(os.path.exists(path_to_test))

    def test_combine_path_create(self):
        container = initialize_container()
        file_service = container.file_service()

        path_to_test = os.path.join('tests', 'results', 'temp')

        if os.path.exists(path_to_test):
            os.rmdir(path_to_test)

        file_service.combine_path('tests', 'results', 'temp', create_if_missing=True)

        self.assertTrue(os.path.exists(path_to_test))

        os.rmdir(path_to_test)

if __name__ == '__main__':
    unittest.main()