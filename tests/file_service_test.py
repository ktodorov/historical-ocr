import os
from tests.containers.test_container import TestContainer
import unittest


class FileServiceTest(unittest.TestCase):
    def test_combine_path_missing(self):
        container = TestContainer()

        file_service = container.file_service()

        path_to_test = os.path.join('tests', 'results', 'temp')

        if os.path.exists(path_to_test):
            os.rmdir(path_to_test)

        self.assertRaises(Exception, lambda: file_service.combine_path('tests', 'results', 'temp', create_if_missing=False))
        self.assertFalse(os.path.exists(path_to_test))

    def test_combine_path_create(self):
        container = TestContainer()
        file_service = container.file_service()

        path_to_test = os.path.join('tests', 'results', 'temp')

        if os.path.exists(path_to_test):
            os.rmdir(path_to_test)

        file_service.combine_path('tests', 'results', 'temp', create_if_missing=True)

        self.assertTrue(os.path.exists(path_to_test))

        os.rmdir(path_to_test)

if __name__ == '__main__':
    unittest.main()