import numpy as np
from copy import deepcopy
import math

from tests.fakes.log_service_fake import LogServiceFake
from enums.language import Language
from enums.configuration import Configuration
from enums.challenge import Challenge
import os
from tests.fakes.argument_service_fake import ArgumentServiceFake
from dependency_injection.ioc_container import IocContainer
import dependency_injector.providers as providers
import unittest


def initialize_container() -> IocContainer:
    container = IocContainer()
    container.arguments_service.override(
        providers.Factory(ArgumentServiceFake,
                          custom_values={
                              'data_folder': os.path.join('tests', 'data'),
                              'output_folder': os.path.join('tests', 'results')
                          }))

    container.log_service.override(providers.Factory(LogServiceFake))
    return container


class TestMetricsService(unittest.TestCase):
    def test_cosine_distance_zero_for_equal_vectors(self):
        container = initialize_container()
        metrics_service = container.metrics_service()

        rand_vector = np.random.randint(512, size=512)
        rand_list = list(rand_vector)
        rand_list_clone = deepcopy(rand_list)

        self.assertEqual(metrics_service.calculate_cosine_distance(rand_list, rand_list_clone), 0.0)


if __name__ == '__main__':
    unittest.main()
