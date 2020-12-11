from typing import overload
import unittest
import os
import numpy as np

from dependency_injection.ioc_container import IocContainer
import dependency_injector.providers as providers
from tests.fakes.argument_service_fake import ArgumentServiceFake
from tests.fakes.log_service_fake import LogServiceFake
from run import initialize_seed

def initialize_container(override_args = None) -> IocContainer:
    custom_args = {
    }
    
    if override_args is not None:
        for key, value in override_args.items():
            custom_args[key] = value

    container = IocContainer()
    container.arguments_service.override(
        providers.Factory(
            ArgumentServiceFake,
            custom_values=custom_args))

    container.log_service.override(providers.Factory(LogServiceFake))
    return container


class TestInitializationService(unittest.TestCase):
    def test_stochasticity_when_same_seed_different_container(self):
        # first initialization
        container_1 = initialize_container(override_args={
            'seed': 13,
            'device': 'cpu'
        })

        arguments_service_1 = container_1.arguments_service()
        initialize_seed(arguments_service_1.seed, arguments_service_1.device)

        rand_vector_1 = np.random.randint(512, size=512)
        rand_list_1 = list(rand_vector_1)

        # second initialization
        container_2 = initialize_container(override_args={
            'seed': 13,
            'device': 'cpu'
        })

        arguments_service_2 = container_2.arguments_service()
        initialize_seed(arguments_service_2.seed, arguments_service_2.device)

        rand_vector_2 = np.random.randint(512, size=512)
        rand_list_2 = list(rand_vector_2)


        self.assertEqual(rand_list_1, rand_list_2)

    def test_stochasticity_when_different_seed(self):
        # first initialization
        container_1 = initialize_container(override_args={
            'seed': 13,
            'device': 'cpu'
        })

        arguments_service_1 = container_1.arguments_service()
        initialize_seed(arguments_service_1.seed, arguments_service_1.device)

        rand_vector_1 = np.random.randint(512, size=512)
        rand_list_1 = list(rand_vector_1)

        # second initialization
        container_2 = initialize_container(override_args={
            'seed': 42,
            'device': 'cpu'
        })

        arguments_service_2 = container_2.arguments_service()
        initialize_seed(arguments_service_2.seed, arguments_service_2.device)

        rand_vector_2 = np.random.randint(512, size=512)
        rand_list_2 = list(rand_vector_2)

        # assert
        self.assertNotEqual(rand_list_1, rand_list_2)


    def test_stochasticity_when_same_seed_same_container(self):
        # first initialization
        container_1 = initialize_container(override_args={
            'seed': 13,
            'device': 'cpu'
        })

        arguments_service_1 = container_1.arguments_service()
        initialize_seed(arguments_service_1.seed, arguments_service_1.device)

        rand_vector_1 = np.random.randint(512, size=512)
        rand_list_1 = list(rand_vector_1)

        rand_vector_2 = np.random.randint(512, size=512)
        rand_list_2 = list(rand_vector_2)

        # assert
        self.assertNotEqual(rand_list_1, rand_list_2)


if __name__ == '__main__':
    unittest.main()
