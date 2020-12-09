import random
import sys
import logging
import numpy as np

import torch

from dependency_injection.ioc_container import IocContainer


def initialize_seed(seed: int, device: str):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if device == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    # Configure container:
    container = IocContainer()

    container.logger().addHandler(logging.StreamHandler(sys.stdout))

    arguments_service = container.arguments_service()
    initialize_seed(arguments_service.seed, arguments_service.device)

    # Run application:
    container.main()
