from torch import optim
from torch.optim.optimizer import Optimizer
from overrides import overrides

from models.model_base import ModelBase

from optimizers.optimizer_base import OptimizerBase
from services.arguments.arguments_service_base import ArgumentsServiceBase


class SparseAdamOptimizer(OptimizerBase):
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            model: ModelBase):
        super().__init__(arguments_service, model)
        self._weight_decay = arguments_service.weight_decay

    def _init_optimizer(self) -> Optimizer:
        optimizer = optim.SparseAdam(
            self._model.optimizer_parameters(),
            lr=self._learning_rate)

        return optimizer

    def step(self):
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()
