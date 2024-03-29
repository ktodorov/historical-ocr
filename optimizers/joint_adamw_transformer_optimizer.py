from torch import optim
from torch.optim.optimizer import Optimizer
from overrides import overrides

from models.model_base import ModelBase

from optimizers.optimizer_base import OptimizerBase
from services.arguments.arguments_service_base import ArgumentsServiceBase

from transformers import AdamW


class JointAdamWTransformerOptimizer(OptimizerBase):
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            model: ModelBase):
        super().__init__(arguments_service, model)
        self._weight_decay = arguments_service.weight_decay

    def _init_optimizer(self) -> Optimizer:
        model1_parameters, model2_parameters = self._model.optimizer_parameters()
        optimizer1 = AdamW(model1_parameters, lr=self._learning_rate, weight_decay=self._weight_decay)
        optimizer2 = AdamW(model2_parameters, lr=self._learning_rate, weight_decay=self._weight_decay)
        return (optimizer1, optimizer2)

    def step(self):
        self._optimizer[0].step()
        self._optimizer[1].step()

    def zero_grad(self):
        self._optimizer[0].zero_grad()
        self._optimizer[1].zero_grad()