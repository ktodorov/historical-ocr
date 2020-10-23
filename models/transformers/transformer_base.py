import os 
from overrides import overrides

from models.model_base import ModelBase
from transformers import PreTrainedModel, PretrainedConfig

from entities.model_checkpoint import ModelCheckpoint
from entities.metric import Metric

from services.arguments.pretrained_arguments_service import PretrainedArgumentsService
from services.data_service import DataService

class TransformerBase(ModelBase):
    def __init__(
            self,
            arguments_service: PretrainedArgumentsService,
            data_service: DataService,
            output_hidden_states: bool = False):
        super(TransformerBase, self).__init__(data_service, arguments_service)

        self._output_hidden_states = output_hidden_states

        if arguments_service.resume_training or arguments_service.evaluate or arguments_service.run_experiments:
            self._transformer_model = None
        else:
            config = PretrainedConfig.from_pretrained(arguments_service.pretrained_weights, return_dict=True)
            config.output_hidden_states = output_hidden_states

            self._transformer_model: PreTrainedModel = self._model_type.from_pretrained(
                arguments_service.pretrained_weights,
                config=config)

        self._arguments_service = arguments_service

    @property
    def transformer_model(self) -> PreTrainedModel:
        return self._transformer_model

    @overrides
    def forward(self, input_batch, **kwargs):
        pass

    @overrides
    def named_parameters(self):
        return self._transformer_model.named_parameters()

    @overrides
    def parameters(self):
        return self._transformer_model.parameters()

    @overrides
    def compare_metric(self, best_metric: Metric, new_metrics: Metric) -> bool:
        if best_metric.is_new or best_metric.get_current_loss() > new_metrics.get_current_loss():
            return True

        return False

    @overrides
    def save(
            self,
            path: str,
            epoch: int,
            iteration: int,
            best_metrics: object,
            resets_left: int,
            name_prefix: str = None) -> bool:

        checkpoint_name = self._arguments_service.checkpoint_name

        if checkpoint_name:
            name_prefix = f'{name_prefix}_{checkpoint_name}'

        saved = super().save(path, epoch, iteration, best_metrics,
                             resets_left, name_prefix, save_model_dict=False)

        if not saved:
            return saved

        pretrained_weights_path = self._get_pretrained_path(
            path, name_prefix, create_if_missing=True)

        self._transformer_model.save_pretrained(pretrained_weights_path)

        return saved

    @overrides
    def load(
            self,
            path: str,
            name_prefix: str = None,
            load_model_dict: bool = True,
            load_model_only: bool = False) -> ModelCheckpoint:

        checkpoint_name = self._arguments_service.checkpoint_name

        if checkpoint_name:
            name_prefix = f'{name_prefix}_{checkpoint_name}'

        model_checkpoint = super().load(path, name_prefix, load_model_dict=False)
        if not load_model_only and not model_checkpoint:
            return None

        if load_model_dict:
            self._load_transformer_model(path, name_prefix)

        return model_checkpoint

    @property
    def _model_type(self) -> type:
        return PreTrainedModel

    def _load_transformer_model(self, path: str, name_prefix: str):
        pretrained_weights_path = self._get_pretrained_path(path, name_prefix)

        config = PretrainedConfig.from_pretrained(pretrained_weights_path)
        config.output_hidden_states = True

        self._transformer_model = self._model_type.from_pretrained(
            pretrained_weights_path, config=config).to(self._arguments_service.device)

    def _get_pretrained_path(self, path: str, name_prefix: str, create_if_missing: bool = False):
        file_name = f'{name_prefix}_pretrained_weights'
        pretrained_weights_path = os.path.join(path, file_name)

        if create_if_missing and not os.path.exists(pretrained_weights_path):
            os.mkdir(pretrained_weights_path)

        return pretrained_weights_path