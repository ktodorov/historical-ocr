from services.log_service import LogService
from overrides import overrides

from models.transformers.transformer_base import TransformerBase
from transformers import XLNetLMHeadModel

from services.arguments.pretrained_arguments_service import PretrainedArgumentsService
from services.data_service import DataService

class XLNet(TransformerBase):
    def __init__(
            self,
            arguments_service: PretrainedArgumentsService,
            data_service: DataService,
            log_service: LogService,
            output_hidden_states: bool = False):
        super().__init__(arguments_service, data_service, log_service, output_hidden_states)

    def forward(self, input_batch, **kwargs):
        input, labels, attention_masks = input_batch
        outputs = self.transformer_model.forward(input, labels=labels, attention_mask=attention_masks)
        return outputs.loss

    @property
    def _model_type(self) -> type:
        return XLNetLMHeadModel

    @property
    def transformer_model(self) -> XLNetLMHeadModel:
        return self._transformer_model