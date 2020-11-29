from overrides import overrides
import torch

from models.transformers.transformer_base import TransformerBase
from transformers import BertForMaskedLM

from services.arguments.pretrained_arguments_service import PretrainedArgumentsService
from services.data_service import DataService

class BERT(TransformerBase):
    def __init__(
            self,
            arguments_service: PretrainedArgumentsService,
            data_service: DataService,
            output_hidden_states: bool = False):
        super().__init__(arguments_service, data_service, output_hidden_states)

    @overrides
    def forward(self, input_batch, **kwargs):
        input, labels, attention_masks = input_batch
        outputs = self.transformer_model.forward(input, labels=labels, attention_mask=attention_masks)
        return outputs.loss

    @property
    def _model_type(self) -> type:
        return BertForMaskedLM

    @property
    def transformer_model(self) -> BertForMaskedLM:
        return self._transformer_model

    @overrides
    def get_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        outputs = self._transformer_model.forward(tokens)
        padded_embeddings = outputs[1][-1] # BatchSize X MaxLength X EmbeddingSize
        mask = (tokens > 0).unsqueeze(-1).repeat(1, 1, 768).float()
        means = (torch.sum(padded_embeddings * mask, dim=1) / mask.sum(dim=1)).cpu().tolist()

        return means