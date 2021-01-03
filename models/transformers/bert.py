from typing import List
from entities.word_evaluation import WordEvaluation
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
        outputs = self.transformer_model.forward(
            input, labels=labels, attention_mask=attention_masks)
        return outputs.loss

    @property
    def _model_type(self) -> type:
        return BertForMaskedLM

    @property
    def transformer_model(self) -> BertForMaskedLM:
        return self._transformer_model

    @overrides
    def get_embeddings(self, tokens: List[str], vocab_ids: torch.Tensor, skip_unknown: bool = False) -> List[WordEvaluation]:
        outputs = self._transformer_model.forward(
            vocab_ids, output_hidden_states=True)
        # BatchSize X MaxLength X EmbeddingSize
        padded_embeddings = outputs[1][-1]
        mask = (tokens > 0).unsqueeze(-1).repeat(1, 1, 768).float()
        means = (torch.sum(padded_embeddings * mask, dim=1) /
                 mask.sum(dim=1)).cpu().tolist()

        result: List[WordEvaluation] = []
        for i, token in enumerate(tokens):
            result.append(WordEvaluation(
                word=token,
                embeddings_list=[means[i]]))

        return result
