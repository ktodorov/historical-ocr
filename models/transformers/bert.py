from services.tokenize.base_tokenize_service import BaseTokenizeService
from services.log_service import LogService
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
            log_service: LogService,
            tokenize_service: BaseTokenizeService,
            output_hidden_states: bool = False):
        super().__init__(arguments_service, data_service, log_service, output_hidden_states)

        self._tokenize_service = tokenize_service

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
    def get_embeddings(self, tokens: List[str], skip_unknown: bool = False) -> List[WordEvaluation]:
        # encode the tokens
        vocab_ids_list, _, _, _ = self._tokenize_service.encode_sequences(tokens)
        vocab_ids = torch.Tensor(vocab_ids_list).to(self._arguments_service.device)

        # process through the pipeline
        mask = (vocab_ids != self._arguments_service.padding_idx)
        outputs = self._transformer_model.forward(
            vocab_ids, mask, output_hidden_states=True)

        # BatchSize X MaxLength X EmbeddingSize
        padded_embeddings = outputs[1][0]
        mask = mask.unsqueeze(-1).repeat(1, 1, 768).float()
        embeddings_means = (torch.sum(padded_embeddings * mask, dim=1) /
                 mask.sum(dim=1)).cpu().tolist()

        return embeddings_means
