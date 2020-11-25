from overrides import overrides
from transformers import XLNetTokenizerFast

from services.arguments.pretrained_arguments_service import PretrainedArgumentsService

from services.tokenize.transformer_tokenize_service import TransformerTokenizeService

class XLNetTokenizeService(TransformerTokenizeService):
    def __init__(
            self,
            arguments_service: PretrainedArgumentsService):
        super().__init__(arguments_service)

    @property
    @overrides
    def _tokenizer_type(self) -> type:
        return XLNetTokenizerFast