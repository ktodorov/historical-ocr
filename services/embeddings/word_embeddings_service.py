from services.embeddings.word_alignment_service import WordAlignmentService
from services.vocabulary_service import VocabularyService
from services.log_service import LogService
from models.evaluation_model import EvaluationModel
from services.arguments.ocr_evaluation_arguments_service import OCREvaluationArgumentsService
from typing import List
from tqdm import tqdm

from entities.word_evaluation import WordEvaluation
from enums.ocr_output_type import OCROutputType
from torch.utils.data import DataLoader


class WordEmbeddingsService:
    def __init__(
            self,
            arguments_service: OCREvaluationArgumentsService,
            log_service: LogService,
            vocabulary_service: VocabularyService,
            word_alignment_service: WordAlignmentService):

        self._arguments_service = arguments_service
        self._vocabulary_service = vocabulary_service
        self._log_service = log_service
        self._word_alignment_service = word_alignment_service

    def generate_embeddings(self, model: EvaluationModel, dataloader: DataLoader) -> List[WordEvaluation]:
        result: List[WordEvaluation] = []

        self._log_service.log_debug('Processing common vocabulary tokens')
        dataloader_length = len(dataloader)
        for i, tokens in tqdm(iterable=enumerate(dataloader), desc=f'Processing common vocabulary tokens', total=dataloader_length):
            outputs = model.get_embeddings(tokens)
            result.extend(outputs)

        if self._arguments_service.separate_neighbourhood_vocabularies:
            processed_tokens = [we.word for we in result]
            for ocr_output_type in [OCROutputType.Raw, OCROutputType.GroundTruth]:
                self._log_service.log_debug(
                    f'Processing unique vocabulary tokens for {ocr_output_type.value} type')
                vocab_key = f'vocab-{self._arguments_service.get_dataset_string()}-{ocr_output_type.value}'
                self._vocabulary_service.load_cached_vocabulary(vocab_key)
                unprocessed_tokens = []
                for _, token in self._vocabulary_service.get_vocabulary_tokens(exclude_special_tokens=True):
                    if token in processed_tokens:
                        continue

                    unprocessed_tokens.append(token)

                batch_size = self._arguments_service.batch_size
                with tqdm(desc=f'Unique {ocr_output_type.value} vocabulary tokens', total=len(unprocessed_tokens)) as progress_bar:
                    for i in range(0, len(unprocessed_tokens), batch_size):
                        tokens = unprocessed_tokens[i:i+batch_size]
                        word_evaluations = model.get_embeddings(
                            tokens, skip_unknown=True)
                        result.extend(word_evaluations)
                        processed_tokens.extend(tokens)
                        progress_bar.update(len(tokens))

        # if self._arguments_service.initialize_randomly:
            # TODO
            # result = self._word_alignment_service.align_word_embeddings(result)

        return result
