from entities.word_evaluation import WordEvaluation
from typing import List
from scipy.spatial import procrustes
import numpy as np

from services.log_service import LogService

class WordAlignmentService:
    def __init__(
        self,
        log_service: LogService):
        self._log_service = log_service

    def align_word_embeddings(self, evaluations: List[WordEvaluation]) -> List[WordEvaluation]:
        if len(evaluations) == 0:
            raise Exception('Evaluations list is empty')

        embeddings_size = evaluations[0].get_embeddings_size()
        model1_embeddings = np.zeros((len(evaluations), embeddings_size))
        model2_embeddings = np.zeros((len(evaluations), embeddings_size))

        for i, word_evaluation in enumerate(evaluations):
            model1_embeddings[i] = word_evaluation.get_embeddings(0)
            model2_embeddings[i] = word_evaluation.get_embeddings(1)

        standardized_model1_embeddings, standardized_model2_embeddings, disparity = procrustes(
            model1_embeddings, model2_embeddings)
        self._log_service.log_debug(f'Disparity found: {disparity}')

        new_evaluations = []
        for i, word_evaluation in enumerate(evaluations):
            new_evaluations.append(WordEvaluation(
                word_evaluation.word,
                [standardized_model1_embeddings[i],
                 standardized_model2_embeddings[i]]))

        return new_evaluations