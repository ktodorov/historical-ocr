from enums.overlap_type import OverlapType
from entities.word_evaluation import WordEvaluation
from typing import Dict, List
from services.metrics_service import MetricsService


class MetricsProcessService:
    def __init__(
        self,
        metrics_service: MetricsService):
        self._metrics_service = metrics_service

    def calculate_cosine_similarities(self, word_evaluations: List[WordEvaluation]) -> Dict[str, float]:
        result = {}
        for word_evaluation in word_evaluations:
            if not word_evaluation.contains_all_embeddings():
                continue

            result[word_evaluation.word] = self._metrics_service.calculate_cosine_similarity(
                list1=word_evaluation.get_embeddings(idx=0),
                list2=word_evaluation.get_embeddings(idx=1))

        return result

    def calculate_cosine_distances(self, word_evaluations: List[WordEvaluation]) -> Dict[str, float]:
        result = {}
        for word_evaluation in word_evaluations:
            if not word_evaluation.contains_all_embeddings(OverlapType.GTvsRaw):
                continue

            cosine_distance = self._metrics_service.calculate_cosine_distance(
                list1=word_evaluation.get_embeddings(idx=0),
                list2=word_evaluation.get_embeddings(idx=1))

            result[word_evaluation.word] = cosine_distance

        return result

    def calculate_euclidean_distances(self, word_evaluations: List[WordEvaluation]) -> Dict[str, float]:
        result = {}

        for word_evaluation in word_evaluations:
            if not word_evaluation.contains_all_embeddings():
                continue

            result[word_evaluation.word] = self._metrics_service.calculate_euclidean_distance(
                list1=word_evaluation.get_embeddings(idx=0),
                list2=word_evaluation.get_embeddings(idx=1))

        return result