import jellyfish
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
import scipy.spatial.distance as scipy_distances
from sklearn.metrics.pairwise import cosine_distances

from typing import Tuple


class MetricsService:
    def __init__(self):
        pass

    def calculate_jaccard_similarity(self, list1: list, list2: list) -> float:
        if len(list1) == 0 and len(list2) == 0:
            return 0

        set1 = set(list1)
        set2 = set(list2)
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def calculate_normalized_levenshtein_distance(self, string1: str, string2: str) -> int:
        result = float(self.calculate_levenshtein_distance(
            string1, string2)) / max(len(string1), len(string2))

        return result

    def calculate_levenshtein_distance(self, string1: str, string2: str) -> int:
        result = jellyfish.levenshtein_distance(string1, string2)
        return result

    def calculate_f1_score(
            self,
            predictions,
            targets) -> float:
        result = f1_score(targets, predictions)
        return result

    def calculate_precision_score(
            self,
            predictions,
            targets) -> float:
        result = precision_score(targets, predictions)
        return result

    def calculate_recall_score(
            self,
            predictions,
            targets) -> float:
        result = recall_score(targets, predictions)
        return result

    def calculate_precision_recall_fscore_support(
            self,
            predictions,
            targets) -> Tuple[float, float, float, float]:
        result = precision_recall_fscore_support(
            targets,
            predictions,
            warn_for=tuple())

        return result

    def calculate_cosine_distance(self, list1: list, list2: list) -> float:
        if np.sum(list1) == 0 or np.sum(list2) == 0:
            return 0

        cosine_distance = scipy_distances.cosine(list1, list2)
        return cosine_distance

    def calculate_euclidean_distance(self, list1: list, list2: list) -> float:
        euclidean_distance = scipy_distances.euclidean(list1, list2)
        return euclidean_distance

    def calculate_cosine_similarity(self, list1: list, list2: list) -> float:
        """
        Calculates cosine similarity using scipy cosine distance.

        Original formula is `cosine_distance = 1 - cosine_similarity`.
        Thus `cosine_similarity = cosine_distance + 1`

        Results are to be described as
        - −1 meaning exactly opposite
        - +1 meaning exactly the same
        - 0 indicating orthogonality
        """

        cosine_distance = scipy_distances.cosine(list1, list2)
        return cosine_distance - 1


    def calculate_cosine_similarities(self, matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """
        Calculates cosine similarity using scipy cosine distance.

        Original formula is `cosine_distance = 1 - cosine_similarity`.
        Thus `cosine_similarity = cosine_distance + 1`

        Results are to be described as
        - −1 meaning exactly opposite
        - +1 meaning exactly the same
        - 0 indicating orthogonality
        """

        cosine_distance = cosine_distances(matrix1, matrix2)
        return cosine_distance
