from typing import List, Dict

import torch

from services.evaluation.base_evaluation_service import BaseEvaluationService

from entities.batch_representation import BatchRepresentation
from enums.evaluation_type import EvaluationType


class OCRQualityEvaluationService(BaseEvaluationService):
    def __init__(self):
        super().__init__()

    def evaluate_batch(
            self,
            output: torch.Tensor,
            batch_input: BatchRepresentation,
            evaluation_types: List[EvaluationType],
            batch_index: int) -> Dict[EvaluationType, List]:
        return {}

    def save_results(self, evaluation: Dict[EvaluationType, List], targets: List[str]):
        print(evaluation)
