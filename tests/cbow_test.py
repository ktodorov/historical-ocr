import torch
from tests.containers.test_container import TestContainer
import unittest


class CBOWTest(unittest.TestCase):

    def test_embedding_matrix_initialization(self):
        main_container = TestContainer()
        metrics_service = main_container.metrics_service()


        # Model 1
        container_1 = TestContainer()
        vocabulary_service_1 = container_1.vocabulary_service()

        tokens_1 = vocabulary_service_1.get_vocabulary_tokens()

        ids_1 = [id for id, _ in tokens_1]
        ids_tensor_1 = torch.Tensor(ids_1).long()

        model_1 = container_1.model()
        embeddings_1 = model_1.get_embeddings(ids_tensor_1)

        # Model 2
        container_2 = TestContainer()
        vocabulary_service_2 = container_2.vocabulary_service()

        tokens_2 = vocabulary_service_2.get_vocabulary_tokens()

        ids_2 = [id for id, _ in tokens_2]
        ids_tensor_2 = torch.Tensor(ids_2).long()

        model_2 = container_2.model()
        embeddings_2 = model_2.get_embeddings(ids_tensor_2)

        for embedding_1, embedding_2 in zip(embeddings_1, embeddings_2):
            self.assertEqual(embedding_1, embedding_2)
            self.assertEqual(metrics_service.calculate_cosine_distance(embedding_1, embedding_2), 0.0)



if __name__ == '__main__':
    unittest.main()