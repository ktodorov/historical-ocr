from typing import List
import numpy as np

class TokensOccurrenceStats:
    def __init__(
        self, 
        sentences: List[List[int]],
        vocabulary_size: int):

        self._mutual_occurrences = np.zeros((vocabulary_size, vocabulary_size))
        self._token_occurrences = np.zeros(vocabulary_size)

        for sentence in sentences:
            for i in range(len(sentence)):
                self._token_occurrences[sentence[i]] = self._token_occurrences[sentence[i]] + 1

                if i > 0:
                    self._mutual_occurrences[sentence[i], sentence[i-1]] = self._mutual_occurrences[sentence[i], sentence[i-1]] + 1
                    self._mutual_occurrences[sentence[i-1], sentence[i]] = self._mutual_occurrences[sentence[i-1], sentence[i]] + 1

                if i < len(sentence) - 1:
                    self._mutual_occurrences[sentence[i], sentence[i+1]] = self._mutual_occurrences[sentence[i], sentence[i+1]] + 1
                    self._mutual_occurrences[sentence[i+1], sentence[i]] = self._mutual_occurrences[sentence[i+1], sentence[i]] + 1

    @property
    def mutual_occurrences(self):
        return self._mutual_occurrences
        
    @property
    def token_occurrences(self):
        return self._token_occurrences