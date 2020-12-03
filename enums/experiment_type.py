from enums.argument_enum import ArgumentEnum

class ExperimentType(ArgumentEnum):
    CosineSimilarity = 'cosine-similarity'
    EuclideanDistance = 'euclidean-distance'
    KLDivergence = 'kl-divergence'