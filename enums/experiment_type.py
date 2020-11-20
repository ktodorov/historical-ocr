from enums.argument_enum import ArgumentEnum

class ExperimentType(ArgumentEnum):
    CosineDistance = 'cosine-distance'
    EuclideanDistance = 'euclidean-distance'
    KLDivergence = 'kl-divergence'