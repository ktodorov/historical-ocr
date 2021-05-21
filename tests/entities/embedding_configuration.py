class EmbeddingConfiguration:
    def __init__(
        self,
        language,
        configuration,
        lr,
        initialize_randomly,
        ocr_output_type):

        self.language = language
        self.configuration = configuration
        self.lr = lr
        self.initialize_randomly = initialize_randomly
        self.ocr_output_type = ocr_output_type
