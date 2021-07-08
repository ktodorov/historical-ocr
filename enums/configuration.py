from enums.argument_enum import ArgumentEnum

class Configuration(ArgumentEnum):
    BERT = 'bert'
    ALBERT = 'albert'
    XLNet = 'xlnet'
    RoBERTa = 'roberta'
    BART = 'bart'
    CBOW = 'cbow'
    SkipGram = 'skip-gram'
    PPMI = 'ppmi'
    GloVe = 'glove'

    @staticmethod
    def get_friendly_name(configuration) -> str:
        if configuration == Configuration.BERT:
            return 'BERT'
        elif configuration == Configuration.ALBERT:
            return 'ALBERT'
        elif configuration == Configuration.SkipGram:
            return 'Skip-gram'
        elif configuration == Configuration.CBOW:
            return 'CBOW'
        elif configuration == Configuration.PPMI:
            return 'PPMI'
        elif configuration == Configuration.GloVe:
            return 'GloVe'

        return None