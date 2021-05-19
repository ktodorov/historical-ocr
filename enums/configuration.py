from enums.argument_enum import ArgumentEnum

class Configuration(ArgumentEnum):
    BERT = 'bert'
    XLNet = 'xlnet'
    RoBERTa = 'roberta'
    BART = 'bart'
    CBOW = 'cbow'
    SkipGram = 'skip-gram'
    PPMI = 'ppmi'

    @staticmethod
    def get_friendly_name(configuration) -> str:
        if configuration == Configuration.BERT:
            return 'BERT'
        elif configuration == Configuration.SkipGram:
            return 'Skip-gram'
        elif configuration == Configuration.CBOW:
            return 'CBOW'
        elif configuration == Configuration.PPMI:
            return 'PPMI'

        return None