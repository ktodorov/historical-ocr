from enums.argument_enum import ArgumentEnum

class Configuration(ArgumentEnum):
    BERT = 'bert'
    XLNet = 'xlnet'
    RoBERTa = 'roberta'
    BART = 'bart'
    CBOW = 'cbow'
