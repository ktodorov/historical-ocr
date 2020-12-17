from enums.run_type import RunType
from enums.configuration import Configuration
from enums.challenge import Challenge
from enums.pretrained_model import PretrainedModel

from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.arguments.pretrained_arguments_service import PretrainedArgumentsService


def get_arguments_service(arguments_service: ArgumentsServiceBase):
    result = 'base'
    challenge = arguments_service.challenge
    run_experiments = arguments_service.run_experiments

    if challenge == Challenge.OCREvaluation:
        if run_experiments:
            result = 'evaluation'
        else:
            result = 'ocr_quality'

    return result


def get_optimizer(arguments_service: ArgumentsServiceBase):
    if arguments_service.evaluate or arguments_service.run_experiments:
        return None

    result = 'base'
    challenge = arguments_service.challenge
    configuration = arguments_service.configuration
    if challenge == Challenge.OCREvaluation:
        if configuration == Configuration.CBOW or configuration == Configuration.SkipGram:
            result = 'sgd'
        elif configuration == Configuration.PPMI:
            result = 'base'
        else:
            result = 'transformer'

    return result


def get_loss_function(arguments_service: ArgumentsServiceBase):
    loss_function = None
    challenge = arguments_service.challenge
    configuration = arguments_service.configuration

    if challenge == Challenge.OCREvaluation:
        if configuration == Configuration.CBOW:
            return 'cross_entropy'
        elif configuration == Configuration.SkipGram:
            return 'simple'
        elif configuration == Configuration.PPMI:
            return 'base'
        else:
            return 'transformer'

    return loss_function


# def register_evaluation_service(
#         arguments_service: ArgumentsServiceBase,
#         file_service: FileService,
#         plot_service: PlotService,
#         metrics_service: MetricsService,
#         process_service: ProcessServiceBase,
#         vocabulary_service: VocabularyService,
#         data_service: DataService,
#         joint_model: bool,
#         configuration: Configuration):
#     evaluation_service = None

#     return evaluation_service


def get_model_type(arguments_service: ArgumentsServiceBase):

    run_experiments = arguments_service.run_experiments
    configuration = arguments_service.configuration

    model = None

    if run_experiments:
        model = 'joint'
    else:
        model = str(configuration.value).replace('-', '_')

    return model


def get_process_service(arguments_service: ArgumentsServiceBase):
    result = None

    challenge = arguments_service.challenge
    run_experiments = arguments_service.run_experiments
    configuration = arguments_service.configuration

    if challenge == Challenge.OCREvaluation:
        if run_experiments:
            result = 'evaluation'
        elif configuration == Configuration.CBOW:
            result = 'cbow'
        elif configuration == Configuration.SkipGram:
            result = 'skip_gram'
        elif configuration == Configuration.PPMI:
            result = 'ppmi'
        else:
            result = 'transformer'

    return result


def get_tokenize_service(arguments_service: ArgumentsServiceBase) -> str:
    pretrained_model_type = None
    if isinstance(arguments_service, PretrainedArgumentsService):
        pretrained_model_type = arguments_service.pretrained_model

    if pretrained_model_type is None:
        configuration = arguments_service.configuration
        return str(configuration.value).replace('-', '_')

    return pretrained_model_type.value


def get_experiment_service(arguments_service: ArgumentsServiceBase):

    run_experiments = arguments_service.run_experiments

    if not run_experiments:
        return 'base'

    return 'ocr_quality'

def include_train_service(arguments_service: ArgumentsServiceBase):
    if arguments_service.run_experiments or arguments_service.evaluate:
        return 'exclude'

    return 'include'

def get_dataset_type(arguments_service: ArgumentsServiceBase):
    joint_model: bool = arguments_service.joint_model
    configuration: Configuration = arguments_service.configuration
    challenge: Challenge = arguments_service.challenge
    result = 'base'

    if not joint_model:
        if challenge == Challenge.OCREvaluation:
            if configuration == Configuration.CBOW:
                result = 'word2vec'
            elif configuration == Configuration.SkipGram:
                result = 'skip_gram'
            elif configuration == Configuration.PPMI:
                result = 'ppmi'
            else:
                result = 'transformer'
    elif joint_model:
        result = 'evaluation'

    return result
