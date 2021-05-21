
from enums.language import Language
from enums.configuration import Configuration
from enums.challenge import Challenge
from enums.ocr_output_type import OCROutputType

configurations = {
    Language.English: {
        Configuration.SkipGram: {
            1e-3: {
                True: [
                    OCROutputType.GroundTruth
                ],
                False: [
                    OCROutputType.GroundTruth,
                    OCROutputType.Raw,
                ]
            },
            1e-4: {
                False: [
                    OCROutputType.GroundTruth,
                    OCROutputType.Raw,
                ]
            }
        },
        Configuration.CBOW: {
            1e-3: {
                False: [
                    OCROutputType.GroundTruth,
                    OCROutputType.Raw,
                ]
            },
            1e-4: {
                False: [
                    OCROutputType.GroundTruth,
                    OCROutputType.Raw,
                ]
            }
        },
        # Configuration.BERT: {
        #     1e-5: {
        #         False: [
        #             OCROutputType.GroundTruth,
        #             OCROutputType.Raw,
        #         ]
        #     },
        #     # 1e-4: {
        #     #     False: [
        #     #         OCROutputType.GroundTruth,
        #     #         OCROutputType.Raw,
        #     #     ]
        #     # }
        # },
    }
}
