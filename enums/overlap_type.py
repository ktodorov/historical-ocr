from enum import Enum

class OverlapType(Enum):
    BASEvsGT = 'BASE-vs-GT'
    BASEvsOCR = 'BASE-vs-OCR'
    BASEvsOG = 'BASE-vs-OG'
    GTvsOCR = 'GT-vs-RAW'