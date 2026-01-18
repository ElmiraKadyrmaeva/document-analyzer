from pydantic import BaseModel

from document_parser.domain.bounding_box import BoundingBox


class Word(BaseModel):
    """Отдельное слово обработанного документа."""

    word: str
    bounding_box: BoundingBox
