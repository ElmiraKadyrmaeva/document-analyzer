from pydantic import BaseModel
from typing import List

from document_parser.domain.word import Word


class Line(BaseModel):
    """Строка текста обработанного документа."""

    words: List[Word]
