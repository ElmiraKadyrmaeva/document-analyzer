from pydantic import BaseModel
from typing import List

from document_parser.domain.line import Line


class Block(BaseModel):
    """Логический блок текста (например, абзац) обработанного документа."""

    lines: List[Line]
