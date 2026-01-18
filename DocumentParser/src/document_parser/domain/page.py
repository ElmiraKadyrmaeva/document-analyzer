from pydantic import BaseModel
from typing import List

from document_parser.domain.block import Block


class Page(BaseModel):
    """Страница обработанного документа."""

    blocks: List[Block]
