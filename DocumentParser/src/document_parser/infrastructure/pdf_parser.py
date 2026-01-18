import os
from typing import List
from collections import defaultdict
from tempfile import TemporaryDirectory

import fitz

from document_parser.domain.base_document_parser import BaseDocumentParser
from document_parser.domain.page import Page
from document_parser.domain.block import Block
from document_parser.domain.document import Document
from document_parser.domain.line import Line
from document_parser.domain.word import Word
from document_parser.domain.bounding_box import BoundingBox


class PdfParser(BaseDocumentParser):
    """
    Парсер pdf документов.
    Использует fitz для извлечения данных.
    В случае отсутствия текста на странице, конвертирует ее в изображение, после использует ImageParser для извлечения данных.
    """

    _SUPPORTED_EXTENSIONS = ['.pdf']
    _RENDER_DPI = 300

    def __init__(self, image_parser: BaseDocumentParser):
        """
        :param image_parser: Экземпляр парсера изображения
        """
        self._image_parser = image_parser

    @property
    def supported_extensions(self) -> List[str]:
        return self._SUPPORTED_EXTENSIONS

    def process(self, path: str) -> Document:
        with fitz.open(path) as file:
            parsed_pages = []

            for page in file:
                words = page.get_text('words')
                page_blocks = self._ocr_page(page) if not words else self._parse_page_words(words)
                parsed_pages.append(Page(blocks=page_blocks))

            return Document(
                path=path,
                source_type='pdf',
                pages=parsed_pages
            )

    def _parse_page_words(self, words: list) -> List[Block]:
        blocks_dict = defaultdict(lambda: defaultdict(list))

        for word in words:
            left, top, right, bottom, text, block_index, line_index, _ = word

            bounding_box = BoundingBox(
                left=left,
                top=top,
                right=right,
                bottom=bottom,
            )
            blocks_dict[block_index][line_index].append(Word(word=text, bounding_box=bounding_box))

        parsed_blocks = []
        for block_index in sorted(blocks_dict.keys()):
            parsed_lines = []

            for line_index in sorted(blocks_dict[block_index].keys()):
                words_list = blocks_dict[block_index][line_index]
                parsed_lines.append(Line(words=words_list))

            parsed_blocks.append(Block(lines=parsed_lines))
        return parsed_blocks

    def _ocr_page(self, page: fitz.Page) -> List[Block]:
        rendered_page = page.get_pixmap(dpi=self._RENDER_DPI)

        with TemporaryDirectory() as tmp_dir:
            page_path = os.path.join(tmp_dir, 'page.png')
            rendered_page.save(page_path)
            parsed_page = self._image_parser.process(page_path)

            if not parsed_page.pages:
                return []

            scale_x = page.rect.width / rendered_page.width
            scale_y = page.rect.height / rendered_page.height

            return self._scale_blocks(parsed_page.pages[0].blocks, scale_x, scale_y)

    def _scale_blocks(self, blocks: List[Block], scale_x: float, scale_y: float) -> List[Block]:
        scaled_blocks = []

        for block in blocks:
            scaled_lines = []

            for line in block.lines:
                scaled_words = []

                for word in line.words:
                    bounding_box = BoundingBox(
                        left=word.bounding_box.left * scale_x,
                        top=word.bounding_box.top * scale_y,
                        right=word.bounding_box.right * scale_x,
                        bottom=word.bounding_box.bottom * scale_y
                    )

                    scaled_words.append(Word(word=word.word, bounding_box=bounding_box))
                scaled_lines.append(Line(words=scaled_words))
            scaled_blocks.append(Block(lines=scaled_lines))

        return scaled_blocks
