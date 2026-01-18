from typing import Optional, List
import platform
import os

import pytesseract
from lxml import etree

from document_parser.domain.base_document_parser import BaseDocumentParser
from document_parser.domain.bounding_box import BoundingBox
from document_parser.domain.line import Line
from document_parser.domain.word import Word
from document_parser.domain.block import Block
from document_parser.domain.document import Document
from document_parser.domain.page import Page


class ImageParser(BaseDocumentParser):
    """
    Парсер изображений документов.
    Использует Tesseract для распознавания текста.
    """

    _SUPPORTED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']
    _TESSERACT_WINDOWS_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    _LXML_NAMESPACE = {'alto': 'http://www.loc.gov/standards/alto/ns-v3#'}

    def __init__(self, lang: str, tesseract_path: Optional[str] = None):
        """
        :param lang: Язык Tesseract OCR
        :param tesseract_path: Путь к Tesseract
        """
        self._lang = lang

        if tesseract_path is not None:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        elif platform.system() == 'Windows' and os.path.exists(self._TESSERACT_WINDOWS_PATH):
            pytesseract.pytesseract.tesseract_cmd = self._TESSERACT_WINDOWS_PATH

    @property
    def supported_extensions(self) -> List[str]:
        return self._SUPPORTED_EXTENSIONS

    def process(self, path: str) -> Document:
        xml_ocr = pytesseract.image_to_alto_xml(path, lang=self._lang)
        root = etree.fromstring(xml_ocr)

        block_elements = root.findall('.//alto:TextBlock', namespaces=self._LXML_NAMESPACE)
        parsed_blocks = [self._parse_block(block) for block in block_elements]

        page = Page(blocks=parsed_blocks)

        return Document(
            path=path,
            source_type='image',
            pages=[page]
        )

    def _parse_block(self, block_element: etree._Element) -> Block:
        line_elements = block_element.findall('.//alto:TextLine', namespaces=self._LXML_NAMESPACE)
        parsed_lines = [self._parse_line(line) for line in line_elements]
        return Block(lines=parsed_lines)

    def _parse_line(self, line_element: etree._Element) -> Line:
        word_elements = line_element.findall('.//alto:String', namespaces=self._LXML_NAMESPACE)
        parsed_words = [self._parse_word(word) for word in word_elements]
        return Line(words=parsed_words)

    def _parse_word(self, word_element: etree._Element) -> Word:
        text = word_element.get('CONTENT')
        left = float(word_element.get('HPOS'))
        top = float(word_element.get('VPOS'))
        width = float(word_element.get('WIDTH'))
        height = float(word_element.get('HEIGHT'))

        bounding_box = BoundingBox(
            left=left,
            top=top,
            right=left + width,
            bottom=top + height
        )

        return Word(word=text, bounding_box=bounding_box)