from abc import ABC, abstractmethod
from typing import List

from document_parser.domain.document import Document


class BaseDocumentParser(ABC):
    """Абстрактный базовый класс для реализаций парсеров документов."""

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """
        :return: Список расширений документов, поддерживаемых парсером
        """
        pass

    @abstractmethod
    def process(self, path: str) -> Document:
        """
       Разбивает документ на страницы, блоки, строки и слова.

        :param path: Путь к файлу, который будет обработан
        :return: Обработанный файл
        """
        pass
