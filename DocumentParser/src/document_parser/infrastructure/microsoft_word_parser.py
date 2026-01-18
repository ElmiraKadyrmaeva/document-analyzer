from typing import Optional, List
import os
import platform
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from document_parser.domain.base_document_parser import BaseDocumentParser
from document_parser.domain.document import Document


class MicrosoftWordParser(BaseDocumentParser):
    """
    Парсер Microsoft Word документов.
    Работает через конвертацию в pdf с помощью LibreOffice, после использует PdfParser для извлечения данных.
    """

    _SUPPORTED_EXTENSIONS = ['.docx', '.doc', '.rtf', '.odt', '.dot', '.dotx']
    _LIBRE_OFFICE_WINDOWS_PATH = r'C:\Program Files\LibreOffice\program\soffice.exe'

    def __init__(self, pdf_parser: BaseDocumentParser, libre_office_path: Optional[str] = None):
        """
        :param pdf_parser: Экземпляр парсера pdf
        :param libre_office_path: Путь к LibreOffice
        """
        self._pdf_parser = pdf_parser

        if libre_office_path is not None:
            self._libre_office_path = libre_office_path

        else:
            if platform.system() == 'Windows' and os.path.exists(self._LIBRE_OFFICE_WINDOWS_PATH):
                self._libre_office_path = self._LIBRE_OFFICE_WINDOWS_PATH
            else:
                self._libre_office_path = 'libreoffice'

    @property
    def supported_extensions(self) -> List[str]:
        return self._SUPPORTED_EXTENSIONS

    def process(self, path: str) -> Document:
        with TemporaryDirectory() as temp_dir:
            pdf_path = self._convert_to_pdf(temp_dir, path)

            result = self._pdf_parser.process(pdf_path)
            result.path = path
            result.source_type = Path(path).suffix.lstrip('.').lower()

            return result

    def _convert_to_pdf(self, out_dir: str, file_path: str) -> str:
        subprocess.run(
            [self._libre_office_path, '--headless', '--convert-to', 'pdf', '--outdir', out_dir, file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        original_path = Path(file_path)
        pdf_path = Path(os.path.join(out_dir, original_path.name)).with_suffix('.pdf')

        return str(pdf_path)
