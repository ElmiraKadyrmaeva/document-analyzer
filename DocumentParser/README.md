# Document Parser

Пакет для разбивки документов на страницы, блоки, строки и слова.

Поддерживаемые форматы документов:
* DOCX, DOC, RTF, ODT, DOT, DOTX
* PDF
* PNG, JPG, JPEG, TIFF, BMP, GIF
## Установка
Для обработки изображений требуется установить [Tesseract](https://github.com/tesseract-ocr/tesseract)

Для обработки документов Microsoft Word требуется установить [LibreOffice](https://libreoffice.org)

### Установка с помощью [uv](https://github.com/astral-sh/uv) (Рекомендуется)
```commandline
uv add "document-parser @ git+https://github.com/i1mk8/DocumentParser.git"
```
### Установка с помощью pip
```commandline
pip install git+https://github.com/i1mk8/DocumentParser.git
```
## Использование
```python
>>> from document_parser import DocumentParser

>>> document_parser = DocumentParser()
>>> result = document_parser.parse('image.png')

>>> result
Document(
    path='image.png',
    source_type='image',
    pages=[Page(
        blocks=[Block(
            lines=[Line(
                words=[Word(
                    word='Пример',
                    bounding_box=BoundingBox(left=28.3, top=32.0, right=36.1, bottom=33.2))]
            )]
        )]
    )]
)

>>> # Результат экспорта в json можно посмотреть в примерах
>>> result.to_json()
...
>>> result.to_dict()
...
>>> result.to_text()
Пример
```
## Примеры
Примеры работы с разными типами документов:
* [DOCX](examples/docx)
* [PDF](examples/pdf)
* [PNG](examples/image)
