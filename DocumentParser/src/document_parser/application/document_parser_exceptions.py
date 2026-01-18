class UnsupportedFileError(Exception):
    """Исключение, выбрасываемое при попытке обработать файл неподдерживаемого формата."""

    def __init__(self, path: str, extensions: str):
        """
        :param path: Путь к файлу
        :param extensions: Расширение файла
        """
        super().__init__(f'Файл {path} с расширением {extensions} не поддерживается')
