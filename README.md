# Document Analyzer

Desktop приложение для анализа документов и построения дерева ссылок.

## Функции
- Загрузка документов (PDF, Word, сканы)
- Анализ связей между документами
- Построение дерева ссылок

## Поддерживаемые форматы
- PDF (`.pdf`)
- Word (`.doc`, `.docx`)
- Изображения (`.jpg`, `.jpeg`, `.png`, `.bmp`)

## Требования

- Windows 10/11 (x64)
- Python 3.11.x
- LibreOffice (для обработки Word-документов)
- Не требуется GPU (используется CPU-only версия PyTorch)

---

## Установка (Windows)

### 1. Установка Python

Рекомендуется использовать Python версии **3.11.x**.

Проверьте установленную версию:

```powershell
python --version
```

### 2. Создание виртуального окружения

В корне проекта выполните:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\activate
python --version
```

### 3. Обновление pip и инструментов сборки
```powershell
python -m pip install --upgrade pip setuptools wheel
```

### 4. Установка PyTorch (CPU-only)
Для работы ML-модуля используется версия PyTorch без поддержки CUDA:
```powershell
pip install torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2+cpu --index-url https://download.pytorch.org/whl/cpu
```
Если возникает ошибка, связанная с NumPy, выполните:
```powershell
pip install "numpy<2"
```

### 5. Установка остальных зависимостей
```powershell
pip install -r requirements.txt
```
### 6. Установка Tesseract OCR (для OCR)

Установите Tesseract OCR.
Установите языковой пакет rus.

### 7. Установка LibreOffice

LibreOffice используется для конвертации файлов .doc и .docx в PDF.
Установите LibreOffice для Windows (x64).
Убедитесь, что путь к soffice.exe указан корректно в main.py, например:
```powershell
D:\LibreOffice\program\soffice.exe
```
При необходимости путь можно изменить под вашу систему.

### Запуск приложения

После установки всех зависимостей выполните:
```powershell
python main.py
```