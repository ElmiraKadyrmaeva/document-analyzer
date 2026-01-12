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

## Технологии
- Python 3.11
- PyQt5
- transformers (HuggingFace)
- LLM: **Qwen/Qwen2.5-3B-Instruct**
- DocumentParser (OCR + структура)
- LibreOffice (конвертация DOC/DOCX → PDF)

---

## Установка (Windows)

### 1. Установка Python
Рекомендуется Python **3.11.x**
```powershell
python --version
```
### 2. Создание виртуального окружения
```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
```
### 3. Обновление pip и инструментов сборки
```powershell
python -m pip install --upgrade pip setuptools wheel
```
### 4. Установка PyTorch (CPU-only)
```powershell
pip install torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2+cpu --index-url https://download.pytorch.org/whl/cpu
```
Если возникнет ошибка с NumPy:
```powershell
pip install "numpy<2"
```
### 5. Установка остальных зависимостей
```powershell
pip install transformers==4.37.2
pip install accelerate
pip install sentencepiece
pip install pyqt5 networkx
pip install git+https://github.com/i1mk8/DocumentParser.git
```
### 6. Установка системных зависимостей
Установить Tesseract OCR (языковой пакет rus)

LibreOffice

Путь к soffice.exe должен быть прописан в main.py, например:
```powershell
C:\LibreOffice\program\soffice.exe
```
### 7. Настройка кеша LLM
```powershell
mkdir C:\hf_cache
$env:HF_HOME="C:\hf_cache"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"
```
### Запуск приложения
```powershell
python main.py
```