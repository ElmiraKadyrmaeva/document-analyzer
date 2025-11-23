import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QWidget, QTextEdit,
                             QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor


class DocumentAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        """Инициализация пользовательского интерфейса"""
        self.setWindowTitle('Анализ связей между документами')
        self.setGeometry(100, 100, 800, 600)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Основной layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Заголовок
        title_label = QLabel('Система анализа документов и построение дерева ссылок')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont('Arial', 16, QFont.Bold))
        main_layout.addWidget(title_label)

        # Область для отображения информации о загруженных файлах
        self.file_info_text = QTextEdit()
        self.file_info_text.setPlaceholderText('Здесь будет отображаться информация о загруженных документах...')
        self.file_info_text.setReadOnly(True)
        main_layout.addWidget(self.file_info_text)

        # Layout для кнопок
        buttons_layout = QHBoxLayout()

        # Кнопка загрузки файлов
        self.upload_btn = QPushButton('Загрузить документы')
        self.upload_btn.setFont(QFont('Arial', 12))
        self.upload_btn.clicked.connect(self.upload_files)
        buttons_layout.addWidget(self.upload_btn)

        # Кнопка анализа
        self.analyze_btn = QPushButton('Анализировать документы')
        self.analyze_btn.setFont(QFont('Arial', 12))
        self.analyze_btn.clicked.connect(self.analyze_documents)
        self.analyze_btn.setEnabled(False)  # не активна
        buttons_layout.addWidget(self.analyze_btn)

        # Кнопка очистки файлов
        self.clear_btn = QPushButton('Очистить файлы')
        self.clear_btn.setFont(QFont('Arial', 12))
        self.clear_btn.clicked.connect(self.clear_files)
        self.clear_btn.setEnabled(False)  #Изначально неактивна, пока нет файлов
        buttons_layout.addWidget(self.clear_btn)

        main_layout.addLayout(buttons_layout)

        # Список для хранения загруженных файлов
        self.uploaded_files = []

        # Статус бар
        self.statusBar().showMessage('Готов к работе. Загрузите документы для анализа.')

    def upload_files(self):
        """Функция для загрузки файлов"""
        try:
            # Диалог выбора файлов с поддержкой множественного выбора
            file_dialog = QFileDialog()
            files, _ = file_dialog.getOpenFileNames(
                self,
                'Выберите документы для анализа',
                '',  #Начальная директория
                'Документы (*.pdf *.docx *.doc);;'
                'PDF файлы (*.pdf);;'
                'Word документы (*.docx *.doc);;'
                'Изображения (*.jpg *.jpeg *.png *.bmp);;'
                'Все файлы (*)'
            )

            if files:
                self.process_uploaded_files(files)

        except Exception as e:
            self.show_error_message(f"Ошибка при загрузке файлов: {str(e)}")

    def process_uploaded_files(self, files):
        """Обработка загруженных файлов"""
        valid_extensions = {'.pdf', '.docx', '.doc', '.jpg', '.jpeg', '.png', '.bmp'}
        new_files = []

        for file_path in files:
            # Проверка расширения файла
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in valid_extensions:
                if file_path not in self.uploaded_files:
                    self.uploaded_files.append(file_path)
                    new_files.append(file_path)
            else:
                self.show_warning_message(
                    f'Файл {os.path.basename(file_path)} имеет неподдерживаемый формат и будет пропущен.')

        if new_files:
            self.update_file_info()
            self.analyze_btn.setEnabled(True)
            self.clear_btn.setEnabled(True)  # Активируем кнопку очистки
            self.statusBar().showMessage(
                f'Загружено {len(new_files)} новых документов. Всего документов: {len(self.uploaded_files)}')
        else:
            self.statusBar().showMessage('Новых подходящих документов не найдено.')

    def update_file_info(self):
        """Обновление информации о загруженных файлах"""
        file_info = "ЗАГРУЖЕННЫЕ ДОКУМЕНТЫ:\n\n"

        for i, file_path in enumerate(self.uploaded_files, 1):
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            file_ext = os.path.splitext(file_path)[1].upper()

            file_info += f"{i}. {file_name}\n"
            file_info += f"   Тип: {file_ext} | Размер: {self.format_file_size(file_size)}\n"
            file_info += f"   Путь: {file_path}\n\n"

        self.file_info_text.setText(file_info)

    def clear_files(self):
        """Функция для очистки всех прикрепленных файлов"""
        if not self.uploaded_files:
            self.show_warning_message("Нет файлов для очистки.")
            return

        # Запрос подтверждения у пользователя
        reply = QMessageBox.question(
            self,
            'Подтверждение очистки',
            f'Вы уверены, что хотите удалить все прикрепленные файлы?\n\n'
            f'Будет удалено файлов: {len(self.uploaded_files)}',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No  # По умолчанию выбрано "Нет"
        )

        if reply == QMessageBox.Yes:
            # Очищаем список файлов
            self.uploaded_files.clear()
            self.file_info_text.clear()
            self.analyze_btn.setEnabled(False)
            self.clear_btn.setEnabled(False)

            # Обновляем статус
            self.statusBar().showMessage('Все файлы успешно удалены. Готов к загрузке новых документов.')

            QMessageBox.information(
                self,
                'Очистка завершена',
                'Все прикрепленные файлы были успешно удалены.'
            )

    def format_file_size(self, size_bytes):
        """Форматирование размера файла в читаемый вид"""
        if size_bytes == 0:
            return "0 B"

        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1

        return f"{size_bytes:.2f} {size_names[i]}"

    def analyze_documents(self):
        """Функция для анализа документов (заглушка)"""
        if not self.uploaded_files:
            self.show_warning_message("Нет документов для анализа.")
            return

        # заглушка - здесь будет логика анализа?
        QMessageBox.information(
            self,
            'Анализ начат',
            f'Начинается анализ {len(self.uploaded_files)} документов.\n\n'
            'Эта функция находится в разработке.'
        )

        # TODO: Здесь будет вызов функций анализа документов?

    def show_error_message(self, message):
        """Показать сообщение об ошибке"""
        QMessageBox.critical(self, 'Ошибка', message)

    def show_warning_message(self, message):
        """Показать предупреждение"""
        QMessageBox.warning(self, 'Предупреждение', message)


def main():
    """Основная функция приложения"""
    app = QApplication(sys.argv)

    app.setStyle('Fusion')

    window = DocumentAnalyzerApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()