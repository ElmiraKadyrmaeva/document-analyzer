import sys
import os
import json
import csv
import traceback
import subprocess
import tempfile
from datetime import datetime

os.environ["PATH"] = r"D:\LibreOffice\program;" + os.environ.get("PATH", "")

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QWidget, QTextEdit,
    QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
    QGraphicsTextItem, QGraphicsLineItem, QGraphicsItem
)
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPen, QBrush

from document_parser import DocumentParser

# LLM import
ML_AVAILABLE = False
ML_IMPORT_ERROR = None
try:
    from ml_model_llm.infer_llm import load_model, predict_with_reason, DEFAULT_MODEL_ID
    ML_AVAILABLE = True
except Exception as e:
    ML_AVAILABLE = False
    ML_IMPORT_ERROR = e
    DEFAULT_MODEL_ID = "неизвестно"

# networkx
try:
    import networkx as nx
    NX_AVAILABLE = True
except Exception:
    NX_AVAILABLE = False

MAX_PAGES_TEXT = 2
MAX_PAGES_JSON = 2
OUTPUTS_DIR_NAME = "outputs"


class GraphView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

    def wheelEvent(self, event):
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)


class GraphWindow(QMainWindow):
    def __init__(self, nodes, edges, labels):
        super().__init__()
        self.setWindowTitle("Граф связей документов")
        self.setGeometry(150, 150, 1100, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        central.setLayout(layout)

        info = QLabel(f"Узлов: {len(nodes)} | Рёбер: {len(edges)}")
        info.setFont(QFont("Arial", 11))
        layout.addWidget(info)

        self.scene = QGraphicsScene()
        self.view = GraphView(self.scene)
        layout.addWidget(self.view)

        self._draw(nodes, edges, labels)

    def _layout_positions(self, nodes, edges):
        if NX_AVAILABLE:
            G = nx.Graph()
            for n in nodes:
                G.add_node(n)
            for e in edges:
                G.add_edge(e["src"], e["dst"], weight=float(e.get("score", 1.0)))

            pos = nx.spring_layout(G, k=1.2, iterations=80, seed=42)
            scale = 320.0
            return {n: (pos[n][0] * scale, pos[n][1] * scale) for n in nodes}

        import math
        R = 320.0
        res = {}
        for i, n in enumerate(nodes):
            ang = 2 * math.pi * i / max(1, len(nodes))
            res[n] = (R * math.cos(ang), R * math.sin(ang))
        return res

    def _draw(self, nodes, edges, labels):
        self.scene.clear()
        pos = self._layout_positions(nodes, edges)

        node_radius = 26
        pen_edge = QPen(Qt.black)
        pen_edge.setWidth(2)

        pen_node = QPen(Qt.black)
        pen_node.setWidth(2)
        brush_node = QBrush(Qt.white)

        for e in edges:
            a = e["src"]
            b = e["dst"]
            x1, y1 = pos[a]
            x2, y2 = pos[b]
            line = QGraphicsLineItem(x1, y1, x2, y2)
            line.setPen(pen_edge)
            line.setZValue(0)
            self.scene.addItem(line)

        for n in nodes:
            x, y = pos[n]
            circle = QGraphicsEllipseItem(
                x - node_radius, y - node_radius,
                node_radius * 2, node_radius * 2
            )
            circle.setPen(pen_node)
            circle.setBrush(brush_node)
            circle.setZValue(1)
            circle.setFlag(QGraphicsItem.ItemIsMovable, True)
            self.scene.addItem(circle)

            name = labels.get(n, n)
            text_item = QGraphicsTextItem(name)
            text_item.setDefaultTextColor(Qt.black)
            text_item.setFont(QFont("Arial", 9))
            text_item.setPos(x + node_radius + 6, y - 10)
            text_item.setZValue(3)
            self.scene.addItem(text_item)

        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-80, -80, 80, 80))
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)


class ModelLoadWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def run(self):
        try:
            model = load_model()
            self.finished.emit(model)
        except Exception:
            self.error.emit(traceback.format_exc())


class AnalyzeWorker(QObject):
    finished = pyqtSignal(str, list)   # report_text, links
    error = pyqtSignal(str)

    def __init__(self, app_ref):
        super().__init__()
        self.app_ref = app_ref

    def run(self):
        try:
            report_text, links = self.app_ref._do_analysis_in_worker()
            self.finished.emit(report_text, links)
        except Exception:
            self.error.emit(traceback.format_exc())


class DocumentAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.parser = DocumentParser(ocr_lang="rus")
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.outputs_dir = os.path.join(self.base_dir, OUTPUTS_DIR_NAME)
        os.makedirs(self.outputs_dir, exist_ok=True)

        self.ml_ready = False
        self.ml_model = None
        self._model_loading = False
        self._analysis_running = False

        self.uploaded_files = []
        self.texts = {}
        self.links = []
        self.tree = {}

        self._ui_timer = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Анализ связей между документами")
        self.setGeometry(100, 100, 980, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        title_label = QLabel("Система анализа документов и построение дерева ссылок")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        main_layout.addWidget(title_label)

        if not ML_AVAILABLE:
            ml_status = f"LLM: недоступна (ошибка импорта: {type(ML_IMPORT_ERROR).__name__})"
        else:
            ml_status = f"LLM: будет загружена при анализе (модель: {DEFAULT_MODEL_ID})"

        self.ml_status_label = QLabel(ml_status)
        self.ml_status_label.setWordWrap(True)
        self.ml_status_label.setFont(QFont("Arial", 10))
        main_layout.addWidget(self.ml_status_label)

        self.file_info_text = QTextEdit()
        self.file_info_text.setPlaceholderText("Здесь будет вывод анализа...")
        self.file_info_text.setReadOnly(True)
        main_layout.addWidget(self.file_info_text)

        buttons_layout = QHBoxLayout()

        self.upload_btn = QPushButton("Загрузить документы")
        self.upload_btn.setFont(QFont("Arial", 12))
        self.upload_btn.clicked.connect(self.upload_files)
        buttons_layout.addWidget(self.upload_btn)

        self.analyze_btn = QPushButton("Анализировать документы")
        self.analyze_btn.setFont(QFont("Arial", 12))
        self.analyze_btn.clicked.connect(self.analyze_documents)
        self.analyze_btn.setEnabled(False)
        buttons_layout.addWidget(self.analyze_btn)

        self.tree_btn = QPushButton("Построить дерево")
        self.tree_btn.setFont(QFont("Arial", 12))
        self.tree_btn.clicked.connect(self.build_tree_visual)
        self.tree_btn.setEnabled(False)
        buttons_layout.addWidget(self.tree_btn)

        self.clear_btn = QPushButton("Очистить файлы")
        self.clear_btn.setFont(QFont("Arial", 12))
        self.clear_btn.clicked.connect(self.clear_files)
        self.clear_btn.setEnabled(False)
        buttons_layout.addWidget(self.clear_btn)

        main_layout.addLayout(buttons_layout)
        self.statusBar().showMessage("Готово. Загрузите документы для анализа.")

    def show_error_message(self, message):
        QMessageBox.critical(self, "Ошибка", message)

    def show_warning_message(self, message):
        QMessageBox.warning(self, "Предупреждение", message)

    def _set_buttons_enabled(self, enabled: bool):
        self.upload_btn.setEnabled(enabled)
        self.analyze_btn.setEnabled(enabled and len(self.uploaded_files) > 0)
        self.clear_btn.setEnabled(enabled and len(self.uploaded_files) > 0)
        self.tree_btn.setEnabled(enabled and len(self.links) > 0 and any(e.get("linked") for e in self.links))

    # ---------- file handling ----------
    def upload_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите документы для анализа",
            "",
            "Документы (*.pdf *.docx *.doc *.jpg *.jpeg *.png *.bmp);;Все файлы (*)"
        )
        if files:
            self.process_uploaded_files(files)

    def process_uploaded_files(self, files):
        valid_extensions = {".pdf", ".docx", ".doc", ".jpg", ".jpeg", ".png", ".bmp"}
        new_files = []

        for file_path in files:
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in valid_extensions and file_path not in self.uploaded_files:
                self.uploaded_files.append(file_path)
                new_files.append(file_path)

        if new_files:
            self.update_file_info()
            self.analyze_btn.setEnabled(True)
            self.clear_btn.setEnabled(True)
            self.tree_btn.setEnabled(False)
            self.statusBar().showMessage(f"Загружено документов: {len(self.uploaded_files)}")

    def update_file_info(self):
        lines = ["ЗАГРУЖЕННЫЕ ДОКУМЕНТЫ", "-" * 60]
        for i, file_path in enumerate(self.uploaded_files, 1):
            lines.append(f"{i}. {os.path.basename(file_path)}")
            lines.append(f"   {file_path}")
        self.file_info_text.setText("\n".join(lines))

    def clear_files(self):
        self.uploaded_files.clear()
        self.texts = {}
        self.links = []
        self.tree = {}
        self.file_info_text.clear()
        self.analyze_btn.setEnabled(False)
        self.tree_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.statusBar().showMessage("Файлы очищены.")

    # ---------- parse helpers ----------
    def parsed_doc_to_text(self, doc, max_pages=2) -> str:
        if not doc or not getattr(doc, "pages", None):
            return ""
        parts = []
        for page in doc.pages[:max_pages]:
            for block in page.blocks:
                for line in block.lines:
                    line_text = " ".join([w.word for w in line.words if getattr(w, "word", "")])
                    if line_text.strip():
                        parts.append(line_text)
        return "\n".join(parts)

    def serialize_doc_to_dict(self, doc, max_pages=2) -> dict:
        if not doc:
            return {}
        out = {"source_path": getattr(doc, "path", None),
               "source_type": getattr(doc, "source_type", None),
               "pages": []}
        pages = getattr(doc, "pages", []) or []
        for page in pages[:max_pages]:
            page_dict = {"blocks": []}
            for block in getattr(page, "blocks", []) or []:
                block_dict = {"lines": []}
                for line in getattr(block, "lines", []) or []:
                    line_dict = {"words": []}
                    for w in getattr(line, "words", []) or []:
                        bbox = getattr(w, "bounding_box", None)
                        line_dict["words"].append({
                            "word": getattr(w, "word", ""),
                            "bbox": bbox.__dict__ if bbox else None
                        })
                    block_dict["lines"].append(line_dict)
                page_dict["blocks"].append(block_dict)
            out["pages"].append(page_dict)
        return out

    def safe_filename(self, name: str) -> str:
        forbidden = '<>:"/\\|?*'
        for ch in forbidden:
            name = name.replace(ch, "_")
        return name.strip()

    def convert_word_to_pdf(self, word_path: str) -> str:
        out_dir = tempfile.mkdtemp(prefix="lo_conv_")
        soffice = r"D:\LibreOffice\program\soffice.exe"
        subprocess.run(
            [soffice, "--headless", "--nologo", "--nofirststartwizard",
             "--convert-to", "pdf", "--outdir", out_dir, word_path],
            check=True
        )
        base = os.path.splitext(os.path.basename(word_path))[0]
        pdf_path = os.path.join(out_dir, base + ".pdf")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"LibreOffice не создал PDF: {pdf_path}")
        return pdf_path

    # ---------- outputs ----------
    def save_txt(self, base_name: str, text: str) -> str:
        path = os.path.join(self.outputs_dir, self.safe_filename(base_name) + ".txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def save_json(self, base_name: str, data: dict) -> str:
        path = os.path.join(self.outputs_dir, self.safe_filename(base_name) + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def save_links_csv(self, links: list) -> str:
        path = os.path.join(self.outputs_dir, "links.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["doc1", "doc2", "linked", "reason"])
            for e in links:
                w.writerow([os.path.basename(e["src"]), os.path.basename(e["dst"]), int(e["linked"]), e.get("reason", "")])
        return path

    def save_tree_json(self, tree: dict) -> str:
        path = os.path.join(self.outputs_dir, "tree.json")
        payload = {"generated_at": datetime.now().isoformat(timespec="seconds"), "tree": tree}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

    # ---------- main actions ----------
    def analyze_documents(self):
        if self._analysis_running:
            return

        if not self.uploaded_files:
            self.show_warning_message("Нет документов для анализа.")
            return

        if not ML_AVAILABLE:
            self.show_error_message(f"LLM недоступна: {type(ML_IMPORT_ERROR).__name__}: {ML_IMPORT_ERROR}")
            return

        if not self.ml_ready and not self._model_loading:
            self._start_model_loading()
            return

        self._start_analysis()

    def _start_model_loading(self):
        self._model_loading = True
        self._set_buttons_enabled(False)
        self.statusBar().showMessage("Загрузка LLM-модели...")

        self.file_info_text.setText(
            "Загрузка модели...\n"
            "Пожалуйста, подождите.\n"
            "После загрузки можно будет снова нажимать кнопки.\n"
        )

        # UI keep-alive timer
        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(lambda: QApplication.processEvents())
        self._ui_timer.start(100)

        self._thread_model = QThread()
        self._worker_model = ModelLoadWorker()
        self._worker_model.moveToThread(self._thread_model)

        self._thread_model.started.connect(self._worker_model.run)
        self._worker_model.finished.connect(self._on_model_loaded)
        self._worker_model.error.connect(self._on_model_load_error)
        self._worker_model.finished.connect(self._thread_model.quit)
        self._worker_model.error.connect(self._thread_model.quit)

        self._thread_model.start()

    def _on_model_loaded(self, model):
        if self._ui_timer:
            self._ui_timer.stop()
            self._ui_timer = None

        self.ml_model = model
        self.ml_ready = True
        self._model_loading = False

        self.ml_status_label.setText(f"LLM: загружена (модель: {DEFAULT_MODEL_ID})")
        self.statusBar().showMessage("Модель загружена. Можно запускать анализ.")
        self._set_buttons_enabled(True)

    def _on_model_load_error(self, err_text):
        if self._ui_timer:
            self._ui_timer.stop()
            self._ui_timer = None

        self._model_loading = False
        self.ml_ready = False
        self.ml_model = None
        self._set_buttons_enabled(True)
        self.ml_status_label.setText("LLM: ошибка загрузки")
        self.show_error_message("Ошибка загрузки модели:\n\n" + err_text)

    def _start_analysis(self):
        self._analysis_running = True
        self._set_buttons_enabled(False)
        self.statusBar().showMessage("Анализ выполняется...")

        self.file_info_text.setText("Анализ выполняется...\nПожалуйста, подождите.\n")

        self._thread_an = QThread()
        self._worker_an = AnalyzeWorker(self)
        self._worker_an.moveToThread(self._thread_an)

        self._thread_an.started.connect(self._worker_an.run)
        self._worker_an.finished.connect(self._on_analysis_finished)
        self._worker_an.error.connect(self._on_analysis_error)
        self._worker_an.finished.connect(self._thread_an.quit)
        self._worker_an.error.connect(self._thread_an.quit)

        self._thread_an.start()

    def _on_analysis_finished(self, report_text, links):
        self.links = links
        self.file_info_text.setText(report_text)
        self._analysis_running = False
        self._set_buttons_enabled(True)
        self.tree_btn.setEnabled(any(e.get("linked") for e in self.links))
        self.statusBar().showMessage("Анализ завершён.")

    def _on_analysis_error(self, err_text):
        self._analysis_running = False
        self._set_buttons_enabled(True)
        self.show_error_message("Ошибка анализа:\n\n" + err_text)

    # Весь тяжёлый анализ — ТОЛЬКО здесь, в worker-потоке
    def _do_analysis_in_worker(self):
        self.texts = {}
        links = []

        report = []
        report.append("ОТЧЁТ О РАБОТЕ СИСТЕМЫ АНАЛИЗА ДОКУМЕНТОВ")
        report.append(f"Дата и время: {datetime.now().isoformat(timespec='seconds')}")
        report.append("")
        report.append("ЭТАП 1. Извлечение текста из документов и сохранение результатов")
        report.append("-" * 70)
        report.append("")

        ok_count = 0

        for path in self.uploaded_files:
            base = os.path.basename(path)
            base_noext = os.path.splitext(base)[0]

            report.append(f"Документ: {base}")
            report.append(f"Путь: {path}")

            if not os.path.exists(path):
                report.append("Статус: ОШИБКА (файл не найден)")
                report.append("")
                continue

            try:
                ext = os.path.splitext(path)[1].lower()
                parse_path = path

                if ext in (".doc", ".docx"):
                    parse_path = self.convert_word_to_pdf(path)
                    report.append("Преобразование: Word → PDF выполнено")
                    report.append(f"PDF: {parse_path}")

                doc = self.parser.parse(parse_path)
                text = self.parsed_doc_to_text(doc, max_pages=MAX_PAGES_TEXT)
                self.texts[path] = text

                txt_path = self.save_txt(base_noext, text)
                json_dict = self.serialize_doc_to_dict(doc, max_pages=MAX_PAGES_JSON)
                json_path = self.save_json(base_noext, json_dict)

                ok_count += 1

                report.append("Статус: УСПЕШНО")
                report.append(f"Длина текста (первые {MAX_PAGES_TEXT} стр.): {len(text)} символов")
                report.append(f"TXT:  {txt_path}")
                report.append(f"JSON: {json_path}")

                fragment = (text or "").replace("\r", "").strip()[:200]
                report.append("Фрагмент текста (до 200 символов):")
                report.append(f"  {fragment if fragment else '(пусто)'}")
                report.append("")

            except Exception as e:
                report.append("Статус: ОШИБКА")
                report.append(f"{type(e).__name__}: {e}")
                report.append("")
                continue

        report.append(f"Итого обработано документов успешно: {ok_count} из {len(self.uploaded_files)}")
        report.append("")
        report.append("ЭТАП 2. Сравнение документов (LLM)")
        report.append("-" * 70)
        report.append("")
        report.append(f"Модель: {DEFAULT_MODEL_ID} (через transformers)")
        report.append("Критерий связи: ответ LLM {related: true/false} + обоснование")
        report.append("")

        valid_paths = [p for p in self.uploaded_files if p in self.texts and self.texts[p].strip()]
        if len(valid_paths) < 2:
            report.append("Недостаточно документов с текстом для сравнения (нужно минимум 2).")
            return "\n".join(report), links

        total_pairs = 0
        linked_pairs = 0

        report.append("Результаты сравнения пар:")
        report.append("")

        for i in range(len(valid_paths)):
            for j in range(i + 1, len(valid_paths)):
                p1 = valid_paths[i]
                p2 = valid_paths[j]
                t1 = self.texts[p1]
                t2 = self.texts[p2]
                total_pairs += 1

                res = predict_with_reason(t1, t2)
                linked = bool(res.get("related", False))
                reason = str(res.get("reason", "")).strip()
                if linked:
                    linked_pairs += 1

                links.append({"src": p1, "dst": p2, "linked": linked, "reason": reason})

                report.append(f"Пара: {os.path.basename(p1)}  ↔  {os.path.basename(p2)}")
                report.append(f"Вывод: {'СВЯЗАНЫ' if linked else 'НЕ СВЯЗАНЫ'}")
                report.append(f"Обоснование: {reason if reason else '(нет обоснования)'}")
                report.append("")

        report.append(f"Итого сравнений: {total_pairs}")
        report.append(f"Итого связей обнаружено: {linked_pairs}")
        report.append("")

        links_csv = self.save_links_csv(links)
        report.append("Файл результатов связей (links.csv):")
        report.append(f"  {links_csv}")
        report.append("")

        report.append("ЭТАП 3. Формирование графа связей")
        report.append("-" * 70)
        report.append("")

        tree = {p: [] for p in valid_paths}
        for e in links:
            if e["linked"]:
                tree[e["src"]].append({"to": e["dst"], "reason": e.get("reason", "")})
                tree[e["dst"]].append({"to": e["src"], "reason": e.get("reason", "")})

        if linked_pairs == 0:
            report.append("Связи между документами не обнаружены. Граф будет без рёбер.")
        else:
            report.append(f"Связи обнаружены. Рёбер в графе: {linked_pairs}")
            report.append("Для визуализации нажмите кнопку «Построить дерево».")

        tree_json = self.save_tree_json(tree)
        report.append("")
        report.append("Файл структуры графа (tree.json):")
        report.append(f"  {tree_json}")

        return "\n".join(report), links

    # ---------- graph ----------
    def build_tree_visual(self):
        if not self.links:
            self.show_warning_message("Сначала выполните анализ документов.")
            return

        edges = [e for e in self.links if e.get("linked")]
        if not edges:
            QMessageBox.information(self, "Граф связей", "Связей не обнаружено, граф пуст.")
            return

        nodes = list({p for p in self.uploaded_files if p in self.texts})
        labels = {p: os.path.basename(p) for p in nodes}

        w = GraphWindow(nodes, edges, labels)
        w.show()
        self._graph_window = w


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = DocumentAnalyzerApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
