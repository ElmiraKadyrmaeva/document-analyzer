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
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPen, QBrush

from document_parser import DocumentParser

# ML import
ML_AVAILABLE = False
ML_IMPORT_ERROR = None
try:
    from ml_model.infer import load_model, predict
    try:
        from ml_model.infer import THRESHOLD
    except Exception:
        THRESHOLD = 0.6
    ML_AVAILABLE = True
except Exception as e:
    ML_AVAILABLE = False
    ML_IMPORT_ERROR = e
    THRESHOLD = 0.6

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
    """QGraphicsView с зумом колесом мыши и перетаскиванием сцены"""
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
    """
    Окно с графическим отображением связей
    nodes: list[str] (id документов)
    edges: list[dict] [{src, dst, score}]
    labels: dict[id -> short_name]
    """
    def __init__(self, nodes, edges, labels, threshold):
        super().__init__()
        self.setWindowTitle("Граф связей документов")
        self.setGeometry(150, 150, 1100, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        central.setLayout(layout)

        info = QLabel(f"Узлов: {len(nodes)} | Рёбер: {len(edges)} | Порог: {threshold}")
        info.setFont(QFont("Arial", 11))
        layout.addWidget(info)

        self.scene = QGraphicsScene()
        self.view = GraphView(self.scene)
        layout.addWidget(self.view)

        self._draw(nodes, edges, labels)

    def _layout_positions(self, nodes, edges):
        """
        Возвращает dict[node_id -> (x,y)]
        Если networkx доступен- spring_layout, иначе круг
        """
        if NX_AVAILABLE:
            G = nx.Graph()
            for n in nodes:
                G.add_node(n)
            for e in edges:
                G.add_edge(e["src"], e["dst"], weight=float(e["score"]))

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

        # edges
        for e in edges:
            a = e["src"]
            b = e["dst"]
            x1, y1 = pos[a]
            x2, y2 = pos[b]
            line = QGraphicsLineItem(x1, y1, x2, y2)
            line.setPen(pen_edge)
            line.setZValue(0)
            self.scene.addItem(line)

            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            score_text = QGraphicsTextItem(f"{float(e['score']):.2f}")
            score_text.setDefaultTextColor(Qt.darkGray)
            score_text.setPos(mx + 4, my + 4)
            score_text.setZValue(2)
            self.scene.addItem(score_text)

        # nodes
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
            circle.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
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


class DocumentAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # OCR/парсер
        self.parser = DocumentParser(ocr_lang="rus")

        # outputs dir
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.outputs_dir = os.path.join(self.base_dir, OUTPUTS_DIR_NAME)
        os.makedirs(self.outputs_dir, exist_ok=True)

        # ML модель
        self.ml_ready = False
        self.ml_load_error = None
        self.ml_model = None

        if ML_AVAILABLE:
            try:
                default_pth = os.path.join(self.base_dir, "ml_model", "models", "rubert_siamese_model.pth")
                try:
                    self.ml_model = load_model(default_pth)
                except TypeError:
                    self.ml_model = load_model()
                self.ml_ready = True
            except Exception as e:
                self.ml_ready = False
                self.ml_load_error = e

        # state
        self.uploaded_files = []
        self.texts = {}   # file_path -> extracted text
        self.links = []   # [{src,dst,score,linked}]
        self.tree = {}    # adjacency

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Анализ связей между документами')
        self.setGeometry(100, 100, 980, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        title_label = QLabel('Система анализа документов и построение дерева ссылок')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont('Arial', 16, QFont.Bold))
        main_layout.addWidget(title_label)

        # статус ML
        if not ML_AVAILABLE:
            ml_status = f"ML: недоступен ({type(ML_IMPORT_ERROR).__name__}: {ML_IMPORT_ERROR})"
        else:
            if self.ml_ready:
                ml_status = f"ML: загружен (порог={THRESHOLD})"
            else:
                ml_status = f"ML: не загружен ({type(self.ml_load_error).__name__}: {self.ml_load_error})"

        self.ml_status_label = QLabel(ml_status)
        self.ml_status_label.setWordWrap(True)
        self.ml_status_label.setFont(QFont("Arial", 10))
        main_layout.addWidget(self.ml_status_label)

        # вывод
        self.file_info_text = QTextEdit()
        self.file_info_text.setPlaceholderText('Здесь будет вывод анализа...')
        self.file_info_text.setReadOnly(True)
        main_layout.addWidget(self.file_info_text)

        # кнопки
        buttons_layout = QHBoxLayout()

        self.upload_btn = QPushButton('Загрузить документы')
        self.upload_btn.setFont(QFont('Arial', 12))
        self.upload_btn.clicked.connect(self.upload_files)
        buttons_layout.addWidget(self.upload_btn)

        self.analyze_btn = QPushButton('Анализировать документы')
        self.analyze_btn.setFont(QFont('Arial', 12))
        self.analyze_btn.clicked.connect(self.analyze_documents)
        self.analyze_btn.setEnabled(False)
        buttons_layout.addWidget(self.analyze_btn)

        self.tree_btn = QPushButton('Построить дерево')
        self.tree_btn.setFont(QFont('Arial', 12))
        self.tree_btn.clicked.connect(self.build_tree_visual)
        self.tree_btn.setEnabled(False)
        buttons_layout.addWidget(self.tree_btn)

        self.clear_btn = QPushButton('Очистить файлы')
        self.clear_btn.setFont(QFont('Arial', 12))
        self.clear_btn.clicked.connect(self.clear_files)
        self.clear_btn.setEnabled(False)
        buttons_layout.addWidget(self.clear_btn)

        main_layout.addLayout(buttons_layout)
        self.statusBar().showMessage('Готов к работе. Загрузите документы для анализа.')

    # UI helpers
    def show_error_message(self, message):
        QMessageBox.critical(self, 'Ошибка', message)

    def show_warning_message(self, message):
        QMessageBox.warning(self, 'Предупреждение', message)

    # file handling
    def upload_files(self):
        try:
            file_dialog = QFileDialog()
            files, _ = file_dialog.getOpenFileNames(
                self,
                'Выберите документы для анализа',
                '',
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
        valid_extensions = {'.pdf', '.docx', '.doc', '.jpg', '.jpeg', '.png', '.bmp'}
        new_files = []

        for file_path in files:
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in valid_extensions:
                if file_path not in self.uploaded_files:
                    self.uploaded_files.append(file_path)
                    new_files.append(file_path)
            else:
                self.show_warning_message(
                    f'Файл {os.path.basename(file_path)} имеет неподдерживаемый формат и будет пропущен.'
                )

        if new_files:
            self.update_file_info()
            self.analyze_btn.setEnabled(True)
            self.clear_btn.setEnabled(True)
            self.tree_btn.setEnabled(False)
            self.statusBar().showMessage(
                f'Загружено {len(new_files)} новых документов. Всего документов: {len(self.uploaded_files)}'
            )
        else:
            self.statusBar().showMessage('Новых подходящих документов не найдено.')

    def update_file_info(self):
        lines = []
        lines.append("ЗАГРУЖЕННЫЕ ДОКУМЕНТЫ")
        lines.append("-" * 60)
        for i, file_path in enumerate(self.uploaded_files, 1):
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_path)[1].upper()
            lines.append(f"{i}. {file_name} ({file_ext})")
            lines.append(f"   Путь: {file_path}")
        self.file_info_text.setText("\n".join(lines))

    def clear_files(self):
        if not self.uploaded_files:
            self.show_warning_message("Нет файлов для очистки.")
            return

        reply = QMessageBox.question(
            self,
            'Подтверждение очистки',
            f'Удалить все прикрепленные файлы?\n\nФайлов: {len(self.uploaded_files)}',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.uploaded_files.clear()
            self.texts = {}
            self.links = []
            self.tree = {}
            self.file_info_text.clear()
            self.analyze_btn.setEnabled(False)
            self.tree_btn.setEnabled(False)
            self.clear_btn.setEnabled(False)
            self.statusBar().showMessage('Файлы удалены. Можно загрузить новые документы.')

    # parse/serialize
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
        out = {
            "source_path": getattr(doc, "path", None),
            "source_type": getattr(doc, "source_type", None),
            "pages": []
        }

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
            [
                soffice,
                "--headless",
                "--nologo",
                "--nofirststartwizard",
                "--convert-to", "pdf",
                "--outdir", out_dir,
                word_path
            ],
            check=True
        )

        base = os.path.splitext(os.path.basename(word_path))[0]
        pdf_path = os.path.join(out_dir, base + ".pdf")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"LibreOffice не создал PDF: {pdf_path}")
        return pdf_path

    # outputs
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
            w.writerow(["doc1", "doc2", "score", "linked", "threshold"])
            for e in links:
                w.writerow([e["src"], e["dst"], f"{e['score']:.6f}", int(e["linked"]), THRESHOLD])
        return path

    def save_tree_json(self, tree: dict) -> str:
        path = os.path.join(self.outputs_dir, "tree.json")
        payload = {
            "threshold": THRESHOLD,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "tree": tree
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

    def analyze_documents(self):
        """
        ЭТАП 1: извлечение текста + сохранение TXT/JSON
        ЭТАП 2: ML скоринг всех пар + сохранение links.csv
        ЭТАП 3: формирование структуры графа (adjacency) + сохранение tree.json
        """
        if not self.uploaded_files:
            self.show_warning_message("Нет документов для анализа.")
            return

        self.texts = {}
        self.links = []
        self.tree = {}

        report = []
        report.append("СИСТЕМА АНАЛИЗА ДОКУМЕНТОВ")
        report.append("Отчёт о выполнении анализа")
        report.append("")
        report.append("1. Извлечение текста из документов")
        report.append("-" * 60)
        report.append("")

        # ---------- ЭТАП 1 ----------
        parsed_count = 0
        for path in self.uploaded_files:
            base = os.path.basename(path)
            base_noext = os.path.splitext(base)[0]

            if not os.path.exists(path):
                report.append("[ОШИБКА]")
                report.append(f"Документ: {base}")
                report.append(f"Источник: {path}")
                report.append("Причина: файл не найден")
                report.append("")
                continue

            try:
                ext = os.path.splitext(path)[1].lower()
                parse_path = path
                converted_info = None

                if ext in (".doc", ".docx"):
                    parse_path = self.convert_word_to_pdf(path)
                    converted_info = f"Выполнено преобразование Word → PDF: {parse_path}"

                doc = self.parser.parse(parse_path)
                text = self.parsed_doc_to_text(doc, max_pages=MAX_PAGES_TEXT)
                self.texts[path] = text

                txt_path = self.save_txt(base_noext, text)
                json_dict = self.serialize_doc_to_dict(doc, max_pages=MAX_PAGES_JSON)
                json_path = self.save_json(base_noext, json_dict)

                parsed_count += 1

                report.append("[УСПЕШНО]")
                report.append(f"Документ: {base}")
                report.append(f"Источник: {path}")
                if converted_info:
                    report.append(converted_info)
                report.append(f"Количество символов (первые {MAX_PAGES_TEXT} страницы): {len(text)}")
                report.append("Результаты сохранены:")
                report.append(f"  - Текстовый файл: {txt_path}")
                report.append(f"  - Структурированный JSON: {json_path}")

                fragment = (text or "").replace("\r", "").strip()
                fragment = fragment[:200]
                if fragment:
                    report.append("Фрагмент текста:")
                    report.append(f'  "{fragment}"')
                else:
                    report.append("Фрагмент текста: отсутствует (пустой текст)")

                report.append("")

            except Exception as e:
                report.append("[ОШИБКА]")
                report.append(f"Документ: {base}")
                report.append(f"Источник: {path}")
                report.append(f"Тип ошибки: {type(e).__name__}")
                report.append(f"Сообщение: {e}")
                report.append("Трассировка:")
                report.append(traceback.format_exc().rstrip())
                report.append("")

        # ---------- ЭТАП 2 ----------
        report.append("2. Семантический анализ и поиск связей")
        report.append("-" * 60)
        report.append("")

        if not self.ml_ready:
            report.append("ML-модель недоступна. Этап семантического сравнения пропущен.")
            if self.ml_load_error:
                report.append(f"Причина: {type(self.ml_load_error).__name__}: {self.ml_load_error}")
            elif (not ML_AVAILABLE) and ML_IMPORT_ERROR:
                report.append(f"Причина: {type(ML_IMPORT_ERROR).__name__}: {ML_IMPORT_ERROR}")
            report.append("")
            self.file_info_text.setText("\n".join(report))
            self.tree_btn.setEnabled(False)
            return

        valid_paths = [p for p in self.uploaded_files if p in self.texts and self.texts[p].strip()]
        if len(valid_paths) < 2:
            report.append("Недостаточно документов с извлечённым текстом для сравнения (нужно минимум 2).")
            report.append("")
            self.file_info_text.setText("\n".join(report))
            self.tree_btn.setEnabled(False)
            return

        report.append("Используемая модель: Siamese RuBERT")
        report.append(f"Порог определения связи: {THRESHOLD:.2f}")
        report.append("")
        report.append("Результаты сравнения:")
        report.append("")

        total_pairs = 0
        linked_pairs = 0

        for i in range(len(valid_paths)):
            for j in range(i + 1, len(valid_paths)):
                p1 = valid_paths[i]
                p2 = valid_paths[j]
                t1 = self.texts[p1]
                t2 = self.texts[p2]
                total_pairs += 1

                try:
                    score = float(predict(t1, t2))
                    linked = score >= THRESHOLD
                    if linked:
                        linked_pairs += 1

                    self.links.append({
                        "src": p1,
                        "dst": p2,
                        "score": score,
                        "linked": linked
                    })

                    report.append(f"Пара документов:")
                    report.append(f"  Документ A: {os.path.basename(p1)}")
                    report.append(f"  Документ B: {os.path.basename(p2)}")
                    report.append(f"  Коэффициент схожести: {score:.4f}")
                    report.append(f"  Вывод: {'связь обнаружена' if linked else 'документы не связаны'}")
                    report.append("")

                except Exception as e:
                    report.append("Пара документов:")
                    report.append(f"  Документ A: {os.path.basename(p1)}")
                    report.append(f"  Документ B: {os.path.basename(p2)}")
                    report.append("  Ошибка вычисления схожести")
                    report.append(f"  Тип ошибки: {type(e).__name__}")
                    report.append(f"  Сообщение: {e}")
                    report.append("")

        report.append(f"Общее количество сравнений: {total_pairs}")
        report.append(f"Количество выявленных связей: {linked_pairs}")
        report.append("")

        links_csv = self.save_links_csv(self.links)
        report.append("Результаты сохранены в файл:")
        report.append(f"  {links_csv}")
        report.append("")

        # ---------- ЭТАП 3 ----------
        report.append("3. Формирование графа связей")
        report.append("-" * 60)
        report.append("")

        tree = {p: [] for p in valid_paths}
        for e in self.links:
            if e["linked"]:
                tree[e["src"]].append({"to": e["dst"], "score": e["score"]})
                tree[e["dst"]].append({"to": e["src"], "score": e["score"]})

        self.tree = tree

        if linked_pairs == 0:
            report.append("По заданному пороговому значению связи между документами не выявлены.")
            report.append("Граф не содержит рёбер.")
        else:
            report.append("Связи между документами выявлены.")
            report.append(f"Количество рёбер в графе: {linked_pairs}")
            report.append("Для визуализации нажмите кнопку «Построить дерево».")

        tree_json = self.save_tree_json(tree)
        report.append("")
        report.append("Структура графа сохранена в файл:")
        report.append(f"  {tree_json}")
        report.append("")

        self.file_info_text.setText("\n".join(report))
        self.tree_btn.setEnabled(True)

    # ---------------- графика ----------------
    def build_tree_visual(self):
        if not self.uploaded_files or not self.links:
            self.show_warning_message("Сначала выполните анализ документов.")
            return

        nodes = list({p for p in self.uploaded_files if p in self.texts})
        labels = {p: os.path.basename(p) for p in nodes}
        edges = [e for e in self.links if e["linked"]]

        if len(edges) == 0:
            QMessageBox.information(
                self,
                "Граф связей",
                f"По порогу {THRESHOLD:.2f} связи не обнаружены.\nГраф не содержит рёбер."
            )
            return

        w = GraphWindow(nodes, edges, labels, THRESHOLD)
        w.show()
        self._graph_window = w


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = DocumentAnalyzerApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
