import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import os
import warnings
import logging

# ============================================================================
# КОНФИГУРАЦИЯ WARNINGS
# ============================================================================

# Подавить все предупреждения от библиотек
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Подавить лишние логи от HuggingFace и PyTorch
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# ============================================================================
# КОНСТАНТЫ
# ============================================================================

THRESHOLD = 0.6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Пути к моделям
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "rubert_siamese_model.pth")


# ============================================================================
# МОДЕЛЬ
# ============================================================================

class SiameseRuBERT(nn.Module):
    """
    Siamese архитектура на базе RuBERT для сравнения семантической похожести
    двух текстов. Использует конкатенцию embeddings и их разности.
    """

    def __init__(self, model_name="DeepPavlov/rubert-base-cased"):
        super(SiameseRuBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768 для base-моделей

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, 512),  # 768*3=2304
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward_one(self, inputs):
        """
        Получить embedding текста из [CLS] токена

        Args:
            inputs: dict с 'input_ids' и 'attention_mask'

        Returns:
            Tensor размера (batch_size, hidden_size)
        """
        outputs = self.bert(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        # Берем эмбеддинг [CLS] токена (первый токен последовательности)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

    def forward(self, doc1_inputs, doc2_inputs):
        """
        Вычислить вероятность похожести между двумя текстами

        Args:
            doc1_inputs: dict с токенизированным первым текстом
            doc2_inputs: dict с токенизированным вторым текстом

        Returns:
            Tensor с вероятностями (batch_size, 1)
        """
        emb1 = self.forward_one(doc1_inputs)
        emb2 = self.forward_one(doc2_inputs)

        # Классическая сиамская конкатенация:
        # [embedding1, embedding2, |embedding1 - embedding2|]
        combined_features = torch.cat(
            (emb1, emb2, torch.abs(emb1 - emb2)),
            dim=1
        )

        return self.classifier(combined_features)


# ============================================================================
# ЗАГРУЗКА И ИНФЕРЕНС
# ============================================================================

class SiameseInference:
    """
    Удобный интерфейс для инференса Siamese RuBERT модели
    """

    def __init__(self, model_path, model_name="DeepPavlov/rubert-base-cased"):
        """
        Инициализировать модель и токенайзер

        Args:
            model_path: путь к .pth файлу с весами модели
            model_name: название модели RuBERT для загрузки архитектуры
        """
        self.device = DEVICE
        self.model = self._load_model(model_path, model_name)
        self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")

    def _load_model(self, model_path, model_name):
        """
        Загрузить модель из checkpoint

        Args:
            model_path: путь к .pth файлу
            model_name: название базовой модели

        Returns:
            Model в eval режиме на нужном device
        """
        model = SiameseRuBERT(model_name=model_name)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def predict(self, text1: str, text2: str, return_logits=False) -> float:
        """
        Предсказать вероятность похожести двух текстов

        Args:
            text1: первый текст
            text2: второй текст
            return_logits: если True, вернуть сырые логиты, иначе вероятность

        Returns:
            Вероятность похожести (0..1) или логит
        """
        # Токенизация
        inputs1 = self.tokenizer(
            text1,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs2 = self.tokenizer(
            text2,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Переместить на device
        inputs1 = {k: v.to(self.device) for k, v in inputs1.items()}
        inputs2 = {k: v.to(self.device) for k, v in inputs2.items()}

        # Инференс
        with torch.no_grad():
            logits = self.model(inputs1, inputs2)

        # Вернуть вероятность
        probability = logits.item()
        return probability

    def predict_batch(self, text_pairs: list) -> list:
        """
        Предсказать вероятности для батча пар текстов

        Args:
            text_pairs: список кортежей (text1, text2)

        Returns:
            Список вероятностей
        """
        probabilities = []
        for text1, text2 in text_pairs:
            prob = self.predict(text1, text2)
            probabilities.append(prob)
        return probabilities

    def is_similar(self, text1: str, text2: str, threshold=THRESHOLD) -> bool:
        """
        Проверить, похожи ли тексты (бинарная классификация)

        Args:
            text1: первый текст
            text2: второй текст
            threshold: порог для классификации (по умолчанию 0.6)

        Returns:
            True если вероятность >= threshold, иначе False
        """
        probability = self.predict(text1, text2)
        return probability >= threshold


# ============================================================================
# ФУНКЦИИ ДЛЯ ОБРАТНОЙ СОВМЕСТИМОСТИ
# ============================================================================

_inference_instance = None


def load_model(path_to_pth: str = None, model_name="DeepPavlov/rubert-base-cased") -> SiameseInference:
    """
    Загрузить модель из checkpoint

    Args:
        path_to_pth: путь к .pth файлу с весами модели
                    Если None, использует DEFAULT_MODEL_PATH (ml_model/models/rubert_siamese_model.pth)
        model_name: название модели RuBERT

    Returns:
        Объект SiameseInference

    Raises:
        FileNotFoundError: если файл модели не найден
    """
    global _inference_instance

    if path_to_pth is None:
        path_to_pth = DEFAULT_MODEL_PATH

    if not os.path.exists(path_to_pth):
        raise FileNotFoundError(
            f"❌ Модель не найдена по пути: {path_to_pth}\n"
            f"📁 Убедитесь, что файл находится в: {MODEL_DIR}/"
        )

    print(f"📦 Загрузка модели из: {path_to_pth}")
    _inference_instance = SiameseInference(path_to_pth, model_name)
    print(f"✅ Модель успешно загружена на {DEVICE}")
    return _inference_instance


def predict(text1: str, text2: str) -> float:
    """
    Предсказать вероятность похожести двух текстов
    Требует предварительной загрузки модели через load_model()

    Args:
        text1: первый текст
        text2: второй текст

    Returns:
        Вероятность похожести (0..1)

    Raises:
        RuntimeError: если модель не загружена
    """
    if _inference_instance is None:
        raise RuntimeError(
            "❌ Модель не загружена. Сначала вызовите load_model()"
        )
    return _inference_instance.predict(text1, text2)


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

if __name__ == "__main__":
    # Вариант 1: Через функции (автоматическая загрузка)
    print("=== Вариант 1: load_model() без аргументов ===")
    try:
        model_inf = load_model()  # Автоматически загружает из ml_model/models/
        prob = predict(
            "Это первый текст для сравнения",
            "Это похожий текст для сравнения"
        )
        print(f"Вероятность похожести: {prob:.4f}\n")
    except FileNotFoundError as e:
        print(f"Ошибка: {e}\n")

    # Вариант 2: Через класс SiameseInference
    print("=== Вариант 2: SiameseInference ===")
    try:
        inference = SiameseInference(DEFAULT_MODEL_PATH)

        text1 = "Москва - столица России"
        text2 = "Россия имеет столицу Москву"

        prob = inference.predict(text1, text2)
        print(f"Текст 1: {text1}")
        print(f"Текст 2: {text2}")
        print(f"Вероятность похожести: {prob:.4f}")
        print(f"Похожи ли? {inference.is_similar(text1, text2)} (порог: {THRESHOLD})\n")

        # Батч инференс
        print("=== Батч инференс ===")
        pairs = [
            ("Кот сидит на столе", "Кот находится на столе"),
            ("Красивая погода", "Идет дождь"),
        ]
        probs = inference.predict_batch(pairs)
        for (t1, t2), prob in zip(pairs, probs):
            print(f"{t1} <-> {t2}: {prob:.4f}")

    except FileNotFoundError as e:
        print(f"❌ Ошибка: {e}")