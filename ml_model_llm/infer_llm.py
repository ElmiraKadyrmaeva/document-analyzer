import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

LOCAL_MODEL_DIR = os.environ.get("DOC_ANALYZER_LLM_PATH", "").strip()


def _get_model_path_or_id():
    return LOCAL_MODEL_DIR if LOCAL_MODEL_DIR else DEFAULT_MODEL_ID


class LLMDocumentLinker:
    def __init__(self, model_id_or_path: str = None, device: str = None):
        self.model_id_or_path = model_id_or_path or _get_model_path_or_id()

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ВАЖНО: low_cpu_mem_usage помогает на слабых ПК
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id_or_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id_or_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(self.device)

        self.model.eval()

    def compare_texts(self, text_a: str, text_b: str, max_chars: int = 3000) -> dict:
        a = (text_a or "")[:max_chars]
        b = (text_b or "")[:max_chars]

        if not a.strip() or not b.strip():
            return {"related": False, "reason": "Один из документов пуст или не содержит текста."}

        system_prompt = (
            "Ты аналитик. Определи, связаны ли два документа семантически "
            "(одна тема, один контекст, версии одного файла или приложение к договору). "
            "Ответь строго JSON без лишнего текста: "
            "{\"related\": boolean, \"reason\": \"короткое обоснование на русском\"}."
        )
        user_content = f"ДОКУМЕНТ 1:\n{a}\n\nДОКУМЕНТ 2:\n{b}\n\nСвязь есть?"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=180,
                do_sample=False
            )

        generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
        response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        try:
            clean = response_text.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean)
            return {
                "related": bool(result.get("related", False)),
                "reason": str(result.get("reason", "")).strip() or "Нет объяснения."
            }
        except Exception:
            return {
                "related": False,
                "reason": "Не удалось распарсить JSON-ответ от LLM.",
                "raw_response": response_text
            }


_inference = None


def load_model():
    global _inference
    if _inference is None:
        _inference = LLMDocumentLinker()
    return _inference


def predict_with_reason(text1: str, text2: str) -> dict:
    model = load_model()
    return model.compare_texts(text1, text2)
