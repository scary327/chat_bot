import token
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
from transformers.pipelines import pipeline
import logging
import os
import requests
from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Generator:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct-Turbo"):
        self.model_name = model_name
        self.API_URL = "https://router.huggingface.co/together/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {os.getenv('MODEL')}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    def generate_response_sber(self, user_input, retrieved_contexts):
        context_str = "\n".join([ctx['answer'] for ctx in retrieved_contexts])
        prompt = (
            f"Вы — медицинский ассистент. Ваша задача — ответить на вопрос пользователя, используя ТОЛЬКО информацию из предоставленного контекста. "
            f"Не добавляйте никаких новых данных, не давайте советов, не упоминая контекст."
            f"Формат ответа: только текст ответа, без повторения вопроса или контекста. Ответ должен быть максимально полным и подробным и если есть рекомендации врача.\n\n"
            f"Контекст: {context_str}\n"
            f"Вопрос: {user_input}\n"
            f"Ответ:"
        )

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1024,
            "model": self.model_name
        }
        
        try:
            response = requests.post(
                self.API_URL,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            print(response.json())
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"]

            if "Ответ:" in result:
                result = result.split("Ответ:")[-1].strip()

            return result

        except Exception as e:
            print(f"[API Error] {e}")
            return "Ошибка при обращении к модели. Попробуйте позже."