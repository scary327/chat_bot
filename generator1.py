import token
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
from transformers.pipelines import pipeline
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class Generator:
    def __init__(self, model_name="sberbank-ai/rugpt3large_based_on_gpt2"):
        self.token=os.getenv("MODEL")
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            model_name,
            token=self.token
        )
        self.model = GPT2LMHeadModel.from_pretrained(
            model_name,
            token=self.token
        )
    
    def generate_response_sber(self, user_input, retrieved_contexts):
        context_str = "\n".join([ctx['answer'] for ctx in retrieved_contexts])
        prompt = (
            f"Вы — медицинский ассистент. Ваша задача — ответить на вопрос пользователя, используя ТОЛЬКО информацию из предоставленного контекста. "
            f"Не добавляйте никаких новых данных, не давайте советов, не упоминая контекст, и не предлагайте обращаться к врачам или другим специалистам. "
            f"Если в контексте нет достаточно информации для ответа, напишите: 'Информации для ответа недостаточно'. "
            f"Формат ответа: только текст ответа, без повторения вопроса или контекста.\n\n"
            f"Контекст:\n{context_str}\n\n"
            f"Вопрос: {user_input}\n\n"
            f"Ответ:"
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
            return_attention_mask=True
            )
        
        # Генерация ответа
        outputs = self.model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=63,
            num_beams=13,  
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Ответ:" in response:
            response = response.split("Ответ:")[-1].strip()

        return response