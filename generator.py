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
        context_str = "\n".join([f"{ctx['answer']}" for i, ctx in enumerate(retrieved_contexts)])
        prompt = (
            f"Вы — врач. Ответьте на вопрос пользователя, используя ТОЛЬКО информацию из КОНТЕКСТА. "
            f"Не добавляйте ничего нового, не угадывайте, не советуйте обращаться к кому-либо ещё. "
            f"Если в контексте нет ответа — напишите: 'Информации для ответа недостаточно'.\n\n"
            f"Вопрос: {user_input}\n"
            f"Контекст: {context_str}\n"
            f"Ответ:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, padding=True, return_attention_mask=True)
        
        # Генерация ответа
        outputs = self.model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=100,
            num_beams=5,
            early_stopping=True,
            top_k=10,
            no_repeat_ngram_size=3,
            temperature=0.1,
            do_sample=False,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    

    def generate_response(self, user_input, retrieved_contexts):
        context_str = "\n".join([f"Контекст {i+1}: {ctx['answer']}" for i, ctx in enumerate(retrieved_contexts)])
        logger.info(f"User input: {user_input}")
        logger.info(f"Retrieved contexts:\n{context_str}")
        
        prompt = (
            f"На основе вопроса и контекста сгенерируй краткий ответ на русском языке:\n"
            f"Вопрос: {user_input}\n"
            f"Контекст: {context_str}\n"
            f"Ответ:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        outputs = self.model.generate(
            inputs['input_ids']
        )
        
        response = self.tokenizer.decode(outputs[0])
        logger.info(f"Raw response: {response}")

        to_doctor = retrieved_contexts[0]['to_doctor'] if retrieved_contexts else "врачу"
        final_response = f"{response} Обратитесь к {to_doctor} для консультации."
        logger.info(f"Final response: {final_response}")
        
        return final_response