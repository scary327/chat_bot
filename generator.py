import token
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Generator:
    def __init__(self, model_name="mistralai/Mistral-Nemo-Instruct-2407"):
        self.token=os.getenv("MODEL")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=self.token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=self.token
        )
    
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
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        outputs = self.model.generate(
            inputs['input_ids'],
            max_length=250, 
            num_beams=1,
            no_repeat_ngram_size=4,
            length_penalty=1.0,
            do_sample=False,
            forced_bos_token_id=self.tokenizer.get_lang_id("ru")
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Raw response: {response}")

        to_doctor = retrieved_contexts[0]['to_doctor'] if retrieved_contexts else "врачу"
        final_response = f"{response} Обратитесь к {to_doctor} для консультации."
        logger.info(f"Final response: {final_response}")
        
        return final_response