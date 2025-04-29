import token
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration
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