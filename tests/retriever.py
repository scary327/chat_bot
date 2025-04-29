from sentence_transformers import SentenceTransformer, util
import torch
from datasets import load_dataset

class Retriever:
    def __init__(self, model_name, dataset_name="Mykes/rus_med_dialogues"):
        self.model = SentenceTransformer(model_name)
        self.dataset = load_dataset(dataset_name, split="train")
        self.questions = [item['user_question'] for item in self.dataset]
        self.answers = [item['assistant_answer'] for item in self.dataset]
        self.to_doctors = [item['to_doctor'] for item in self.dataset]
        self.question_embeddings = self.model.encode(self.questions, convert_to_tensor=True)
    
    def get_retrieved_answers(self, user_input):
        user_embedding = self.model.encode(user_input, convert_to_tensor=True)
        cos_scores = util.cos_sim(user_embedding, self.question_embeddings)[0]
        top_k = 3
        top_results = torch.topk(cos_scores, k=top_k)
        results = []
        for idx, score in zip(top_results.indices, top_results.values):
            results.append(self.answers[idx])
        return results