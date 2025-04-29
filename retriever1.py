from sentence_transformers import SentenceTransformer, util
import torch
from datasets import load_dataset

class Retriever:
    def __init__(self, dataset_name="Mykes/rus_med_dialogues", model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        self.dataset = load_dataset(dataset_name, split="train")
        self.questions = [item['user_question'] for item in self.dataset]
        self.answers = [item['assistant_answer'] for item in self.dataset]
        self.to_doctors = [item['to_doctor'] for item in self.dataset]
        self.question_embeddings = self.model.encode(self.questions, convert_to_tensor=True)
    
    def get_retrieved_answer(self, user_input):
        user_embedding = self.model.encode(user_input, convert_to_tensor=True)
        cos_scores = util.cos_sim(user_embedding, self.question_embeddings)[0]
        top_k = 3
        top_results = torch.topk(cos_scores, k=top_k)
        results = []
        for idx, score in zip(top_results.indices, top_results.values):
            results.append({
                'question': self.questions[idx],
                'answer': self.answers[idx],
                'to_doctor': self.to_doctors[idx],
                'score': score.item()
            })
        return results