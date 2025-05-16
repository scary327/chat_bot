from sentence_transformers import SentenceTransformer, util
import torch
from datasets import load_dataset
import os

class Retriever:
    def __init__(self, dataset_name="Mykes/rus_med_dialogues", model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", cache_dir="./embeddings"):
        self.model = SentenceTransformer(model_name)
        self.dataset = load_dataset(dataset_name, split="train")
        self.questions = [item['user_question'] for item in self.dataset]
        self.answers = [item['assistant_answer'] for item in self.dataset]
        self.to_doctors = [item['to_doctor'] for item in self.dataset]
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "embeddings.pt")
        if os.path.exists(cache_file):
            print(f"Loading cached embeddings from {cache_file}")
            self.question_embeddings = torch.load(cache_file)
        else:
            print(f"Computing embeddings for {len(self.questions)} questions")
            self.question_embeddings = self.model.encode(self.questions, convert_to_tensor=True, show_progress_bar=True)
            torch.save(self.question_embeddings, cache_file)
            print(f"Saved embeddings to {cache_file}")
    
    def get_retrieved_answer(self, user_input):
        user_embedding = self.model.encode(user_input, convert_to_tensor=True)
        cos_scores = util.cos_sim(user_embedding, self.question_embeddings)[0]
        top_k = 4
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