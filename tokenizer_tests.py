from sentence_transformers import SentenceTransformer, util
import torch
from datasets import load_dataset, concatenate_datasets
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import pandas as pd
import json

class TokenizerTest:
    def __init__(self, model_name: str, dataset_name="Mykes/rus_med_dialogues"):
        if not model_name:
            raise Exception("No model name given")
        
        self.model = SentenceTransformer(model_name)
        dataset_train = load_dataset(dataset_name, split="train")
        dataset_test = load_dataset(dataset_name, split="test")
        self.dataset = concatenate_datasets([dataset_train, dataset_test])
        self.model_name = model_name

        self.questions = [item["user_question"] for item in self.dataset]
        self.answers = [item["assistant_answer"] for item in self.dataset]
        self.question_embeddings = self.model.encode(self.questions, convert_to_tensor=True, show_progress_bar=True)

    def calculate_bleu(self, ref, cand):
        if not isinstance(ref, list):
            ref = list(ref)
        return sentence_bleu(ref, cand)

    def find_relevant_contexts(self, user_input, top_k=3):
        user_embedding = self.model.encode(user_input, convert_to_tensor=True)
        cos_scores = util.cos_sim(user_embedding, self.question_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        results = []
        for idx, score in zip(top_results.indices, top_results.values):
            results.append(self.answers[idx])
        return results
    
    def run_test(self, top_k=3):
        bleu_scores = []
        output = []
        # Итерируемся по тестовому датасету
        for item in tqdm(self.dataset, desc="Тестируем штучки"):
            question = item["user_question"]
            answer = item["assistant_answer"]
            
            # Находим топ-k релевантных контекстов из train
            contexts = self.find_relevant_contexts(question, top_k=top_k)
            
            # Вычисляем BLEU между каждым контекстом и ground truth
            bleu_score = self.calculate_bleu(contexts, answer)
            bleu_scores.append(bleu_score)
            output.append({
                "question": question,
                "ground_truth": answer,
                "contexts": " || ".join(contexts),  # Объединяем контексты в строку
                "bleu_score": bleu_score
            })

        avg_blue = sum(bleu_scores) / len(bleu_scores) if len(bleu_scores) != 0 else 0
        print(f"Average BLEU score: {avg_blue:.4f}")

        json_data = {
            "csv file": f"{self.model_name} (k={top_k}).csv",
            "top_k": top_k,
            "avg_bleu": avg_blue,
        }

        df = pd.DataFrame(output)
        df.to_csv(f"{self.model_name.split("/")[1]} (k={top_k}) (bleu={avg_blue}).csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    model_name = "sentence-transformers/LaBSE"
    tester = TokenizerTest(model_name)
    for i in range(3, 6):
        tester.run_test(top_k=i)
    