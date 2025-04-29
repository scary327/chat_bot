from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
import torch
from retriever import Retriever
from generator import Generator
from datasets import load_dataset, concatenate_datasets
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import pandas as pd
import json


class ModelTester:
    def __init__(self, generator_model, stranformer, dataset_name="Mykes/rus_med_dialogues"):
        dataset_train = load_dataset(dataset_name, split="train")
        dataset_test = load_dataset(dataset_name, split="test")

        self.dataset = concatenate_datasets([dataset_train, dataset_test])
        self.name = generator_model['name'] + " + " + stranformer

        self.retriever = Retriever(stranformer)
        self.generator = Generator(
            model_name=generator_model['name'],
            tokenizer_class=generator_model['tokenizer'],
            model_class=generator_model['model']
        )

    def calculate_bleu(self, ref, cand):
        return sentence_bleu(ref, cand)

    def run_test(self):
        bleu_scores = []
        output = []
        i=0
        for item in tqdm(self.dataset, desc=f"{self.name}"):
            question = item["user_question"]
            answer = item["assistant_answer"]
            
            contexts = self.retriever.get_retrieved_answers(question)
            generated_answer = self.generator.generate_answer(question, contexts)
            
            bleu_score = self.calculate_bleu([answer], generated_answer)

            bleu_scores.append(bleu_score)

            output.append({
                "question": question,
                "ground_truth": answer,
                "contexts": "\n".join(contexts),
                "generated_answer": generated_answer,
                "bleu_score": bleu_score
            })
            i+=1
            if i == 4: break

        avg_blue = sum(bleu_scores) / len(bleu_scores) if len(bleu_scores) != 0 else 0
        print(f"Average BLEU score: {avg_blue:.4f}")

        df = pd.DataFrame(output)
        df.to_csv(f"{self.name.replace("/", "--")} (bleu={avg_blue}).csv", index=False, encoding="utf-8")

    


if __name__ == "__main__":
    transformers = [
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        # "sentence-transformers/LaBSE",
        # "intfloat/multilingual-e5-large"
    ]
    generator_models = [
        {
            "name": "sberbank-ai/rugpt3large_based_on_gpt2",
            "tokenizer": GPT2Tokenizer,
            "model": GPT2LMHeadModel
        }
    ]
    for transformer in transformers:
        for generator in generator_models:
            test = ModelTester(
                generator_model=generator,
                stranformer=transformer
            )
            test.run_test()