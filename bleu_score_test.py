import time
from generator_api import Generator
from retriever1 import Retriever
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import pandas as pd
import os


def run_test(retriever, generator):
    bleu_scores = []
    output = []
    i=0
    dataset = load_dataset("Mykes/rus_med_dialogues", split="test").select(range(150, 200))
    
    # Имя файла для сохранения промежуточных результатов
    ongoing_file = "BLEU_score_results_(100-150)_promt.csv"
    
    # Создаем файл с заголовками, если он не существует
    if not os.path.exists(ongoing_file):
        pd.DataFrame(columns=["question", "ground_truth", "generated_answer", "bleu_score"]).to_csv(
            ongoing_file, index=False, encoding="utf-8")

    for item in tqdm(dataset, desc=f"BLEU score test"):
        question = item["user_question"]
        answer = item["assistant_answer"]

        contexts = retriever.get_retrieved_answer(question)
        generated_answer = generator.generate_response_sber(question, contexts)

        if generated_answer == "Ошибка при обращении к модели. Попробуйте позже.":
            i += 1
            continue

        bleu_score = sentence_bleu([answer], generated_answer)
        bleu_scores.append(bleu_score)

        result = {
            "question": question,
            "ground_truth": answer,
            "generated_answer": generated_answer,
            "bleu_score": bleu_score
        }
        output.append(result)

        # Добавляем новую строку в файл
        pd.DataFrame([result]).to_csv(ongoing_file, mode='a', header=False, index=False, encoding="utf-8")

        i += 1
        if i % 5 == 0:
            time.sleep(180)

    avg_blue = sum(bleu_scores) / len(bleu_scores) if len(bleu_scores) != 0 else 0
    print(f"Average BLEU score: {avg_blue:.4f}")

    # Сохраняем итоговый файл с полными результатами
    df = pd.DataFrame(output)
    df.to_csv(f"BLEU score test (final, bleu={avg_blue:.4f}).csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    retriever = Retriever()
    generator = Generator()
    run_test(retriever, generator)