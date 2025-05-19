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
    i=30
    dataset = load_dataset("Mykes/rus_med_dialogues", split="test").select(range(i, 300))
    
    # Имя файла для сохранения промежуточных результатов
    ongoing_file = "Generated_answers_Qwen_ongoing.csv"
    
    # Создаем файл с заголовками, если он не существует
    if not os.path.exists(ongoing_file):
        pd.DataFrame(columns=["id", "question", "ground_truth", "generated_answer"]).to_csv(
            ongoing_file, index=False, encoding="utf-8")

    for item in tqdm(dataset, desc=f"Generating answers"):
        question = item["user_question"]
        answer = item["assistant_answer"]

        contexts = retriever.get_retrieved_answer(question)
        generated_answer = generator.generate_response_sber(question, contexts)

        if generated_answer == "Ошибка при обращении к модели. Попробуйте позже.":
            i += 1
            continue

        result = {
            "id": i,
            "question": question,
            "ground_truth": answer,
            "generated_answer": generated_answer,
        }
        output.append(result)

        # Добавляем новую строку в файл
        pd.DataFrame([result]).to_csv(ongoing_file, mode='a', header=False, index=False, encoding="utf-8")

        i += 1
        if i % 5 == 0:
            time.sleep(120)

    # Сохраняем итоговый файл с полными результатами
    df = pd.DataFrame(output)
    df.to_csv(f"Generated_answers_Qwen.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    retriever = Retriever()
    generator = Generator()
    run_test(retriever, generator)