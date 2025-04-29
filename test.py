import pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score
from retriever import Retriever

def evaluate_f1(dataset_name="Mykes/rus_med_dialogues", model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
    test_dataset = load_dataset(dataset_name, split="test")
    test_data = test_dataset.to_pandas()
    
    if 'user_question' not in test_data.columns or 'assistant_answer' not in test_data.columns:
        raise ValueError("Тестовый датасет должен содержать колонки 'user_question' и 'assistant_answer'.")
    
    retriever = Retriever(dataset_name, model_name)
    
    y_true = []
    y_pred = []
    
    for idx, row in test_data.iterrows():
        question = row['user_question']
        true_answer = row['assistant_answer']
        
        # Поиск ближайшего ответа
        _, pred_answer, _ = retriever.get_retrieved_answer(question)
        
        y_true.append(true_answer)
        y_pred.append(pred_answer)
    
    # Бинарная классификация для F1 (1 если ответ совпадает, 0 если нет)
    y_true_bin = [1 if true == pred else 0 for true, pred in zip(y_true, y_pred)]
    y_pred_bin = [1] * len(y_true_bin)  # Упрощение для F1
    
    f1 = f1_score(y_true_bin, y_pred_bin)
    print(f"F1-score: {f1:.2f}")
    return f1

if __name__ == "__main__":
    evaluate_f1()