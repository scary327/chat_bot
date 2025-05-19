import os
import pandas as pd
import numpy as np
from datetime import datetime
from bert_score import score

def calculate_f1_score(file_path):
    """Рассчитывает средний F1 с использованием BERTScore из CSV файла."""
    try:
        # Загружаем CSV файл
        df = pd.read_csv(file_path)
        
        # Проверяем наличие необходимых столбцов
        if 'ground_truth' not in df.columns or 'generated_answer' not in df.columns:
            print(f"Ошибка: в файле {file_path} отсутствуют необходимые столбцы 'ground_truth' или 'generated_answer'")
            return None, 0
        
        # Удаляем строки с NaN значениями
        df = df.dropna(subset=['ground_truth', 'generated_answer'])
        
        # Если после удаления NaN строк не осталось данных
        if len(df) == 0:
            print(f"Ошибка: в файле {file_path} нет валидных значений")
            return None, 0
        
        # Получаем списки для сравнения
        references = df['ground_truth'].tolist()
        candidates = df['generated_answer'].tolist()
        count = len(df)
        
        # Вычисляем BERTScore
        try:
            P, R, F1 = score(candidates, references, lang="ru", verbose=True)
            bert_f1 = F1.mean().item()  # Среднее значение F1
            print(f"BERTScore F1 для {file_path}: {bert_f1:.4f}")
            return bert_f1, count
        except Exception as e:
            print(f"Ошибка при вычислении BERTScore для {file_path}: {str(e)}")
            return None, 0
            
    except Exception as e:
        print(f"Ошибка при обработке файла {file_path}: {str(e)}")
        return None, 0

def main():
    # Путь к папке с CSV файлами
    scores_dir = "/Users/daniilterentev/Desktop/DPO_project/f1_global"
    
    # Путь к выходному файлу
    output_file = f"/Users/daniilterentev/Desktop/DPO_project/f1_scores_global_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Список для хранения результатов
    results = []
    
    # Обрабатываем все CSV файлы в папке
    for filename in os.listdir(scores_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(scores_dir, filename)
            f1_score, count = calculate_f1_score(file_path)
            
            if f1_score is not None:
                results.append({
                    'file_name': filename,
                    'f1_score': f1_score,
                })
                print(f"Файл: {filename}, F1 Score: {f1_score:.4f}, Количество примеров: {count}")
    
    # Создаем DataFrame с результатами
    if results:
        results_df = pd.DataFrame(results)
        
        # Сортируем по F1 score в убывающем порядке
        results_df = results_df.sort_values(by='f1_score', ascending=False)
        
        # Сохраняем в CSV
        results_df.to_csv(output_file, index=False)
        print(f"\nРезультаты F1 сохранены в файл: {output_file}")
    else:
        print("Не найдено CSV файлов с валидными данными.")

if __name__ == "__main__":
    main() 