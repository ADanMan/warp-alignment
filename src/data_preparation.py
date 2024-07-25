import yaml
from datasets import load_dataset
from typing import List, Tuple

def load_config() -> dict:
    with open('configs/config.yaml', 'r') as file:
        return yaml.safe_load(file)

def load_imdb_dataset():
    """Загрузка датасета IMDB"""
    return load_dataset("stanfordnlp/imdb")

def create_pairs(dataset) -> List[Tuple[str, str]]:
    """Создание пар положительных и отрицательных комментариев"""
    positive = [item for item in dataset if item['label'] == 1]
    negative = [item for item in dataset if item['label'] == 0]
    return list(zip(positive, negative))

def prepare_data() -> List[Tuple[str, str]]:
    """Подготовка данных для обучения модели наград"""
    dataset = load_imdb_dataset()
    pairs = create_pairs(dataset['train'])
    return [(pos['text'], neg['text']) for pos, neg in pairs]

if __name__ == "__main__":
    train_pairs = prepare_data()
    print(f"Подготовлено {len(train_pairs)} пар для обучения")
    print("Пример пары:")
    print(f"Положительный отзыв: {train_pairs[0][0][:100]}...")
    print(f"Отрицательный отзыв: {train_pairs[0][1][:100]}...")