import yaml
import datasets
from typing import List, Tuple

def load_config() -> dict:
    with open('configs/config.yaml', 'r') as file:
        return yaml.safe_load(file)

def load_imdb_dataset() -> datasets.DatasetDict:
    """Загрузка датасета IMDB"""
    config = load_config()
    return datasets.load_dataset(config['data']['dataset_name'])

def create_pairs(dataset: datasets.Dataset) -> List[Tuple[str, str]]:
    """Создание пар положительных и отрицательных комментариев"""
    positive = dataset.filter(lambda x: x['label'] == 1)
    negative = dataset.filter(lambda x: x['label'] == 0)
    pairs = []
    for pos, neg in zip(positive, negative):
        pairs.append((pos['text'], neg['text']))
    return pairs

def prepare_data() -> List[Tuple[str, str]]:
    """Подготовка данных для обучения модели наград"""
    dataset = load_imdb_dataset()
    return create_pairs(dataset['train'])

if __name__ == "__main__":
    train_pairs = prepare_data()
    print(f"Подготовлено {len(train_pairs)} пар для обучения")