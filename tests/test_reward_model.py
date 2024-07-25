import pytest
from src.reward_model import train_reward_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Требуется GPU")
def test_train_reward_model():
    train_reward_model()
    assert os.path.exists('results/models/reward_model')
    
    model = AutoModelForSequenceClassification.from_pretrained('results/models/reward_model')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')
    
    test_texts = [
        "This movie was fantastic!",
        "Absolutely terrible film."
    ]
    inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    assert logits.shape == (2, 2)  # 2 примера, 2 класса (положительный/отрицательный)
    assert logits[0][1] > logits[0][0]  # Позитивный отзыв должен иметь более высокий скор для положительного класса
    assert logits[1][0] > logits[1][1]  # Негативный отзыв должен иметь более высокий скор для отрицательного класса