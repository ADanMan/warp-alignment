import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForSequenceClassification
from data_preparation import load_config, load_imdb_dataset

def evaluate_models():
    """Оценка моделей WARP и SFT"""
    config = load_config()
    
    # Загрузка моделей
    warp_model = GPT2LMHeadModel.from_pretrained(config['warp']['output_dir'])
    sft_model = GPT2LMHeadModel.from_pretrained(config['warp']['model_name'])
    reward_model = AutoModelForSequenceClassification.from_pretrained(config['reward_model']['output_dir'])
    tokenizer = GPT2Tokenizer.from_pretrained(config['warp']['model_name'])
    
    # Подготовка тестовых промптов
    dataset = load_imdb_dataset()
    test_prompts = [text[:config['evaluation']['max_test_length']] for text in dataset['test']['text'][:config['evaluation']['num_test_samples']]]
    
    # Генерация завершений
    with torch.no_grad():
        inputs = tokenizer(test_prompts, return_tensors="pt", padding=True, truncation=True)
        warp_outputs = warp_model.generate(**inputs)
        sft_outputs = sft_model.generate(**inputs)
    
    # Вычисление наград
    warp_rewards = reward_model(warp_outputs).logits.mean()
    sft_rewards = reward_model(sft_outputs).logits.mean()
    
    # Вычисление KL-дивергенции
    warp_logits = warp_model(warp_outputs).logits
    sft_logits = sft_model(sft_outputs).logits
    kl_div = F.kl_div(F.log_softmax(warp_logits, dim=-1), F.softmax(sft_logits, dim=-1), reduction='batchmean')
    
    return {
        "warp_reward": warp_rewards.item(),
        "sft_reward": sft_rewards.item(),
        "kl_divergence": kl_div.item()
    }

def generate_samples(model, tokenizer, prompts, max_length=50):
    """Генерация примеров текста с использованием модели"""
    generated_texts = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append(generated_text)
    return generated_texts

if __name__ == "__main__":
    results = evaluate_models()
    print(f"Результаты оценки:")
    print(f"Средняя награда WARP: {results['warp_reward']}")
    print(f"Средняя награда SFT: {results['sft_reward']}")
    print(f"KL-дивергенция: {results['kl_divergence']}")
    
    # Генерация примеров
    config = load_config()
    warp_model = GPT2LMHeadModel.from_pretrained(config['warp']['output_dir'])
    tokenizer = GPT2Tokenizer.from_pretrained(config['warp']['model_name'])
    
    test_prompts = [
        "This movie was",
        "I really enjoyed",
        "The acting was"
    ]
    
    generated_texts = generate_samples(warp_model, tokenizer, test_prompts)
    
    print("\nПримеры сгенерированных текстов:")
    for prompt, text in zip(test_prompts, generated_texts):
        print(f"Промпт: {prompt}")
        print(f"Сгенерированный текст: {text}\n")