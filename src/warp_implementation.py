import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForSequenceClassification
from data_preparation import load_config, load_imdb_dataset

def warp_update(theta_init, theta_m, theta_ema, r_beta, eta, mu):
    """Логика обновления WARP"""
    theta = theta_init + eta * (theta_m - theta_init)
    theta_ema = mu * theta_ema + (1 - mu) * theta
    return theta, theta_ema

def slerp(theta_init, theta_m_list, lam):
    """Сферическая линейная интерполяция"""
    def slerp_two(p0, p1, t):
        omega = torch.arccos((p0 * p1).sum() / (p0.norm() * p1.norm()))
        so = torch.sin(omega)
        return torch.sin((1.0 - t) * omega) / so * p0 + torch.sin(t * omega) / so * p1
    
    result = theta_init
    for theta_m in theta_m_list:
        result = slerp_two(result, theta_m, lam)
    
    return result

def train_warp():
    """Обучение с использованием WARP"""
    config = load_config()
    
    # Загрузка моделей
    gpt2_model = GPT2LMHeadModel.from_pretrained(config['warp']['model_name'])
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(config['warp']['model_name'])
    reward_model = AutoModelForSequenceClassification.from_pretrained(config['reward_model']['output_dir'])
    
    # Подготовка данных
    dataset = load_imdb_dataset()
    prompts = [text[:config['data']['max_length']] for text in dataset['train']['text']]
    
    # Параметры WARP
    I, M, T = config['warp']['I'], config['warp']['M'], config['warp']['T']
    mu, lam, eta = config['warp']['mu'], config['warp']['lam'], config['warp']['eta']
    
    theta_init = gpt2_model.state_dict()
    
    for i in range(I):
        for m in range(M):
            theta_m = theta_init.copy()
            theta_ema = theta_init.copy()
            
            for t in range(T):
                # Генерация завершения
                inputs = gpt2_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                outputs = gpt2_model.generate(**inputs)
                
                # Вычисление награды
                rewards = reward_model(outputs).logits
                
                # Обновление параметров
                theta_m, theta_ema = warp_update(theta_init, theta_m, theta_ema, rewards, eta, mu)
            
        # Объединение весов
        theta_slerp = slerp(theta_init, [theta_m for _ in range(M)], lam=1/M)
        theta_init = (1 - eta) * theta_init + eta * theta_slerp
    
    gpt2_model.load_state_dict(theta_init)
    gpt2_model.save_pretrained(config['warp']['output_dir'])

if __name__ == "__main__":
    train_warp()
    print("WARP обучение завершено")