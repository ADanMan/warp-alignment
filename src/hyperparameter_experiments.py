import matplotlib.pyplot as plt
from data_preparation import load_config
from warp_implementation import train_warp
from training_evaluation import evaluate_models

def run_hyperparameter_experiments():
    """Проведение экспериментов с гиперпараметрами"""
    config = load_config()
    
    param_name = config['hyperparameter_experiment']['parameter']
    param_values = config['hyperparameter_experiment']['values']
    
    results = []
    
    for value in param_values:
        # Обновление значения гиперпараметра
        config['warp'][param_name] = value
        
        # Обучение WARP с новым значением гиперпараметра
        train_warp()
        
        # Оценка результатов
        eval_results = evaluate_models()
        
        results.append({
            'param_value': value,
            'reward': eval_results['warp_reward'],
            'kl_div': eval_results['kl_divergence']
        })
    
    # Построение графика результатов
    plt.figure(figsize=(10, 6))
    plt.plot([r['param_value'] for r in results], [r['reward'] for r in results], 'bo-', label='Награда')
    plt.plot([r['param_value'] for r in results], [r['kl_div'] for r in results], 'ro-', label='KL-дивергенция')
    plt.xlabel(param_name)
    plt.ylabel('Значение метрики')
    plt.legend()
    plt.title(f'Влияние {param_name} на награду и KL-дивергенцию')
    plt.savefig('results/figures/hyperparameter_experiment.png')
    plt.close()
    
    return results

if __name__ == "__main__":
    results = run_hyperparameter_experiments()
    print("Эксперименты с гиперпараметрами завершены")
    print("Результаты сохранены в файле results/figures/hyperparameter_experiment.png")