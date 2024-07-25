from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer
from data_preparation import load_config, prepare_data

def train_reward_model():
    """Обучение модели наград"""
    config = load_config()
    train_pairs = prepare_data()

    tokenizer = AutoTokenizer.from_pretrained(config['reward_model']['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(config['reward_model']['model_name'])

    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_pairs,
        args=RewardTrainer.RewardConfig(
            num_train_epochs=config['reward_model']['num_train_epochs'],
            output_dir=config['reward_model']['output_dir']
        )
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    train_reward_model()
    print("Модель наград обучена и сохранена")