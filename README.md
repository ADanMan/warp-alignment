# WARP Alignment

Этот проект представляет собой реализацию и анализ метода WARP (Weight Averaged Rewarded Policies) для алаймента языковых моделей на датасете IMDB.

## Структура проекта

```
warp-alignment/
│
├── data/
│   └── README.md
│
├── src/
│   ├── data_preparation.py
│   ├── reward_model.py
│   ├── warp_implementation.py
│   ├── training_evaluation.py
│   └── hyperparameter_experiments.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_reward_model_training.ipynb
│   ├── 03_warp_training.ipynb
│   └── 04_results_analysis.ipynb
│
├── configs/
│   └── config.yaml
│
├── results/
│   ├── figures/
│   └── models/
│
├── tests/
│   ├── test_data_preparation.py
│   ├── test_reward_model.py
│   └── test_warp_implementation.py
│
├── requirements.txt
├── setup.py
├── README.md
└── report.md
```

## Установка

1. Клонируйте репозиторий:
   ```
   git clone https://github.com/ADanMan/warp-alignment.git
   cd warp-alignment
   ```

2. Создайте виртуальное окружение и активируйте его:
   ```
   python -m venv venv
   source venv/bin/activate  # На Windows используйте venv\Scripts\activate
   ```

3. Установите зависимости:
   ```
   pip install -e .
   ```

## Использование

1. Исследование данных:
   ```
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```

2. Обучение модели наград:
   ```
   python src/reward_model.py
   ```

3. Обучение с использованием WARP:
   ```
   python src/warp_implementation.py
   ```

4. Проведение экспериментов с гиперпараметрами:
   ```
   python src/hyperparameter_experiments.py
   ```

5. Анализ результатов:
   ```
   jupyter notebook notebooks/04_results_analysis.ipynb
   ```

## Тестирование

Для запуска тестов используйте:

```
pytest tests/
```

## Отчет

Подробный отчет о проведенных экспериментах и анализ результатов доступен в файле [report.md](report.md).

## Лицензия

Этот проект распространяется под лицензией MIT. Подробности см. в файле [LICENSE](LICENSE).