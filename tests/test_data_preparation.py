import pytest
from src.data_preparation import load_imdb_dataset, create_pairs, prepare_data

def test_load_imdb_dataset():
    dataset = load_imdb_dataset()
    assert 'train' in dataset
    assert 'test' in dataset
    assert len(dataset['train']) == 25000
    assert len(dataset['test']) == 25000

def test_create_pairs():
    mock_dataset = {
        'text': ['Positive review 1', 'Negative review 1', 'Positive review 2', 'Negative review 2'],
        'label': [1, 0, 1, 0]
    }
    pairs = create_pairs(mock_dataset)
    assert len(pairs) == 2
    assert pairs[0] == ('Positive review 1', 'Negative review 1')
    assert pairs[1] == ('Positive review 2', 'Negative review 2')

def test_prepare_data():
    data = prepare_data()
    assert len(data) > 0
    assert isinstance(data[0], tuple)
    assert len(data[0]) == 2
    assert isinstance(data[0][0], str)
    assert isinstance(data[0][1], str)