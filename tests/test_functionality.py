import pytest
from src.FakeClass import DetectFake
import pandas as pd
from pathlib import Path
parent_path = Path().parent.resolve()

@pytest.fixture
def example_data_for_PAM():
    detect_fake = DetectFake('passive_aggressive', data=pd.read_csv(parent_path / 'tests/testdata.csv', delimiter=';'))
    detect_fake.prepare_data()
    return detect_fake.data['text'][0]


@pytest.fixture
def example_data_for_NT():
    data = pd.read_csv(parent_path / 'tests/testdata.csv', delimiter=';')
    data = data.drop(columns=['id', 'title', 'author'], axis=1)
    data = data.dropna(axis=0)
    data['clean_news'] = data['text'].str.lower()
    data['clean_news'] = data['clean_news'].str.replace(r'[^A-Za-z0-9\s]', '')
    data['clean_news'] = data['clean_news'].str.replace(r'\n', '')
    data['clean_news'] = data['clean_news'].str.replace(r'\s+', ' ')
    return data


def test_prepare_data(example_data_for_PAM):
    data = pd.read_csv(parent_path / 'tests/testdata.csv', delimiter=';')
    assert data['text'][0] == example_data_for_PAM


def test_cleaning_news_for_neural_net(example_data_for_NT):
    detect_fake = DetectFake('neural_network', data=pd.read_csv(parent_path / 'tests/testdata.csv', delimiter=';'))
    detect_fake.cleaning_news_for_neural_net()
    assert detect_fake.data['clean_news'][0] == example_data_for_NT['clean_news'][0]

