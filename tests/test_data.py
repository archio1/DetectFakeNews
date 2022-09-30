import pandas as pd
from src.FakeClass import DetectFake
from pathlib import Path


parent_path = Path().parent.resolve()

def test_checkdata_for_real():
    data_frame_true = pd.read_csv(parent_path / 'True.csv')
    test_data = 'Real'
    detect_fake = DetectFake('passive_aggressive', model_path= parent_path / 'resources/passive_aggressive_model')
    assert test_data == detect_fake.predict(data_frame_true['text'][1])


def test_checkdata_for_false():
    data_frame_fake = pd.read_csv(parent_path / 'src/Fake.csv')
    test_data = 'Fake'
    detect_fake = DetectFake('passive_aggressive', model_path= parent_path / 'resources/passive_aggressive_model')
    assert test_data == detect_fake.predict(data_frame_fake['text'][3])


def test_checkdata_for_NT_on_fake():
    data_frame_fake = pd.read_csv(parent_path / 'src/Fake.csv')
    detect_fake = DetectFake('neural_network', data=data_frame_fake.iloc[54], model_path=parent_path / 'resources/neural_model')
    assert 0.1 <= detect_fake.predict_by_neural_network() <= 0.6


def test_checkdata_for_NT_on_true():
    data_frame_fake = pd.read_csv('C:/Users/User/PycharmProjects/pythonProject/DetectFakeNews/True.csv')
    detect_fake = DetectFake('neural_network', data=data_frame_fake.iloc[3], model_path=parent_path / 'resources/neural_model')
    assert 0.5 <= detect_fake.predict_by_neural_network() <= 1

