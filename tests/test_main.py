import sys
import pytest
# sys.path.append('C:/Users/User/PycharmProjects/pythonProject/DetectFakeNews/src')
from src.FakeClass import DetectFake

def test_module_name_check_negative():
    expected_error = ''
    expected_message = 'Model name not found'
    try:
        detect_fake = DetectFake('asd')
    except Exception as ex:
        expected_error = str(ex)
    assert expected_message in expected_error, 'error not raised, or other type error'

@pytest.mark.parametrize(
    'expected_name, model_name',
        [
            ('passive_aggressive', 'passive_aggressive'),
            ('neural_network', 'neural_network')
        ]
)
def test_module_name_check_positive(expected_name, model_name):
    actual_result = ''
    try:
        detect_fake_true = DetectFake(model_name)
        actual_result = detect_fake_true.model_name
    except Exception as ex:
        actual_result = str(ex)
    assert expected_name == actual_result, 'expected message not found'




