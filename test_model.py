import os
import pytest


@pytest.fixture
def model():
    from model import Model
    model = Model()
    return model


def test_db_string():
    assert os.environ['DB_URI'] != ''


def test_model_init(model):
    assert model.model is not None


def test_prediction(model):
    prediction = model.predict('Petra', 'Gayane', 'Igor', 'Felipe')
    assert type(prediction) == type([0])
    assert len(prediction) == 1
    assert prediction[0] <= 1
    assert prediction[0] >= 0
