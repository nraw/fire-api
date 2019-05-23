from model import Model
import os


def test_db_string():
    assert os.environ['DB_URI'] != ''


def test_model_init():
    model = Model()
    assert model.model is not None
