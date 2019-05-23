from model import Model
import fire

model = Model()


def predict(p1, p2, p3, p4):
    """The function to predict EVERYTHING

    :param p1: Name of Player 1 
    :param p2: Name of Player 2
    :param p3: Name of Player 3
    :param p4: Name of Player 4
    """
    return model.predict(p1, p2, p3, p4)


def retrain():
    """If you ever felt the model just ain't good enough.."""
    model.retrain()
    return 'retrained'


if __name__ == '__main__':
    fire.Fire()
