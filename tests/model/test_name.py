from elisa.model.add import Powerlaw


def test_model_name():
    model = Powerlaw() + Powerlaw()
    assert repr(model) == 'powerlaw + powerlaw2'
