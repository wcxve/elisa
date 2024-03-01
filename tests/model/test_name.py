from elisa.model.add import PowerLaw


def test_model_name():
    model = PowerLaw() + PowerLaw()
    assert model.name == 'powerlaw + powerlaw_2'
