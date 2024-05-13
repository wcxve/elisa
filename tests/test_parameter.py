from elisa.models.parameter import UniformParameter


def test_param_name():
    a = UniformParameter('a', 1.5, 0.0, 2.0)
    a2 = UniformParameter('a', 0.5, 0.0, 2.0)
    b = a + a2
    assert str(b) == "a + a'"
    assert b.default == 2.0
