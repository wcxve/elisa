from elisa.model.base import UniformParameter


def test_param_name():
    a = UniformParameter("a", "a", 1.0, 0.0, 2.0)
    a2 = UniformParameter("a2", "a2", 1.0, 0.0, 2.0)
    b = a + a2
    assert repr(b) == "a + a2"
    assert b.default == 2.0
