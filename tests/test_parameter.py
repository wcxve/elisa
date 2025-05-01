from elisa.models.parameter import UniformParameter


def test_param_name():
    a = UniformParameter('a', 1.5, 0.0, 2.0)
    a2 = UniformParameter('a', 0.5, 0.0, 2.0)
    b = a + a2
    assert str(b) == "a + a'"
    assert b.default == 2.0


def test_param_bound():
    p = UniformParameter('p', 0.0, 0.0, 1.0, fixed=True)
    assert p.default == p.min
    p.fixed = False
    assert p.min < p.default < p.max

    p = UniformParameter('p', 0.0, -1.0, 0.0, fixed=True)
    assert p.default == p.max
    p.fixed = False
    assert p.min < p.default < p.max

    p = UniformParameter('p', 0.0, 0.0, 1.0)
    assert p.min < p.default < p.max

    p = UniformParameter('p', 0.0, -1.0, 0.0)
    assert p.min < p.default < p.max
