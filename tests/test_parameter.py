import pytest

from elisa.models.parameter import UniformParameter


def test_param_name():
    a = UniformParameter('a', 1.5, 0.0, 2.0)
    a2 = UniformParameter('a', 0.5, 0.0, 2.0)
    b = a + a2
    assert str(b) == "a + a'"
    assert b.default == 2.0


def test_param_bound():
    # test default value can be equal to the lower bound if fixed
    p = UniformParameter(name='p', default=0.0, min=0.0, max=1.0, fixed=True)
    assert p.default == p.min
    # test default value is automatically adjusted if fixed is set to False
    p.fixed = False
    assert p.min < p.default < p.max

    # test default value can be equal to the upper bound if fixed
    p = UniformParameter(name='p', default=0.0, min=-1.0, max=0.0, fixed=True)
    assert p.default == p.max
    # test default value is automatically adjusted if fixed is set to False
    p.fixed = False
    assert p.min < p.default < p.max

    # test default value is automatically adjusted during initialization
    p = UniformParameter(name='p', default=0.0, min=0.0, max=1.0)
    assert p.min < p.default < p.max

    # test default value is automatically adjusted during initialization
    p = UniformParameter(name='p', default=0.0, min=-1.0, max=0.0)
    assert p.min < p.default < p.max

    # if the min is greater than the max, an error is raised
    with pytest.raises(ValueError):
        UniformParameter(name='p', default=0.0, min=1.0, max=0.0)

    # if the min is equal to the max, an error is raised
    with pytest.raises(ValueError):
        UniformParameter(name='p', default=0.0, min=0.0, max=0.0)

    # if the default value is not in the range, an error is raised
    with pytest.raises(ValueError):
        UniformParameter(name='p', default=-1.0, min=0.0, max=1.0)
    with pytest.raises(ValueError):
        UniformParameter(name='p', default=1.0, min=-1.0, max=0.0)

    # if the lower bound is non-positive when log is True, an error is raised
    with pytest.raises(ValueError):
        UniformParameter(name='p', default=0.5, min=0.0, max=1.0, log=True)
    with pytest.raises(ValueError):
        UniformParameter(name='p', default=0.5, min=-1.0, max=1.0, log=True)
