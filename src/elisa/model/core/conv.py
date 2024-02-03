"""ConvolutionComponent models."""
from __future__ import annotations

from typing import Literal

from elisa.model.core.model import (
    ComponentBase,
    ComponentMeta,
    ModelBase,
    ParamConfig,
)
from elisa.util.typing import Array, JAXArray, JAXFloat


class ConvolutionModel(ModelBase):
    def _eval(
        self, egrid: Array, params: dict[str, dict[str, JAXFloat]]
    ) -> JAXArray:
        pass

    @property
    def name(self) -> str:
        pass

    @property
    def latex(self) -> str:
        pass

    @property
    def type(self) -> Literal['add', 'mul']:
        pass


class ConvolutionMeta(ComponentMeta):
    def __call__(cls, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class ConvolutionComponent(ComponentBase, metaclass=ConvolutionMeta):
    """Prototype class to define convolution component."""

    _model: ModelBase | None = None

    @property
    def type(self) -> Literal['add', 'mul', 'conv']:
        if self._model is None:
            return 'conv'
        else:
            return self._model.type

    def __call__(self, model: ModelBase) -> ConvolutionModel:
        return ConvolutionModel(self)


class RedShift(ConvolutionComponent):
    _config = (ParamConfig('z', 'z', '', 0.0, -0.999, 10.0),)

    def _eval(self, egrid, params) -> JAXArray:
        pass


class VelocityShift(ConvolutionComponent):
    pass


class PhFlux(ConvolutionComponent):
    pass


class EnFlux(ConvolutionComponent):
    pass


# pl = Powerlaw()
# redshift = RedShift()
# m = tbabs * redshift(tbabs * pl)
