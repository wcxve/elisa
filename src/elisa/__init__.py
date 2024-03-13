from .__about__ import __version__ as __version__
from .data import Data as Data, Response as Response, Spectrum as Spectrum
from .infer import BayesianFit as BayesianFit, MaxLikeFit as MaxLikeFit
from .models.model import (
    AnaIntAdditive as AnaIntAdditive,
    AnaIntMultiplicative as AnaIntMultiplicative,
    ConvolutionComponent as ConvolutionComponent,
    NumIntAdditive as NumIntAdditive,
    NumIntMultiplicative as NumIntMultiplicative,
)
from .models.parameter import (
    CompositeParameter as CompositeParameter,
    ConstantInterval as ConstantInterval,
    ConstantValue as ConstantValue,
    DistParameter as DistParameter,
    UniformParameter as UniformParameter,
)
from .util import jax_enable_x64, set_cpu_cores

jax_enable_x64(True)
set_cpu_cores(4)
