from .__about__ import __version__ as __version__
from .data import (
    Data as Data,
    ObservationData as ObservationData,
    Response as Response,
    ResponseData as ResponseData,
    Spectrum as Spectrum,
    SpectrumData as SpectrumData,
)
from .infer import BayesFit as BayesFit, MaxLikeFit as MaxLikeFit
from .models.model import (
    AnaIntAdditive as AnaIntAdditive,
    AnaIntMultiplicative as AnaIntMultiplicative,
    ConvolutionComponent as ConvolutionComponent,
    NumIntAdditive as NumIntAdditive,
    NumIntMultiplicative as NumIntMultiplicative,
    ParamConfig as ParamConfig,
    PyAnaInt as PyAnaInt,
    PyNumInt as PyNumInt,
)
from .models.parameter import (
    CompositeParameter as CompositeParameter,
    ConstantInterval as ConstantInterval,
    ConstantValue as ConstantValue,
    DistParameter as DistParameter,
    UniformParameter as UniformParameter,
)
from .util import (
    jax_debug_nans as jax_debug_nans,
    jax_enable_x64,
    set_cpu_cores,
    set_jax_platform,
)

jax_enable_x64(True)
set_jax_platform('cpu')
set_cpu_cores(4)
