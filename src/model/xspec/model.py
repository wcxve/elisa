from ..base import SpectralComponent, SpectralModel
from .base import XspecNumericGradOp

__all__ = ['SSS_ice', 'TBabs', 'TBfeo', 'TBgas', 'TBgrain', 'TBpcf', 'TBrel', 'TBvarabs', 'absori', 'acisabs', 'agauss', 'agnsed', 'agnslim', 'apec', 'bapec', 'bbody', 'bbodyrad', 'bexrav', 'bexriv', 'bkn2pow', 'bknpower', 'bmc', 'bremss', 'brnei', 'btapec', 'bvapec', 'bvrnei', 'bvtapec', 'bvvapec', 'bvvrnei', 'bvvtapec', 'bwcycl', 'c6mekl', 'c6pmekl', 'c6pvmkl', 'c6vmekl', 'cabs', 'carbatm', 'cemekl', 'cevmkl', 'cflow', 'cflux', 'cglumin', 'clumin', 'compLS', 'compPS', 'compST', 'compTT', 'compbb', 'compmag', 'comptb', 'compth', 'constant', 'cpflux', 'cph', 'cplinear', 'cutoffpl', 'cyclabs', 'disk', 'diskbb', 'diskir', 'diskline', 'diskm', 'disko', 'diskpbb', 'diskpn', 'dust', 'edge', 'eplogpar', 'eqpair', 'eqtherm', 'equil', 'expabs', 'expdec', 'expfac', 'ezdiskbb', 'gabs', 'gadem', 'gaussian', 'gnei', 'grad', 'grbcomp', 'grbjet', 'grbm', 'gsmooth', 'hatm', 'heilin', 'highecut', 'hrefl', 'ireflect', 'ismabs', 'ismdust', 'jet', 'kdblur', 'kdblur2', 'kerrbb', 'kerrconv', 'kerrd', 'kerrdisk', 'kyconv', 'kyrline', 'laor', 'laor2', 'log10con', 'logconst', 'logpar', 'lorentz', 'lsmooth', 'lyman', 'meka', 'mekal', 'mkcflow', 'nei', 'nlapec', 'notch', 'npshock', 'nsa', 'nsagrav', 'nsatmos', 'nsmax', 'nsmaxg', 'nsx', 'nteea', 'nthComp', 'olivineabs', 'optxagn', 'optxagnf', 'partcov', 'pcfabs', 'pegpwrlw', 'pexmon', 'pexrav', 'pexriv', 'phabs', 'plabs', 'plcabs', 'polconst', 'pollin', 'polpow', 'posm', 'powerlaw', 'pshock', 'pwab', 'qsosed', 'raymond', 'rdblur', 'redden', 'redge', 'reflect', 'refsch', 'rfxconv', 'rgsxsrc', 'rnei', 'sedov', 'simpl', 'sirf', 'slimbh', 'smaug', 'smedge', 'snapec', 'spexpcut', 'spline', 'srcut', 'sresc', 'ssa', 'step', 'swind1', 'tapec', 'thcomp', 'uvred', 'vapec', 'varabs', 'vashift', 'vbremss', 'vcph', 'vequil', 'vgadem', 'vgnei', 'vmcflow', 'vmeka', 'vmekal', 'vmshift', 'vnei', 'vnpshock', 'voigt', 'vphabs', 'vpshock', 'vraymond', 'vrnei', 'vsedov', 'vtapec', 'vvapec', 'vvgnei', 'vvnei', 'vvnpshock', 'vvpshock', 'vvrnei', 'vvsedov', 'vvtapec', 'vvwdem', 'vwdem', 'wabs', 'wdem', 'wndabs', 'xilconv', 'xion', 'xscat', 'zTBabs', 'zagauss', 'zashift', 'zbabs', 'zbbody', 'zbknpower', 'zbremss', 'zcutoffpl', 'zdust', 'zedge', 'zgauss', 'zhighect', 'zigm', 'zkerrbb', 'zlogpar', 'zmshift', 'zpcfabs', 'zphabs', 'zpowerlw', 'zredden', 'zsmdust', 'zvarabs', 'zvfeabs', 'zvphabs', 'zwabs', 'zwndabs', 'zxipab', 'zxipcf']


class SSS_iceOp(XspecNumericGradOp):
    modname = 'SSS_ice'
    optype = 'mul'

class SSS_iceComponent(SpectralComponent):
    _comp_name = 'SSS_ice'
    _config = {'clumps': (0.0, 0.0, 10.0, False, False)}
    _op_class = SSS_iceOp
    def __init__(self, clumps, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class SSS_ice(SpectralModel):
    def __init__(self, clumps=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([SSS_iceComponent(clumps, name, grad_method, eps)])


class TBabsOp(XspecNumericGradOp):
    modname = 'TBabs'
    optype = 'mul'

class TBabsComponent(SpectralComponent):
    _comp_name = 'TBabs'
    _config = {'nH': (1.0, 0.0, 1000000.0, False, False)}
    _op_class = TBabsOp
    def __init__(self, nH, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class TBabs(SpectralModel):
    def __init__(self, nH=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([TBabsComponent(nH, name, grad_method, eps)])


class TBfeoOp(XspecNumericGradOp):
    modname = 'TBfeo'
    optype = 'mul'

class TBfeoComponent(SpectralComponent):
    _comp_name = 'TBfeo'
    _config = {'nH': (1.0, 0.0, 1000000.0, False, False), 'O': (1.0, -1e+38, 1e+38, True, False), 'Fe': (1.0, -1e+38, 1e+38, True, False), 'redshift': (0.0, -1.0, 10.0, True, False)}
    _op_class = TBfeoOp
    def __init__(self, nH, O, Fe, redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class TBfeo(SpectralModel):
    def __init__(self, nH=None, O=None, Fe=None, redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([TBfeoComponent(nH, O, Fe, redshift, name, grad_method, eps)])


class TBgasOp(XspecNumericGradOp):
    modname = 'TBgas'
    optype = 'mul'

class TBgasComponent(SpectralComponent):
    _comp_name = 'TBgas'
    _config = {'nH': (1.0, 0.0, 1000000.0, False, False), 'redshift': (0.0, -1.0, 10.0, True, False)}
    _op_class = TBgasOp
    def __init__(self, nH, redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class TBgas(SpectralModel):
    def __init__(self, nH=None, redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([TBgasComponent(nH, redshift, name, grad_method, eps)])


class TBgrainOp(XspecNumericGradOp):
    modname = 'TBgrain'
    optype = 'mul'

class TBgrainComponent(SpectralComponent):
    _comp_name = 'TBgrain'
    _config = {'nH': (1.0, 0.0, 1000000.0, False, False), 'h2': (0.2, 0.0, 1.0, True, False), 'rho': (1.0, 0.0, 5.0, True, False), 'amin': (0.025, 0.0, 0.25, True, False), 'amax': (0.25, 0.0, 1.0, True, False), 'PL': (3.5, 0.0, 5.0, True, False)}
    _op_class = TBgrainOp
    def __init__(self, nH, h2, rho, amin, amax, PL, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class TBgrain(SpectralModel):
    def __init__(self, nH=None, h2=None, rho=None, amin=None, amax=None, PL=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([TBgrainComponent(nH, h2, rho, amin, amax, PL, name, grad_method, eps)])


class TBpcfOp(XspecNumericGradOp):
    modname = 'TBpcf'
    optype = 'mul'

class TBpcfComponent(SpectralComponent):
    _comp_name = 'TBpcf'
    _config = {'nH': (1.0, 0.0, 1000000.0, False, False), 'pcf': (0.5, 0.0, 1.0, False, False), 'redshift': (0.0, -1.0, 10.0, True, False)}
    _op_class = TBpcfOp
    def __init__(self, nH, pcf, redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class TBpcf(SpectralModel):
    def __init__(self, nH=None, pcf=None, redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([TBpcfComponent(nH, pcf, redshift, name, grad_method, eps)])


class TBrelOp(XspecNumericGradOp):
    modname = 'TBrel'
    optype = 'mul'

class TBrelComponent(SpectralComponent):
    _comp_name = 'TBrel'
    _config = {'nH': (0.0, -1000000.0, 1000000.0, False, False), 'He': (1.0, 0.0, 1e+38, True, False), 'C': (1.0, 0.0, 1e+38, True, False), 'N': (1.0, 0.0, 1e+38, True, False), 'O': (1.0, 0.0, 1e+38, True, False), 'Ne': (1.0, 0.0, 1e+38, True, False), 'Na': (1.0, 0.0, 1e+38, True, False), 'Mg': (1.0, 0.0, 1e+38, True, False), 'Al': (1.0, 0.0, 1e+38, True, False), 'Si': (1.0, 0.0, 1e+38, True, False), 'S': (1.0, 0.0, 1e+38, True, False), 'Cl': (1.0, 0.0, 1e+38, True, False), 'Ar': (1.0, 0.0, 1e+38, True, False), 'Ca': (1.0, 0.0, 1e+38, True, False), 'Cr': (1.0, 0.0, 1e+38, True, False), 'Fe': (1.0, 0.0, 1e+38, True, False), 'Co': (1.0, 0.0, 1e+38, True, False), 'Ni': (1.0, 0.0, 1e+38, True, False), 'H2': (0.2, 0.0, 1.0, True, False), 'rho': (1.0, 0.0, 5.0, True, False), 'amin': (0.025, 0.0, 0.25, True, False), 'amax': (0.25, 0.0, 1.0, True, False), 'PL': (3.5, 0.0, 5.0, True, False), 'H_dep': (1.0, 0.0, 1.0, True, False), 'He_dep': (1.0, 0.0, 1.0, True, False), 'C_dep': (0.5, 0.0, 1.0, True, False), 'N_dep': (1.0, 0.0, 1.0, True, False), 'O_dep': (0.6, 0.0, 1.0, True, False), 'Ne_dep': (1.0, 0.0, 1.0, True, False), 'Na_dep': (0.25, 0.0, 1.0, True, False), 'Mg_dep': (0.2, 0.0, 1.0, True, False), 'Al_dep': (0.02, 0.0, 1.0, True, False), 'Si_dep': (0.1, 0.0, 1.0, True, False), 'S_dep': (0.6, 0.0, 1.0, True, False), 'Cl_dep': (0.5, 0.0, 1.0, True, False), 'Ar_dep': (1.0, 0.0, 1.0, True, False), 'Ca_dep': (0.003, 0.0, 1.0, True, False), 'Cr_dep': (0.03, 0.0, 1.0, True, False), 'Fe_dep': (0.3, 0.0, 1.0, True, False), 'Co_dep': (0.05, 0.0, 1.0, True, False), 'Ni_dep': (0.04, 0.0, 1.0, True, False), 'redshift': (0.0, -1.0, 10.0, True, False)}
    _op_class = TBrelOp
    def __init__(self, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Cl, Ar, Ca, Cr, Fe, Co, Ni, H2, rho, amin, amax, PL, H_dep, He_dep, C_dep, N_dep, O_dep, Ne_dep, Na_dep, Mg_dep, Al_dep, Si_dep, S_dep, Cl_dep, Ar_dep, Ca_dep, Cr_dep, Fe_dep, Co_dep, Ni_dep, redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class TBrel(SpectralModel):
    def __init__(self, nH=None, He=None, C=None, N=None, O=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, S=None, Cl=None, Ar=None, Ca=None, Cr=None, Fe=None, Co=None, Ni=None, H2=None, rho=None, amin=None, amax=None, PL=None, H_dep=None, He_dep=None, C_dep=None, N_dep=None, O_dep=None, Ne_dep=None, Na_dep=None, Mg_dep=None, Al_dep=None, Si_dep=None, S_dep=None, Cl_dep=None, Ar_dep=None, Ca_dep=None, Cr_dep=None, Fe_dep=None, Co_dep=None, Ni_dep=None, redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([TBrelComponent(nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Cl, Ar, Ca, Cr, Fe, Co, Ni, H2, rho, amin, amax, PL, H_dep, He_dep, C_dep, N_dep, O_dep, Ne_dep, Na_dep, Mg_dep, Al_dep, Si_dep, S_dep, Cl_dep, Ar_dep, Ca_dep, Cr_dep, Fe_dep, Co_dep, Ni_dep, redshift, name, grad_method, eps)])


class TBvarabsOp(XspecNumericGradOp):
    modname = 'TBvarabs'
    optype = 'mul'

class TBvarabsComponent(SpectralComponent):
    _comp_name = 'TBvarabs'
    _config = {'nH': (1.0, 0.0, 1000000.0, False, False), 'He': (1.0, 0.0, 1.0, True, False), 'C': (1.0, 0.0, 1.0, True, False), 'N': (1.0, 0.0, 1.0, True, False), 'O': (1.0, 0.0, 1.0, True, False), 'Ne': (1.0, 0.0, 1.0, True, False), 'Na': (1.0, 0.0, 1.0, True, False), 'Mg': (1.0, 0.0, 1.0, True, False), 'Al': (1.0, 0.0, 1.0, True, False), 'Si': (1.0, 0.0, 1.0, True, False), 'S': (1.0, 0.0, 1.0, True, False), 'Cl': (1.0, 0.0, 1.0, True, False), 'Ar': (1.0, 0.0, 1.0, True, False), 'Ca': (1.0, 0.0, 1.0, True, False), 'Cr': (1.0, 0.0, 1.0, True, False), 'Fe': (1.0, 0.0, 1.0, True, False), 'Co': (1.0, 0.0, 1.0, True, False), 'Ni': (1.0, 0.0, 1.0, True, False), 'H2': (0.2, 0.0, 1.0, True, False), 'rho': (1.0, 0.0, 5.0, True, False), 'amin': (0.025, 0.0, 0.25, True, False), 'amax': (0.25, 0.0, 1.0, True, False), 'PL': (3.5, 0.0, 5.0, True, False), 'H_dep': (1.0, 0.0, 1.0, True, False), 'He_dep': (1.0, 0.0, 1.0, True, False), 'C_dep': (1.0, 0.0, 1.0, True, False), 'N_dep': (1.0, 0.0, 1.0, True, False), 'O_dep': (1.0, 0.0, 1.0, True, False), 'Ne_dep': (1.0, 0.0, 1.0, True, False), 'Na_dep': (1.0, 0.0, 1.0, True, False), 'Mg_dep': (1.0, 0.0, 1.0, True, False), 'Al_dep': (1.0, 0.0, 1.0, True, False), 'Si_dep': (1.0, 0.0, 1.0, True, False), 'S_dep': (1.0, 0.0, 1.0, True, False), 'Cl_dep': (1.0, 0.0, 1.0, True, False), 'Ar_dep': (1.0, 0.0, 1.0, True, False), 'Ca_dep': (1.0, 0.0, 1.0, True, False), 'Cr_dep': (1.0, 0.0, 1.0, True, False), 'Fe_dep': (1.0, 0.0, 1.0, True, False), 'Co_dep': (1.0, 0.0, 1.0, True, False), 'Ni_dep': (1.0, 0.0, 1.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False)}
    _op_class = TBvarabsOp
    def __init__(self, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Cl, Ar, Ca, Cr, Fe, Co, Ni, H2, rho, amin, amax, PL, H_dep, He_dep, C_dep, N_dep, O_dep, Ne_dep, Na_dep, Mg_dep, Al_dep, Si_dep, S_dep, Cl_dep, Ar_dep, Ca_dep, Cr_dep, Fe_dep, Co_dep, Ni_dep, Redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class TBvarabs(SpectralModel):
    def __init__(self, nH=None, He=None, C=None, N=None, O=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, S=None, Cl=None, Ar=None, Ca=None, Cr=None, Fe=None, Co=None, Ni=None, H2=None, rho=None, amin=None, amax=None, PL=None, H_dep=None, He_dep=None, C_dep=None, N_dep=None, O_dep=None, Ne_dep=None, Na_dep=None, Mg_dep=None, Al_dep=None, Si_dep=None, S_dep=None, Cl_dep=None, Ar_dep=None, Ca_dep=None, Cr_dep=None, Fe_dep=None, Co_dep=None, Ni_dep=None, Redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([TBvarabsComponent(nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Cl, Ar, Ca, Cr, Fe, Co, Ni, H2, rho, amin, amax, PL, H_dep, He_dep, C_dep, N_dep, O_dep, Ne_dep, Na_dep, Mg_dep, Al_dep, Si_dep, S_dep, Cl_dep, Ar_dep, Ca_dep, Cr_dep, Fe_dep, Co_dep, Ni_dep, Redshift, name, grad_method, eps)])


class absoriOp(XspecNumericGradOp):
    modname = 'absori'
    optype = 'mul'

class absoriComponent(SpectralComponent):
    _comp_name = 'absori'
    _config = {'PhoIndex': (2.0, 0.0, 4.0, True, False), 'nH': (1.0, 0.0, 100.0, False, False), 'Temp_abs': (30000.0, 10000.0, 1000000.0, True, False), 'xi': (1.0, 0.0, 5000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'Fe_abund': (1.0, 0.0, 1000000.0, True, False)}
    _op_class = absoriOp
    def __init__(self, PhoIndex, nH, Temp_abs, xi, Redshift, Fe_abund, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class absori(SpectralModel):
    def __init__(self, PhoIndex=None, nH=None, Temp_abs=None, xi=None, Redshift=None, Fe_abund=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([absoriComponent(PhoIndex, nH, Temp_abs, xi, Redshift, Fe_abund, name, grad_method, eps)])


class acisabsOp(XspecNumericGradOp):
    modname = 'acisabs'
    optype = 'mul'

class acisabsComponent(SpectralComponent):
    _comp_name = 'acisabs'
    _config = {'Tdays': (850.0, 0.0, 10000.0, True, False), 'norm': (0.00722, 0.0, 1.0, True, False), 'tauinf': (0.582, 0.0, 1.0, True, False), 'tefold': (620.0, 1.0, 10000.0, True, False), 'nC': (10.0, 0.0, 50.0, True, False), 'nH': (20.0, 1.0, 50.0, True, False), 'nO': (2.0, 0.0, 50.0, True, False), 'nN': (1.0, 0.0, 50.0, True, False)}
    _op_class = acisabsOp
    def __init__(self, Tdays, norm, tauinf, tefold, nC, nH, nO, nN, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class acisabs(SpectralModel):
    def __init__(self, Tdays=None, norm=None, tauinf=None, tefold=None, nC=None, nH=None, nO=None, nN=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([acisabsComponent(Tdays, norm, tauinf, tefold, nC, nH, nO, nN, name, grad_method, eps)])


class agaussOp(XspecNumericGradOp):
    modname = 'agauss'
    optype = 'add'

class agaussComponent(SpectralComponent):
    _comp_name = 'agauss'
    _config = {'LineE': (10.0, 0.0, 1000000.0, False, False), 'Sigma': (1.0, 0.0, 1000000.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = agaussOp
    def __init__(self, LineE, Sigma, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class agauss(SpectralModel):
    def __init__(self, LineE=None, Sigma=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([agaussComponent(LineE, Sigma, norm, name, grad_method, eps)])


class agnsedOp(XspecNumericGradOp):
    modname = 'agnsed'
    optype = 'add'

class agnsedComponent(SpectralComponent):
    _comp_name = 'agnsed'
    _config = {'mass': (10000000.0, 1.0, 10000000000.0, True, False), 'dist': (100.0, 0.01, 1000000000.0, True, False), 'logmdot': (-1.0, -10.0, 2.0, False, False), 'astar': (0.0, -1.0, 0.998, True, False), 'cosi': (0.5, 0.05, 1.0, True, False), 'kTe_hot': (100.0, 10.0, 300.0, True, False), 'kTe_warm': (0.2, 0.1, 0.5, False, False), 'Gamma_hot': (1.7, 1.3, 3.0, False, False), 'Gamma_warm': (2.7, 2.0, 10.0, False, False), 'R_hot': (10.0, 6.0, 500.0, False, False), 'R_warm': (20.0, 6.0, 500.0, False, False), 'logrout': (-1.0, -3.0, 7.0, True, False), 'Htmax': (10.0, 6.0, 10.0, True, False), 'reprocess': (1.0, 0.0, 1.0, True, False), 'redshift': (0.0, 0.0, 1.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = agnsedOp
    def __init__(self, mass, dist, logmdot, astar, cosi, kTe_hot, kTe_warm, Gamma_hot, Gamma_warm, R_hot, R_warm, logrout, Htmax, reprocess, redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class agnsed(SpectralModel):
    def __init__(self, mass=None, dist=None, logmdot=None, astar=None, cosi=None, kTe_hot=None, kTe_warm=None, Gamma_hot=None, Gamma_warm=None, R_hot=None, R_warm=None, logrout=None, Htmax=None, reprocess=None, redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([agnsedComponent(mass, dist, logmdot, astar, cosi, kTe_hot, kTe_warm, Gamma_hot, Gamma_warm, R_hot, R_warm, logrout, Htmax, reprocess, redshift, norm, name, grad_method, eps)])


class agnslimOp(XspecNumericGradOp):
    modname = 'agnslim'
    optype = 'add'

class agnslimComponent(SpectralComponent):
    _comp_name = 'agnslim'
    _config = {'mass': (10000000.0, 1.0, 10000000000.0, True, False), 'dist': (100.0, 0.01, 1000000000.0, True, False), 'logmdot': (1.0, -10.0, 3.0, False, False), 'astar': (0.0, 0.0, 0.998, True, False), 'cosi': (0.5, 0.05, 1.0, True, False), 'kTe_hot': (100.0, 10.0, 300.0, True, False), 'kTe_warm': (0.2, 0.1, 0.5, False, False), 'Gamma_hot': (2.4, 1.3, 3.0, False, False), 'Gamma_warm': (3.0, 2.0, 10.0, False, False), 'R_hot': (10.0, 2.0, 500.0, False, False), 'R_warm': (20.0, 2.0, 500.0, False, False), 'logrout': (-1.0, -3.0, 7.0, True, False), 'rin': (-1.0, -1.0, 100.0, True, False), 'redshift': (0.0, 0.0, 5.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = agnslimOp
    def __init__(self, mass, dist, logmdot, astar, cosi, kTe_hot, kTe_warm, Gamma_hot, Gamma_warm, R_hot, R_warm, logrout, rin, redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class agnslim(SpectralModel):
    def __init__(self, mass=None, dist=None, logmdot=None, astar=None, cosi=None, kTe_hot=None, kTe_warm=None, Gamma_hot=None, Gamma_warm=None, R_hot=None, R_warm=None, logrout=None, rin=None, redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([agnslimComponent(mass, dist, logmdot, astar, cosi, kTe_hot, kTe_warm, Gamma_hot, Gamma_warm, R_hot, R_warm, logrout, rin, redshift, norm, name, grad_method, eps)])


class apecOp(XspecNumericGradOp):
    modname = 'apec'
    optype = 'add'

class apecComponent(SpectralComponent):
    _comp_name = 'apec'
    _config = {'kT': (1.0, 0.008, 64.0, False, False), 'Abundanc': (1.0, 0.0, 5.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = apecOp
    def __init__(self, kT, Abundanc, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class apec(SpectralModel):
    def __init__(self, kT=None, Abundanc=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([apecComponent(kT, Abundanc, Redshift, norm, name, grad_method, eps)])


class bapecOp(XspecNumericGradOp):
    modname = 'bapec'
    optype = 'add'

class bapecComponent(SpectralComponent):
    _comp_name = 'bapec'
    _config = {'kT': (1.0, 0.008, 64.0, False, False), 'Abundanc': (1.0, 0.0, 5.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'Velocity': (0.0, 0.0, 10000.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = bapecOp
    def __init__(self, kT, Abundanc, Redshift, Velocity, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class bapec(SpectralModel):
    def __init__(self, kT=None, Abundanc=None, Redshift=None, Velocity=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([bapecComponent(kT, Abundanc, Redshift, Velocity, norm, name, grad_method, eps)])


class bbodyOp(XspecNumericGradOp):
    modname = 'bbody'
    optype = 'add'

class bbodyComponent(SpectralComponent):
    _comp_name = 'bbody'
    _config = {'kT': (3.0, 0.0001, 200.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = bbodyOp
    def __init__(self, kT, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class bbody(SpectralModel):
    def __init__(self, kT=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([bbodyComponent(kT, norm, name, grad_method, eps)])


class bbodyradOp(XspecNumericGradOp):
    modname = 'bbodyrad'
    optype = 'add'

class bbodyradComponent(SpectralComponent):
    _comp_name = 'bbodyrad'
    _config = {'kT': (3.0, 0.0001, 200.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = bbodyradOp
    def __init__(self, kT, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class bbodyrad(SpectralModel):
    def __init__(self, kT=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([bbodyradComponent(kT, norm, name, grad_method, eps)])


class bexravOp(XspecNumericGradOp):
    modname = 'bexrav'
    optype = 'add'

class bexravComponent(SpectralComponent):
    _comp_name = 'bexrav'
    _config = {'Gamma1': (2.0, -10.0, 10.0, False, False), 'breakE': (10.0, 0.1, 1000.0, False, False), 'Gamma2': (2.0, -10.0, 10.0, False, False), 'foldE': (100.0, 1.0, 1000000.0, False, False), 'rel_refl': (0.0, 0.0, 10.0, False, False), 'cosIncl': (0.45, 0.05, 0.95, True, False), 'abund': (1.0, 0.0, 1000000.0, True, False), 'Fe_abund': (1.0, 0.0, 1000000.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = bexravOp
    def __init__(self, Gamma1, breakE, Gamma2, foldE, rel_refl, cosIncl, abund, Fe_abund, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class bexrav(SpectralModel):
    def __init__(self, Gamma1=None, breakE=None, Gamma2=None, foldE=None, rel_refl=None, cosIncl=None, abund=None, Fe_abund=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([bexravComponent(Gamma1, breakE, Gamma2, foldE, rel_refl, cosIncl, abund, Fe_abund, Redshift, norm, name, grad_method, eps)])


class bexrivOp(XspecNumericGradOp):
    modname = 'bexriv'
    optype = 'add'

class bexrivComponent(SpectralComponent):
    _comp_name = 'bexriv'
    _config = {'Gamma1': (2.0, -10.0, 10.0, False, False), 'breakE': (10.0, 0.1, 1000.0, False, False), 'Gamma2': (2.0, -10.0, 10.0, False, False), 'foldE': (100.0, 1.0, 1000000.0, False, False), 'rel_refl': (0.0, 0.0, 1000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'abund': (1.0, 0.0, 1000000.0, True, False), 'Fe_abund': (1.0, 0.0, 1000000.0, True, False), 'cosIncl': (0.45, 0.05, 0.95, True, False), 'T_disk': (30000.0, 10000.0, 1000000.0, True, False), 'xi': (1.0, 0.0, 5000.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = bexrivOp
    def __init__(self, Gamma1, breakE, Gamma2, foldE, rel_refl, Redshift, abund, Fe_abund, cosIncl, T_disk, xi, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class bexriv(SpectralModel):
    def __init__(self, Gamma1=None, breakE=None, Gamma2=None, foldE=None, rel_refl=None, Redshift=None, abund=None, Fe_abund=None, cosIncl=None, T_disk=None, xi=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([bexrivComponent(Gamma1, breakE, Gamma2, foldE, rel_refl, Redshift, abund, Fe_abund, cosIncl, T_disk, xi, norm, name, grad_method, eps)])


class bkn2powOp(XspecNumericGradOp):
    modname = 'bkn2pow'
    optype = 'add'

class bkn2powComponent(SpectralComponent):
    _comp_name = 'bkn2pow'
    _config = {'PhoIndx1': (1.0, -3.0, 10.0, False, False), 'BreakE1': (5.0, 0.0, 1000000.0, False, False), 'PhoIndx2': (2.0, -3.0, 10.0, False, False), 'BreakE2': (10.0, 0.0, 1000000.0, False, False), 'PhoIndx3': (3.0, -3.0, 10.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = bkn2powOp
    def __init__(self, PhoIndx1, BreakE1, PhoIndx2, BreakE2, PhoIndx3, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class bkn2pow(SpectralModel):
    def __init__(self, PhoIndx1=None, BreakE1=None, PhoIndx2=None, BreakE2=None, PhoIndx3=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([bkn2powComponent(PhoIndx1, BreakE1, PhoIndx2, BreakE2, PhoIndx3, norm, name, grad_method, eps)])


class bknpowerOp(XspecNumericGradOp):
    modname = 'bknpower'
    optype = 'add'

class bknpowerComponent(SpectralComponent):
    _comp_name = 'bknpower'
    _config = {'PhoIndx1': (1.0, -3.0, 10.0, False, False), 'BreakE': (5.0, 0.0, 1000000.0, False, False), 'PhoIndx2': (2.0, -3.0, 10.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = bknpowerOp
    def __init__(self, PhoIndx1, BreakE, PhoIndx2, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class bknpower(SpectralModel):
    def __init__(self, PhoIndx1=None, BreakE=None, PhoIndx2=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([bknpowerComponent(PhoIndx1, BreakE, PhoIndx2, norm, name, grad_method, eps)])


class bmcOp(XspecNumericGradOp):
    modname = 'bmc'
    optype = 'add'

class bmcComponent(SpectralComponent):
    _comp_name = 'bmc'
    _config = {'kT': (1.0, 0.0001, 200.0, False, False), 'alpha': (1.0, 0.0001, 6.0, False, False), 'log_A': (0.0, -8.0, 8.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = bmcOp
    def __init__(self, kT, alpha, log_A, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class bmc(SpectralModel):
    def __init__(self, kT=None, alpha=None, log_A=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([bmcComponent(kT, alpha, log_A, norm, name, grad_method, eps)])


class bremssOp(XspecNumericGradOp):
    modname = 'bremss'
    optype = 'add'

class bremssComponent(SpectralComponent):
    _comp_name = 'bremss'
    _config = {'kT': (7.0, 0.0001, 200.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = bremssOp
    def __init__(self, kT, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class bremss(SpectralModel):
    def __init__(self, kT=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([bremssComponent(kT, norm, name, grad_method, eps)])


class brneiOp(XspecNumericGradOp):
    modname = 'brnei'
    optype = 'add'

class brneiComponent(SpectralComponent):
    _comp_name = 'brnei'
    _config = {'kT': (0.5, 0.0808, 79.9, False, False), 'kT_init': (1.0, 0.0808, 79.9, False, False), 'Abundanc': (1.0, 0.0, 10000.0, True, False), 'Tau': (100000000000.0, 100000000.0, 50000000000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'Velocity': (0.0, 0.0, 10000.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = brneiOp
    def __init__(self, kT, kT_init, Abundanc, Tau, Redshift, Velocity, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class brnei(SpectralModel):
    def __init__(self, kT=None, kT_init=None, Abundanc=None, Tau=None, Redshift=None, Velocity=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([brneiComponent(kT, kT_init, Abundanc, Tau, Redshift, Velocity, norm, name, grad_method, eps)])


class btapecOp(XspecNumericGradOp):
    modname = 'btapec'
    optype = 'add'

class btapecComponent(SpectralComponent):
    _comp_name = 'btapec'
    _config = {'kT': (1.0, 0.008, 64.0, False, False), 'kTi': (1.0, 0.008, 64.0, False, False), 'Abundanc': (1.0, 0.0, 5.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'Velocity': (0.0, 0.0, 10000.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = btapecOp
    def __init__(self, kT, kTi, Abundanc, Redshift, Velocity, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class btapec(SpectralModel):
    def __init__(self, kT=None, kTi=None, Abundanc=None, Redshift=None, Velocity=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([btapecComponent(kT, kTi, Abundanc, Redshift, Velocity, norm, name, grad_method, eps)])


class bvapecOp(XspecNumericGradOp):
    modname = 'bvapec'
    optype = 'add'

class bvapecComponent(SpectralComponent):
    _comp_name = 'bvapec'
    _config = {'kT': (6.5, 0.0808, 68.447, False, False), 'He': (1.0, 0.0, 1000.0, True, False), 'C': (1.0, 0.0, 1000.0, True, False), 'N': (1.0, 0.0, 1000.0, True, False), 'O': (1.0, 0.0, 1000.0, True, False), 'Ne': (1.0, 0.0, 1000.0, True, False), 'Mg': (1.0, 0.0, 1000.0, True, False), 'Al': (1.0, 0.0, 1000.0, True, False), 'Si': (1.0, 0.0, 1000.0, True, False), 'S': (1.0, 0.0, 1000.0, True, False), 'Ar': (1.0, 0.0, 1000.0, True, False), 'Ca': (1.0, 0.0, 1000.0, True, False), 'Fe': (1.0, 0.0, 1000.0, True, False), 'Ni': (1.0, 0.0, 1000.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'Velocity': (0.0, 0.0, 10000.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = bvapecOp
    def __init__(self, kT, He, C, N, O, Ne, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, Velocity, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class bvapec(SpectralModel):
    def __init__(self, kT=None, He=None, C=None, N=None, O=None, Ne=None, Mg=None, Al=None, Si=None, S=None, Ar=None, Ca=None, Fe=None, Ni=None, Redshift=None, Velocity=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([bvapecComponent(kT, He, C, N, O, Ne, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, Velocity, norm, name, grad_method, eps)])


class bvrneiOp(XspecNumericGradOp):
    modname = 'bvrnei'
    optype = 'add'

class bvrneiComponent(SpectralComponent):
    _comp_name = 'bvrnei'
    _config = {'kT': (0.5, 0.0808, 79.9, False, False), 'kT_init': (1.0, 0.0808, 79.9, False, False), 'H': (1.0, 0.0, 1.0, True, False), 'He': (1.0, 0.0, 10000.0, True, False), 'C': (1.0, 0.0, 10000.0, True, False), 'N': (1.0, 0.0, 10000.0, True, False), 'O': (1.0, 0.0, 10000.0, True, False), 'Ne': (1.0, 0.0, 10000.0, True, False), 'Mg': (1.0, 0.0, 10000.0, True, False), 'Si': (1.0, 0.0, 10000.0, True, False), 'S': (1.0, 0.0, 10000.0, True, False), 'Ar': (1.0, 0.0, 10000.0, True, False), 'Ca': (1.0, 0.0, 10000.0, True, False), 'Fe': (1.0, 0.0, 10000.0, True, False), 'Ni': (1.0, 0.0, 10000.0, True, False), 'Tau': (100000000000.0, 100000000.0, 50000000000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'Velocity': (0.0, 0.0, 10000.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = bvrneiOp
    def __init__(self, kT, kT_init, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau, Redshift, Velocity, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class bvrnei(SpectralModel):
    def __init__(self, kT=None, kT_init=None, H=None, He=None, C=None, N=None, O=None, Ne=None, Mg=None, Si=None, S=None, Ar=None, Ca=None, Fe=None, Ni=None, Tau=None, Redshift=None, Velocity=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([bvrneiComponent(kT, kT_init, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau, Redshift, Velocity, norm, name, grad_method, eps)])


class bvtapecOp(XspecNumericGradOp):
    modname = 'bvtapec'
    optype = 'add'

class bvtapecComponent(SpectralComponent):
    _comp_name = 'bvtapec'
    _config = {'kT': (6.5, 0.0808, 68.447, False, False), 'kTi': (6.5, 0.0808, 68.447, False, False), 'He': (1.0, 0.0, 1000.0, True, False), 'C': (1.0, 0.0, 1000.0, True, False), 'N': (1.0, 0.0, 1000.0, True, False), 'O': (1.0, 0.0, 1000.0, True, False), 'Ne': (1.0, 0.0, 1000.0, True, False), 'Mg': (1.0, 0.0, 1000.0, True, False), 'Al': (1.0, 0.0, 1000.0, True, False), 'Si': (1.0, 0.0, 1000.0, True, False), 'S': (1.0, 0.0, 1000.0, True, False), 'Ar': (1.0, 0.0, 1000.0, True, False), 'Ca': (1.0, 0.0, 1000.0, True, False), 'Fe': (1.0, 0.0, 1000.0, True, False), 'Ni': (1.0, 0.0, 1000.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'Velocity': (0.0, 0.0, 10000.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = bvtapecOp
    def __init__(self, kT, kTi, He, C, N, O, Ne, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, Velocity, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class bvtapec(SpectralModel):
    def __init__(self, kT=None, kTi=None, He=None, C=None, N=None, O=None, Ne=None, Mg=None, Al=None, Si=None, S=None, Ar=None, Ca=None, Fe=None, Ni=None, Redshift=None, Velocity=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([bvtapecComponent(kT, kTi, He, C, N, O, Ne, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, Velocity, norm, name, grad_method, eps)])


class bvvapecOp(XspecNumericGradOp):
    modname = 'bvvapec'
    optype = 'add'

class bvvapecComponent(SpectralComponent):
    _comp_name = 'bvvapec'
    _config = {'kT': (6.5, 0.0808, 68.447, False, False), 'H': (1.0, 0.0, 1000.0, True, False), 'He': (1.0, 0.0, 1000.0, True, False), 'Li': (1.0, 0.0, 1000.0, True, False), 'Be': (1.0, 0.0, 1000.0, True, False), 'B': (1.0, 0.0, 1000.0, True, False), 'C': (1.0, 0.0, 1000.0, True, False), 'N': (1.0, 0.0, 1000.0, True, False), 'O': (1.0, 0.0, 1000.0, True, False), 'F': (1.0, 0.0, 1000.0, True, False), 'Ne': (1.0, 0.0, 1000.0, True, False), 'Na': (1.0, 0.0, 1000.0, True, False), 'Mg': (1.0, 0.0, 1000.0, True, False), 'Al': (1.0, 0.0, 1000.0, True, False), 'Si': (1.0, 0.0, 1000.0, True, False), 'P': (1.0, 0.0, 1000.0, True, False), 'S': (1.0, 0.0, 1000.0, True, False), 'Cl': (1.0, 0.0, 1000.0, True, False), 'Ar': (1.0, 0.0, 1000.0, True, False), 'K': (1.0, 0.0, 1000.0, True, False), 'Ca': (1.0, 0.0, 1000.0, True, False), 'Sc': (1.0, 0.0, 1000.0, True, False), 'Ti': (1.0, 0.0, 1000.0, True, False), 'V': (1.0, 0.0, 1000.0, True, False), 'Cr': (1.0, 0.0, 1000.0, True, False), 'Mn': (1.0, 0.0, 1000.0, True, False), 'Fe': (1.0, 0.0, 1000.0, True, False), 'Co': (1.0, 0.0, 1000.0, True, False), 'Ni': (1.0, 0.0, 1000.0, True, False), 'Cu': (1.0, 0.0, 1000.0, True, False), 'Zn': (1.0, 0.0, 1000.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'Velocity': (0.0, 0.0, 10000.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = bvvapecOp
    def __init__(self, kT, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Redshift, Velocity, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class bvvapec(SpectralModel):
    def __init__(self, kT=None, H=None, He=None, Li=None, Be=None, B=None, C=None, N=None, O=None, F=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, P=None, S=None, Cl=None, Ar=None, K=None, Ca=None, Sc=None, Ti=None, V=None, Cr=None, Mn=None, Fe=None, Co=None, Ni=None, Cu=None, Zn=None, Redshift=None, Velocity=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([bvvapecComponent(kT, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Redshift, Velocity, norm, name, grad_method, eps)])


class bvvrneiOp(XspecNumericGradOp):
    modname = 'bvvrnei'
    optype = 'add'

class bvvrneiComponent(SpectralComponent):
    _comp_name = 'bvvrnei'
    _config = {'kT': (0.5, 0.0808, 79.9, False, False), 'kT_init': (1.0, 0.0808, 79.9, False, False), 'H': (1.0, 0.0, 1000.0, True, False), 'He': (1.0, 0.0, 1000.0, True, False), 'Li': (1.0, 0.0, 1000.0, True, False), 'Be': (1.0, 0.0, 1000.0, True, False), 'B': (1.0, 0.0, 1000.0, True, False), 'C': (1.0, 0.0, 1000.0, True, False), 'N': (1.0, 0.0, 1000.0, True, False), 'O': (1.0, 0.0, 1000.0, True, False), 'F': (1.0, 0.0, 1000.0, True, False), 'Ne': (1.0, 0.0, 1000.0, True, False), 'Na': (1.0, 0.0, 1000.0, True, False), 'Mg': (1.0, 0.0, 1000.0, True, False), 'Al': (1.0, 0.0, 1000.0, True, False), 'Si': (1.0, 0.0, 1000.0, True, False), 'P': (1.0, 0.0, 1000.0, True, False), 'S': (1.0, 0.0, 1000.0, True, False), 'Cl': (1.0, 0.0, 1000.0, True, False), 'Ar': (1.0, 0.0, 1000.0, True, False), 'K': (1.0, 0.0, 1000.0, True, False), 'Ca': (1.0, 0.0, 1000.0, True, False), 'Sc': (1.0, 0.0, 1000.0, True, False), 'Ti': (1.0, 0.0, 1000.0, True, False), 'V': (1.0, 0.0, 1000.0, True, False), 'Cr': (1.0, 0.0, 1000.0, True, False), 'Mn': (1.0, 0.0, 1000.0, True, False), 'Fe': (1.0, 0.0, 1000.0, True, False), 'Co': (1.0, 0.0, 1000.0, True, False), 'Ni': (1.0, 0.0, 1000.0, True, False), 'Cu': (1.0, 0.0, 1000.0, True, False), 'Zn': (1.0, 0.0, 1000.0, True, False), 'Tau': (100000000000.0, 100000000.0, 50000000000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'Velocity': (0.0, 0.0, 10000.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = bvvrneiOp
    def __init__(self, kT, kT_init, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau, Redshift, Velocity, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class bvvrnei(SpectralModel):
    def __init__(self, kT=None, kT_init=None, H=None, He=None, Li=None, Be=None, B=None, C=None, N=None, O=None, F=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, P=None, S=None, Cl=None, Ar=None, K=None, Ca=None, Sc=None, Ti=None, V=None, Cr=None, Mn=None, Fe=None, Co=None, Ni=None, Cu=None, Zn=None, Tau=None, Redshift=None, Velocity=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([bvvrneiComponent(kT, kT_init, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau, Redshift, Velocity, norm, name, grad_method, eps)])


class bvvtapecOp(XspecNumericGradOp):
    modname = 'bvvtapec'
    optype = 'add'

class bvvtapecComponent(SpectralComponent):
    _comp_name = 'bvvtapec'
    _config = {'kT': (6.5, 0.0808, 68.447, False, False), 'kTi': (6.5, 0.0808, 68.447, False, False), 'H': (1.0, 0.0, 1000.0, True, False), 'He': (1.0, 0.0, 1000.0, True, False), 'Li': (1.0, 0.0, 1000.0, True, False), 'Be': (1.0, 0.0, 1000.0, True, False), 'B': (1.0, 0.0, 1000.0, True, False), 'C': (1.0, 0.0, 1000.0, True, False), 'N': (1.0, 0.0, 1000.0, True, False), 'O': (1.0, 0.0, 1000.0, True, False), 'F': (1.0, 0.0, 1000.0, True, False), 'Ne': (1.0, 0.0, 1000.0, True, False), 'Na': (1.0, 0.0, 1000.0, True, False), 'Mg': (1.0, 0.0, 1000.0, True, False), 'Al': (1.0, 0.0, 1000.0, True, False), 'Si': (1.0, 0.0, 1000.0, True, False), 'P': (1.0, 0.0, 1000.0, True, False), 'S': (1.0, 0.0, 1000.0, True, False), 'Cl': (1.0, 0.0, 1000.0, True, False), 'Ar': (1.0, 0.0, 1000.0, True, False), 'K': (1.0, 0.0, 1000.0, True, False), 'Ca': (1.0, 0.0, 1000.0, True, False), 'Sc': (1.0, 0.0, 1000.0, True, False), 'Ti': (1.0, 0.0, 1000.0, True, False), 'V': (1.0, 0.0, 1000.0, True, False), 'Cr': (1.0, 0.0, 1000.0, True, False), 'Mn': (1.0, 0.0, 1000.0, True, False), 'Fe': (1.0, 0.0, 1000.0, True, False), 'Co': (1.0, 0.0, 1000.0, True, False), 'Ni': (1.0, 0.0, 1000.0, True, False), 'Cu': (1.0, 0.0, 1000.0, True, False), 'Zn': (1.0, 0.0, 1000.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'Velocity': (0.0, 0.0, 10000.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = bvvtapecOp
    def __init__(self, kT, kTi, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Redshift, Velocity, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class bvvtapec(SpectralModel):
    def __init__(self, kT=None, kTi=None, H=None, He=None, Li=None, Be=None, B=None, C=None, N=None, O=None, F=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, P=None, S=None, Cl=None, Ar=None, K=None, Ca=None, Sc=None, Ti=None, V=None, Cr=None, Mn=None, Fe=None, Co=None, Ni=None, Cu=None, Zn=None, Redshift=None, Velocity=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([bvvtapecComponent(kT, kTi, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Redshift, Velocity, norm, name, grad_method, eps)])


class bwcyclOp(XspecNumericGradOp):
    modname = 'bwcycl'
    optype = 'add'

class bwcyclComponent(SpectralComponent):
    _comp_name = 'bwcycl'
    _config = {'Radius': (10.0, 5.0, 20.0, True, False), 'Mass': (1.4, 1.0, 3.0, True, False), 'csi': (1.5, 0.01, 20.0, False, False), 'delta': (1.8, 0.01, 20.0, False, False), 'B': (4.0, 0.01, 100.0, False, False), 'Mdot': (1.0, 1e-06, 1000000.0, False, False), 'Te': (5.0, 0.1, 100.0, False, False), 'r0': (44.0, 10.0, 1000.0, False, False), 'D': (5.0, 1.0, 20.0, True, False), 'BBnorm': (0.0, 0.0, 100.0, True, False), 'CYCnorm': (1.0, -1.0, 100.0, True, False), 'FFnorm': (1.0, -1.0, 100.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = bwcyclOp
    def __init__(self, Radius, Mass, csi, delta, B, Mdot, Te, r0, D, BBnorm, CYCnorm, FFnorm, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class bwcycl(SpectralModel):
    def __init__(self, Radius=None, Mass=None, csi=None, delta=None, B=None, Mdot=None, Te=None, r0=None, D=None, BBnorm=None, CYCnorm=None, FFnorm=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([bwcyclComponent(Radius, Mass, csi, delta, B, Mdot, Te, r0, D, BBnorm, CYCnorm, FFnorm, norm, name, grad_method, eps)])


class c6meklOp(XspecNumericGradOp):
    modname = 'c6mekl'
    optype = 'add'

class c6meklComponent(SpectralComponent):
    _comp_name = 'c6mekl'
    _config = {'CPcoef1': (1.0, -1.0, 1.0, False, False), 'CPcoef2': (0.5, -1.0, 1.0, False, False), 'CPcoef3': (0.5, -1.0, 1.0, False, False), 'CPcoef4': (0.5, -1.0, 1.0, False, False), 'CPcoef5': (0.5, -1.0, 1.0, False, False), 'CPcoef6': (0.5, -1.0, 1.0, False, False), 'nH': (1.0, 1e-06, 1e+20, True, False), 'abundanc': (1.0, 0.0, 10.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'switch': (1, None, None, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = c6meklOp
    def __init__(self, CPcoef1, CPcoef2, CPcoef3, CPcoef4, CPcoef5, CPcoef6, nH, abundanc, Redshift, switch, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class c6mekl(SpectralModel):
    def __init__(self, CPcoef1=None, CPcoef2=None, CPcoef3=None, CPcoef4=None, CPcoef5=None, CPcoef6=None, nH=None, abundanc=None, Redshift=None, switch=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([c6meklComponent(CPcoef1, CPcoef2, CPcoef3, CPcoef4, CPcoef5, CPcoef6, nH, abundanc, Redshift, switch, norm, name, grad_method, eps)])


class c6pmeklOp(XspecNumericGradOp):
    modname = 'c6pmekl'
    optype = 'add'

class c6pmeklComponent(SpectralComponent):
    _comp_name = 'c6pmekl'
    _config = {'CPcoef1': (1.0, -1.0, 1.0, False, False), 'CPcoef2': (0.5, -1.0, 1.0, False, False), 'CPcoef3': (0.5, -1.0, 1.0, False, False), 'CPcoef4': (0.5, -1.0, 1.0, False, False), 'CPcoef5': (0.5, -1.0, 1.0, False, False), 'CPcoef6': (0.5, -1.0, 1.0, False, False), 'nH': (1.0, 1e-06, 1e+20, True, False), 'abundanc': (1.0, 0.0, 10.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'switch': (1, None, None, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = c6pmeklOp
    def __init__(self, CPcoef1, CPcoef2, CPcoef3, CPcoef4, CPcoef5, CPcoef6, nH, abundanc, Redshift, switch, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class c6pmekl(SpectralModel):
    def __init__(self, CPcoef1=None, CPcoef2=None, CPcoef3=None, CPcoef4=None, CPcoef5=None, CPcoef6=None, nH=None, abundanc=None, Redshift=None, switch=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([c6pmeklComponent(CPcoef1, CPcoef2, CPcoef3, CPcoef4, CPcoef5, CPcoef6, nH, abundanc, Redshift, switch, norm, name, grad_method, eps)])


class c6pvmklOp(XspecNumericGradOp):
    modname = 'c6pvmkl'
    optype = 'add'

class c6pvmklComponent(SpectralComponent):
    _comp_name = 'c6pvmkl'
    _config = {'CPcoef1': (1.0, -1.0, 1.0, False, False), 'CPcoef2': (0.5, -1.0, 1.0, False, False), 'CPcoef3': (0.5, -1.0, 1.0, False, False), 'CPcoef4': (0.5, -1.0, 1.0, False, False), 'CPcoef5': (0.5, -1.0, 1.0, False, False), 'CPcoef6': (0.5, -1.0, 1.0, False, False), 'nH': (1.0, 1e-06, 1e+20, True, False), 'He': (1.0, 0.0, 10.0, True, False), 'C': (1.0, 0.0, 10.0, True, False), 'N': (1.0, 0.0, 10.0, True, False), 'O': (1.0, 0.0, 10.0, True, False), 'Ne': (1.0, 0.0, 10.0, True, False), 'Na': (1.0, 0.0, 10.0, True, False), 'Mg': (1.0, 0.0, 10.0, True, False), 'Al': (1.0, 0.0, 10.0, True, False), 'Si': (1.0, 0.0, 10.0, True, False), 'S': (1.0, 0.0, 10.0, True, False), 'Ar': (1.0, 0.0, 10.0, True, False), 'Ca': (1.0, 0.0, 10.0, True, False), 'Fe': (1.0, 0.0, 10.0, True, False), 'Ni': (1.0, 0.0, 10.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'switch': (1, None, None, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = c6pvmklOp
    def __init__(self, CPcoef1, CPcoef2, CPcoef3, CPcoef4, CPcoef5, CPcoef6, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class c6pvmkl(SpectralModel):
    def __init__(self, CPcoef1=None, CPcoef2=None, CPcoef3=None, CPcoef4=None, CPcoef5=None, CPcoef6=None, nH=None, He=None, C=None, N=None, O=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, S=None, Ar=None, Ca=None, Fe=None, Ni=None, Redshift=None, switch=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([c6pvmklComponent(CPcoef1, CPcoef2, CPcoef3, CPcoef4, CPcoef5, CPcoef6, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch, norm, name, grad_method, eps)])


class c6vmeklOp(XspecNumericGradOp):
    modname = 'c6vmekl'
    optype = 'add'

class c6vmeklComponent(SpectralComponent):
    _comp_name = 'c6vmekl'
    _config = {'CPcoef1': (1.0, -1.0, 1.0, False, False), 'CPcoef2': (0.5, -1.0, 1.0, False, False), 'CPcoef3': (0.5, -1.0, 1.0, False, False), 'CPcoef4': (0.5, -1.0, 1.0, False, False), 'CPcoef5': (0.5, -1.0, 1.0, False, False), 'CPcoef6': (0.5, -1.0, 1.0, False, False), 'nH': (1.0, 1e-06, 1e+20, True, False), 'He': (1.0, 0.0, 10.0, True, False), 'C': (1.0, 0.0, 10.0, True, False), 'N': (1.0, 0.0, 10.0, True, False), 'O': (1.0, 0.0, 10.0, True, False), 'Ne': (1.0, 0.0, 10.0, True, False), 'Na': (1.0, 0.0, 10.0, True, False), 'Mg': (1.0, 0.0, 10.0, True, False), 'Al': (1.0, 0.0, 10.0, True, False), 'Si': (1.0, 0.0, 10.0, True, False), 'S': (1.0, 0.0, 10.0, True, False), 'Ar': (1.0, 0.0, 10.0, True, False), 'Ca': (1.0, 0.0, 10.0, True, False), 'Fe': (1.0, 0.0, 10.0, True, False), 'Ni': (1.0, 0.0, 10.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'switch': (1, None, None, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = c6vmeklOp
    def __init__(self, CPcoef1, CPcoef2, CPcoef3, CPcoef4, CPcoef5, CPcoef6, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class c6vmekl(SpectralModel):
    def __init__(self, CPcoef1=None, CPcoef2=None, CPcoef3=None, CPcoef4=None, CPcoef5=None, CPcoef6=None, nH=None, He=None, C=None, N=None, O=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, S=None, Ar=None, Ca=None, Fe=None, Ni=None, Redshift=None, switch=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([c6vmeklComponent(CPcoef1, CPcoef2, CPcoef3, CPcoef4, CPcoef5, CPcoef6, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch, norm, name, grad_method, eps)])


class cabsOp(XspecNumericGradOp):
    modname = 'cabs'
    optype = 'mul'

class cabsComponent(SpectralComponent):
    _comp_name = 'cabs'
    _config = {'nH': (1.0, 0.0, 1000000.0, False, False)}
    _op_class = cabsOp
    def __init__(self, nH, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class cabs(SpectralModel):
    def __init__(self, nH=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([cabsComponent(nH, name, grad_method, eps)])


class carbatmOp(XspecNumericGradOp):
    modname = 'carbatm'
    optype = 'add'

class carbatmComponent(SpectralComponent):
    _comp_name = 'carbatm'
    _config = {'T': (2.0, 1.0, 4.0, False, False), 'NSmass': (1.4, 0.6, 2.8, False, False), 'NSrad': (10.0, 6.0, 23.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = carbatmOp
    def __init__(self, T, NSmass, NSrad, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class carbatm(SpectralModel):
    def __init__(self, T=None, NSmass=None, NSrad=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([carbatmComponent(T, NSmass, NSrad, norm, name, grad_method, eps)])


class cemeklOp(XspecNumericGradOp):
    modname = 'cemekl'
    optype = 'add'

class cemeklComponent(SpectralComponent):
    _comp_name = 'cemekl'
    _config = {'alpha': (1.0, 0.01, 20.0, True, False), 'Tmax': (1.0, 0.01, 100.0, False, False), 'nH': (1.0, 1e-06, 1e+20, True, False), 'abundanc': (1.0, 0.0, 10.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'switch': (1, 0.0, 1.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = cemeklOp
    def __init__(self, alpha, Tmax, nH, abundanc, Redshift, switch, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class cemekl(SpectralModel):
    def __init__(self, alpha=None, Tmax=None, nH=None, abundanc=None, Redshift=None, switch=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([cemeklComponent(alpha, Tmax, nH, abundanc, Redshift, switch, norm, name, grad_method, eps)])


class cevmklOp(XspecNumericGradOp):
    modname = 'cevmkl'
    optype = 'add'

class cevmklComponent(SpectralComponent):
    _comp_name = 'cevmkl'
    _config = {'alpha': (1.0, 0.01, 20.0, True, False), 'Tmax': (1.0, 0.01, 100.0, False, False), 'nH': (1.0, 1e-06, 1e+20, True, False), 'He': (1.0, 0.0, 10.0, True, False), 'C': (1.0, 0.0, 10.0, True, False), 'N': (1.0, 0.0, 10.0, True, False), 'O': (1.0, 0.0, 10.0, True, False), 'Ne': (1.0, 0.0, 10.0, True, False), 'Na': (1.0, 0.0, 10.0, True, False), 'Mg': (1.0, 0.0, 10.0, True, False), 'Al': (1.0, 0.0, 10.0, True, False), 'Si': (1.0, 0.0, 10.0, True, False), 'S': (1.0, 0.0, 10.0, True, False), 'Ar': (1.0, 0.0, 10.0, True, False), 'Ca': (1.0, 0.0, 10.0, True, False), 'Fe': (1.0, 0.0, 10.0, True, False), 'Ni': (1.0, 0.0, 10.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'switch': (1, 0.0, 1.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = cevmklOp
    def __init__(self, alpha, Tmax, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class cevmkl(SpectralModel):
    def __init__(self, alpha=None, Tmax=None, nH=None, He=None, C=None, N=None, O=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, S=None, Ar=None, Ca=None, Fe=None, Ni=None, Redshift=None, switch=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([cevmklComponent(alpha, Tmax, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch, norm, name, grad_method, eps)])


class cflowOp(XspecNumericGradOp):
    modname = 'cflow'
    optype = 'add'

class cflowComponent(SpectralComponent):
    _comp_name = 'cflow'
    _config = {'slope': (0.0, -5.0, 5.0, False, False), 'lowT': (0.1, 0.0808, 79.9, False, False), 'highT': (4.0, 0.0808, 79.9, False, False), 'Abundanc': (1.0, 0.0, 5.0, False, False), 'redshift': (0.1, 1e-10, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = cflowOp
    def __init__(self, slope, lowT, highT, Abundanc, redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class cflow(SpectralModel):
    def __init__(self, slope=None, lowT=None, highT=None, Abundanc=None, redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([cflowComponent(slope, lowT, highT, Abundanc, redshift, norm, name, grad_method, eps)])


class cfluxOp(XspecNumericGradOp):
    modname = 'cflux'
    optype = 'con'

class cfluxComponent(SpectralComponent):
    _comp_name = 'cflux'
    _config = {'Emin': (0.5, 0.0, 1000000.0, True, False), 'Emax': (10.0, 0.0, 1000000.0, True, False), 'lg10Flux': (-12.0, -100.0, 100.0, False, False)}
    _op_class = cfluxOp
    def __init__(self, Emin, Emax, lg10Flux, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class cflux(SpectralModel):
    def __init__(self, Emin=None, Emax=None, lg10Flux=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([cfluxComponent(Emin, Emax, lg10Flux, name, grad_method, eps)])


class cgluminOp(XspecNumericGradOp):
    modname = 'cglumin'
    optype = 'con'

class cgluminComponent(SpectralComponent):
    _comp_name = 'cglumin'
    _config = {'Emin': (0.5, 0.0, 1000000.0, True, False), 'Emax': (10.0, 0.0, 1000000.0, True, False), 'Distance': (10.0, 0.0, 1000000.0, True, False), 'lg10Lum': (40.0, -100.0, 100.0, False, False)}
    _op_class = cgluminOp
    def __init__(self, Emin, Emax, Distance, lg10Lum, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class cglumin(SpectralModel):
    def __init__(self, Emin=None, Emax=None, Distance=None, lg10Lum=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([cgluminComponent(Emin, Emax, Distance, lg10Lum, name, grad_method, eps)])


class cluminOp(XspecNumericGradOp):
    modname = 'clumin'
    optype = 'con'

class cluminComponent(SpectralComponent):
    _comp_name = 'clumin'
    _config = {'Emin': (0.5, 0.0, 1000000.0, True, False), 'Emax': (10.0, 0.0, 1000000.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'lg10Lum': (40.0, -100.0, 100.0, False, False)}
    _op_class = cluminOp
    def __init__(self, Emin, Emax, Redshift, lg10Lum, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class clumin(SpectralModel):
    def __init__(self, Emin=None, Emax=None, Redshift=None, lg10Lum=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([cluminComponent(Emin, Emax, Redshift, lg10Lum, name, grad_method, eps)])


class compLSOp(XspecNumericGradOp):
    modname = 'compLS'
    optype = 'add'

class compLSComponent(SpectralComponent):
    _comp_name = 'compLS'
    _config = {'kT': (2.0, 0.001, 20.0, False, False), 'tau': (10.0, 0.0001, 200.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = compLSOp
    def __init__(self, kT, tau, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class compLS(SpectralModel):
    def __init__(self, kT=None, tau=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([compLSComponent(kT, tau, norm, name, grad_method, eps)])


class compPSOp(XspecNumericGradOp):
    modname = 'compPS'
    optype = 'add'

class compPSComponent(SpectralComponent):
    _comp_name = 'compPS'
    _config = {'kTe': (100.0, 20.0, 100000.0, False, False), 'EleIndex': (2.0, 0.0, 5.0, True, False), 'Gmin': (-1.0, -1.0, 10.0, True, False), 'Gmax': (1000.0, 10.0, 10000.0, True, False), 'kTbb': (0.1, 0.001, 10.0, True, False), 'tau_y': (1.0, 0.05, 3.0, False, False), 'geom': (0.0, -5.0, 4.0, True, False), 'HovR_cyl': (1.0, 0.5, 2.0, True, False), 'cosIncl': (0.5, 0.05, 0.95, True, False), 'cov_frac': (1.0, 0.0, 1.0, True, False), 'rel_refl': (0.0, 0.0, 10000.0, True, False), 'Fe_ab_re': (1.0, 0.1, 10.0, True, False), 'Me_ab': (1.0, 0.1, 10.0, True, False), 'xi': (0.0, 0.0, 100000.0, True, False), 'Tdisk': (1000000.0, 10000.0, 1000000.0, True, False), 'Betor10': (-10.0, -10.0, 10.0, True, False), 'Rin': (10.0, 6.001, 10000.0, True, False), 'Rout': (1000.0, 0.0, 1000000.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = compPSOp
    def __init__(self, kTe, EleIndex, Gmin, Gmax, kTbb, tau_y, geom, HovR_cyl, cosIncl, cov_frac, rel_refl, Fe_ab_re, Me_ab, xi, Tdisk, Betor10, Rin, Rout, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class compPS(SpectralModel):
    def __init__(self, kTe=None, EleIndex=None, Gmin=None, Gmax=None, kTbb=None, tau_y=None, geom=None, HovR_cyl=None, cosIncl=None, cov_frac=None, rel_refl=None, Fe_ab_re=None, Me_ab=None, xi=None, Tdisk=None, Betor10=None, Rin=None, Rout=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([compPSComponent(kTe, EleIndex, Gmin, Gmax, kTbb, tau_y, geom, HovR_cyl, cosIncl, cov_frac, rel_refl, Fe_ab_re, Me_ab, xi, Tdisk, Betor10, Rin, Rout, Redshift, norm, name, grad_method, eps)])


class compSTOp(XspecNumericGradOp):
    modname = 'compST'
    optype = 'add'

class compSTComponent(SpectralComponent):
    _comp_name = 'compST'
    _config = {'kT': (2.0, 0.001, 100.0, False, False), 'tau': (10.0, 0.0001, 200.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = compSTOp
    def __init__(self, kT, tau, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class compST(SpectralModel):
    def __init__(self, kT=None, tau=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([compSTComponent(kT, tau, norm, name, grad_method, eps)])


class compTTOp(XspecNumericGradOp):
    modname = 'compTT'
    optype = 'add'

class compTTComponent(SpectralComponent):
    _comp_name = 'compTT'
    _config = {'Redshift': (0.0, -0.999, 10.0, True, False), 'T0': (0.1, 0.001, 100.0, False, False), 'kT': (50.0, 2.0, 500.0, False, False), 'taup': (1.0, 0.01, 200.0, False, False), 'approx': (1.0, 0.0, 200.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = compTTOp
    def __init__(self, Redshift, T0, kT, taup, approx, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class compTT(SpectralModel):
    def __init__(self, Redshift=None, T0=None, kT=None, taup=None, approx=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([compTTComponent(Redshift, T0, kT, taup, approx, norm, name, grad_method, eps)])


class compbbOp(XspecNumericGradOp):
    modname = 'compbb'
    optype = 'add'

class compbbComponent(SpectralComponent):
    _comp_name = 'compbb'
    _config = {'kT': (1.0, 0.0001, 200.0, False, False), 'kTe': (50.0, 1.0, 200.0, True, False), 'tau': (0.1, 0.0, 10.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = compbbOp
    def __init__(self, kT, kTe, tau, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class compbb(SpectralModel):
    def __init__(self, kT=None, kTe=None, tau=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([compbbComponent(kT, kTe, tau, norm, name, grad_method, eps)])


class compmagOp(XspecNumericGradOp):
    modname = 'compmag'
    optype = 'add'

class compmagComponent(SpectralComponent):
    _comp_name = 'compmag'
    _config = {'kTbb': (1.0, 0.2, 10.0, False, False), 'kTe': (5.0, 0.2, 2000.0, False, False), 'tau': (0.5, 0.0, 10.0, False, False), 'eta': (0.5, 0.01, 1.0, False, False), 'beta0': (0.57, 0.0001, 1.0, False, False), 'r0': (0.25, 0.0001, 100.0, False, False), 'A': (0.001, 0.0, 1.0, True, False), 'betaflag': (1.0, 0.0, 2.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = compmagOp
    def __init__(self, kTbb, kTe, tau, eta, beta0, r0, A, betaflag, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class compmag(SpectralModel):
    def __init__(self, kTbb=None, kTe=None, tau=None, eta=None, beta0=None, r0=None, A=None, betaflag=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([compmagComponent(kTbb, kTe, tau, eta, beta0, r0, A, betaflag, norm, name, grad_method, eps)])


class comptbOp(XspecNumericGradOp):
    modname = 'comptb'
    optype = 'add'

class comptbComponent(SpectralComponent):
    _comp_name = 'comptb'
    _config = {'kTs': (1.0, 0.1, 10.0, False, False), 'gamma': (3.0, 1.0, 10.0, True, False), 'alpha': (2.0, 0.0, 400.0, False, False), 'delta': (20.0, 0.0, 200.0, False, False), 'kTe': (5.0, 0.2, 2000.0, False, False), 'log_A': (0.0, -8.0, 8.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = comptbOp
    def __init__(self, kTs, gamma, alpha, delta, kTe, log_A, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class comptb(SpectralModel):
    def __init__(self, kTs=None, gamma=None, alpha=None, delta=None, kTe=None, log_A=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([comptbComponent(kTs, gamma, alpha, delta, kTe, log_A, norm, name, grad_method, eps)])


class compthOp(XspecNumericGradOp):
    modname = 'compth'
    optype = 'add'

class compthComponent(SpectralComponent):
    _comp_name = 'compth'
    _config = {'theta': (1.0, 1e-06, 1000000.0, False, False), 'showbb': (1.0, 0.0, 10000.0, True, False), 'kT_bb': (200.0, 1.0, 400000.0, True, False), 'RefOn': (-1.0, -2.0, 2.0, True, False), 'tau_p': (0.1, 0.0001, 10.0, True, False), 'radius': (10000000.0, 100000.0, 1e+16, True, False), 'g_min': (1.3, 1.2, 1000.0, True, False), 'g_max': (1000.0, 5.0, 10000.0, True, False), 'G_inj': (2.0, 0.0, 5.0, True, False), 'pairinj': (0.0, 0.0, 1.0, True, False), 'cosIncl': (0.5, 0.05, 0.95, True, False), 'Refl': (1.0, 0.0, 2.0, True, False), 'Fe_abund': (1.0, 0.1, 10.0, True, False), 'Ab_met': (1.0, 0.1, 10.0, True, False), 'T_disk': (1000000.0, 10000.0, 1000000.0, True, False), 'xi': (0.0, 0.0, 5000.0, False, False), 'Beta': (-10.0, -10.0, 10.0, True, False), 'Rin': (10.0, 6.001, 10000.0, True, False), 'Rout': (1000.0, 0.0, 1000000.0, True, False), 'redshift': (0.0, 0.0, 4.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = compthOp
    def __init__(self, theta, showbb, kT_bb, RefOn, tau_p, radius, g_min, g_max, G_inj, pairinj, cosIncl, Refl, Fe_abund, Ab_met, T_disk, xi, Beta, Rin, Rout, redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class compth(SpectralModel):
    def __init__(self, theta=None, showbb=None, kT_bb=None, RefOn=None, tau_p=None, radius=None, g_min=None, g_max=None, G_inj=None, pairinj=None, cosIncl=None, Refl=None, Fe_abund=None, Ab_met=None, T_disk=None, xi=None, Beta=None, Rin=None, Rout=None, redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([compthComponent(theta, showbb, kT_bb, RefOn, tau_p, radius, g_min, g_max, G_inj, pairinj, cosIncl, Refl, Fe_abund, Ab_met, T_disk, xi, Beta, Rin, Rout, redshift, norm, name, grad_method, eps)])


class constantOp(XspecNumericGradOp):
    modname = 'constant'
    optype = 'mul'

class constantComponent(SpectralComponent):
    _comp_name = 'constant'
    _config = {'factor': (1.0, 0.0, 10000000000.0, False, False)}
    _op_class = constantOp
    def __init__(self, factor, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class constant(SpectralModel):
    def __init__(self, factor=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([constantComponent(factor, name, grad_method, eps)])


class cpfluxOp(XspecNumericGradOp):
    modname = 'cpflux'
    optype = 'con'

class cpfluxComponent(SpectralComponent):
    _comp_name = 'cpflux'
    _config = {'Emin': (0.5, 0.0, 1000000.0, True, False), 'Emax': (10.0, 0.0, 1000000.0, True, False), 'Flux': (1.0, 0.0, 10000000000.0, False, False)}
    _op_class = cpfluxOp
    def __init__(self, Emin, Emax, Flux, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class cpflux(SpectralModel):
    def __init__(self, Emin=None, Emax=None, Flux=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([cpfluxComponent(Emin, Emax, Flux, name, grad_method, eps)])


class cphOp(XspecNumericGradOp):
    modname = 'cph'
    optype = 'add'

class cphComponent(SpectralComponent):
    _comp_name = 'cph'
    _config = {'peakT': (2.2, 0.1, 100.0, False, False), 'Abund': (1.0, 0.0, 1000.0, True, False), 'Redshift': (0.0, 0.0, 50.0, True, False), 'switch': (1, None, None, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = cphOp
    def __init__(self, peakT, Abund, Redshift, switch, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class cph(SpectralModel):
    def __init__(self, peakT=None, Abund=None, Redshift=None, switch=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([cphComponent(peakT, Abund, Redshift, switch, norm, name, grad_method, eps)])


class cplinearOp(XspecNumericGradOp):
    modname = 'cplinear'
    optype = 'add'

class cplinearComponent(SpectralComponent):
    _comp_name = 'cplinear'
    _config = {'energy00': (0.5, 0.0, 10.0, True, False), 'energy01': (1.0, 0.0, 10.0, True, False), 'energy02': (1.5, 0.0, 10.0, True, False), 'energy03': (2.0, 0.0, 10.0, True, False), 'energy04': (3.0, 0.0, 10.0, True, False), 'energy05': (4.0, 0.0, 10.0, True, False), 'energy06': (5.0, 0.0, 10.0, True, False), 'energy07': (6.0, 0.0, 10.0, True, False), 'energy08': (7.0, 0.0, 10.0, True, False), 'energy09': (8.0, 0.0, 10.0, True, False), 'log_rate00': (0.0, -20.0, 20.0, True, False), 'log_rate01': (1.0, -20.0, 20.0, True, False), 'log_rate02': (0.0, -20.0, 20.0, True, False), 'log_rate03': (1.0, -20.0, 20.0, True, False), 'log_rate04': (0.0, -20.0, 20.0, True, False), 'log_rate05': (1.0, -20.0, 20.0, True, False), 'log_rate06': (0.0, -20.0, 20.0, True, False), 'log_rate07': (1.0, -20.0, 20.0, True, False), 'log_rate08': (0.0, -20.0, 20.0, True, False), 'log_rate09': (1.0, -20.0, 20.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = cplinearOp
    def __init__(self, energy00, energy01, energy02, energy03, energy04, energy05, energy06, energy07, energy08, energy09, log_rate00, log_rate01, log_rate02, log_rate03, log_rate04, log_rate05, log_rate06, log_rate07, log_rate08, log_rate09, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class cplinear(SpectralModel):
    def __init__(self, energy00=None, energy01=None, energy02=None, energy03=None, energy04=None, energy05=None, energy06=None, energy07=None, energy08=None, energy09=None, log_rate00=None, log_rate01=None, log_rate02=None, log_rate03=None, log_rate04=None, log_rate05=None, log_rate06=None, log_rate07=None, log_rate08=None, log_rate09=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([cplinearComponent(energy00, energy01, energy02, energy03, energy04, energy05, energy06, energy07, energy08, energy09, log_rate00, log_rate01, log_rate02, log_rate03, log_rate04, log_rate05, log_rate06, log_rate07, log_rate08, log_rate09, norm, name, grad_method, eps)])


class cutoffplOp(XspecNumericGradOp):
    modname = 'cutoffpl'
    optype = 'add'

class cutoffplComponent(SpectralComponent):
    _comp_name = 'cutoffpl'
    _config = {'PhoIndex': (1.0, -3.0, 10.0, False, False), 'HighECut': (15.0, 0.01, 500.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = cutoffplOp
    def __init__(self, PhoIndex, HighECut, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class cutoffpl(SpectralModel):
    def __init__(self, PhoIndex=None, HighECut=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([cutoffplComponent(PhoIndex, HighECut, norm, name, grad_method, eps)])


class cyclabsOp(XspecNumericGradOp):
    modname = 'cyclabs'
    optype = 'mul'

class cyclabsComponent(SpectralComponent):
    _comp_name = 'cyclabs'
    _config = {'Depth0': (2.0, 0.0, 100.0, False, False), 'E0': (30.0, 1.0, 100.0, False, False), 'Width0': (10.0, 1.0, 100.0, True, False), 'Depth2': (0.0, 0.0, 100.0, True, False), 'Width2': (20.0, 1.0, 100.0, True, False)}
    _op_class = cyclabsOp
    def __init__(self, Depth0, E0, Width0, Depth2, Width2, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class cyclabs(SpectralModel):
    def __init__(self, Depth0=None, E0=None, Width0=None, Depth2=None, Width2=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([cyclabsComponent(Depth0, E0, Width0, Depth2, Width2, name, grad_method, eps)])


class diskOp(XspecNumericGradOp):
    modname = 'disk'
    optype = 'add'

class diskComponent(SpectralComponent):
    _comp_name = 'disk'
    _config = {'accrate': (1.0, 0.0001, 10.0, False, False), 'CenMass': (1.4, 0.1, 20.0, True, False), 'Rinn': (1.03, 1.0, 1.04, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = diskOp
    def __init__(self, accrate, CenMass, Rinn, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class disk(SpectralModel):
    def __init__(self, accrate=None, CenMass=None, Rinn=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([diskComponent(accrate, CenMass, Rinn, norm, name, grad_method, eps)])


class diskbbOp(XspecNumericGradOp):
    modname = 'diskbb'
    optype = 'add'

class diskbbComponent(SpectralComponent):
    _comp_name = 'diskbb'
    _config = {'Tin': (1.0, 0.0, 1000.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = diskbbOp
    def __init__(self, Tin, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class diskbb(SpectralModel):
    def __init__(self, Tin=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([diskbbComponent(Tin, norm, name, grad_method, eps)])


class diskirOp(XspecNumericGradOp):
    modname = 'diskir'
    optype = 'add'

class diskirComponent(SpectralComponent):
    _comp_name = 'diskir'
    _config = {'kT_disk': (1.0, 0.01, 5.0, False, False), 'Gamma': (1.7, 1.001, 10.0, False, False), 'kT_e': (100.0, 1.0, 1000.0, False, False), 'LcovrLd': (0.1, 0.0, 10.0, False, False), 'fin': (0.1, 0.0, 1.0, True, False), 'rirr': (1.2, 1.0001, 10.0, False, False), 'fout': (0.0001, 0.0, 0.1, False, False), 'logrout': (5.0, 3.0, 7.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = diskirOp
    def __init__(self, kT_disk, Gamma, kT_e, LcovrLd, fin, rirr, fout, logrout, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class diskir(SpectralModel):
    def __init__(self, kT_disk=None, Gamma=None, kT_e=None, LcovrLd=None, fin=None, rirr=None, fout=None, logrout=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([diskirComponent(kT_disk, Gamma, kT_e, LcovrLd, fin, rirr, fout, logrout, norm, name, grad_method, eps)])


class disklineOp(XspecNumericGradOp):
    modname = 'diskline'
    optype = 'add'

class disklineComponent(SpectralComponent):
    _comp_name = 'diskline'
    _config = {'LineE': (6.7, 0.0, 100.0, False, False), 'Betor10': (-2.0, -10.0, 20.0, True, False), 'Rin_M': (10.0, 6.0, 10000.0, True, False), 'Rout_M': (1000.0, 0.0, 10000000.0, True, False), 'Incl': (30.0, 0.0, 90.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = disklineOp
    def __init__(self, LineE, Betor10, Rin_M, Rout_M, Incl, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class diskline(SpectralModel):
    def __init__(self, LineE=None, Betor10=None, Rin_M=None, Rout_M=None, Incl=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([disklineComponent(LineE, Betor10, Rin_M, Rout_M, Incl, norm, name, grad_method, eps)])


class diskmOp(XspecNumericGradOp):
    modname = 'diskm'
    optype = 'add'

class diskmComponent(SpectralComponent):
    _comp_name = 'diskm'
    _config = {'accrate': (1.0, 0.0001, 10.0, False, False), 'NSmass': (1.4, 0.1, 20.0, True, False), 'Rinn': (1.03, 1.0, 1.04, True, False), 'alpha': (1.0, 0.001, 20.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = diskmOp
    def __init__(self, accrate, NSmass, Rinn, alpha, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class diskm(SpectralModel):
    def __init__(self, accrate=None, NSmass=None, Rinn=None, alpha=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([diskmComponent(accrate, NSmass, Rinn, alpha, norm, name, grad_method, eps)])


class diskoOp(XspecNumericGradOp):
    modname = 'disko'
    optype = 'add'

class diskoComponent(SpectralComponent):
    _comp_name = 'disko'
    _config = {'accrate': (1.0, 0.0001, 10.0, False, False), 'NSmass': (1.4, 0.1, 20.0, True, False), 'Rinn': (1.03, 1.0, 1.04, True, False), 'alpha': (1.0, 0.001, 20.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = diskoOp
    def __init__(self, accrate, NSmass, Rinn, alpha, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class disko(SpectralModel):
    def __init__(self, accrate=None, NSmass=None, Rinn=None, alpha=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([diskoComponent(accrate, NSmass, Rinn, alpha, norm, name, grad_method, eps)])


class diskpbbOp(XspecNumericGradOp):
    modname = 'diskpbb'
    optype = 'add'

class diskpbbComponent(SpectralComponent):
    _comp_name = 'diskpbb'
    _config = {'Tin': (1.0, 0.1, 10.0, False, False), 'p': (0.75, 0.5, 1.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = diskpbbOp
    def __init__(self, Tin, p, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class diskpbb(SpectralModel):
    def __init__(self, Tin=None, p=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([diskpbbComponent(Tin, p, norm, name, grad_method, eps)])


class diskpnOp(XspecNumericGradOp):
    modname = 'diskpn'
    optype = 'add'

class diskpnComponent(SpectralComponent):
    _comp_name = 'diskpn'
    _config = {'T_max': (1.0, 0.0001, 200.0, False, False), 'R_in': (6.0, 6.0, 1000.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = diskpnOp
    def __init__(self, T_max, R_in, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class diskpn(SpectralModel):
    def __init__(self, T_max=None, R_in=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([diskpnComponent(T_max, R_in, norm, name, grad_method, eps)])


class dustOp(XspecNumericGradOp):
    modname = 'dust'
    optype = 'mul'

class dustComponent(SpectralComponent):
    _comp_name = 'dust'
    _config = {'Frac': (0.066, 0.0, 1.0, True, False), 'Halosz': (2.0, 0.0, 100000.0, True, False)}
    _op_class = dustOp
    def __init__(self, Frac, Halosz, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class dust(SpectralModel):
    def __init__(self, Frac=None, Halosz=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([dustComponent(Frac, Halosz, name, grad_method, eps)])


class edgeOp(XspecNumericGradOp):
    modname = 'edge'
    optype = 'mul'

class edgeComponent(SpectralComponent):
    _comp_name = 'edge'
    _config = {'edgeE': (7.0, 0.0, 100.0, False, False), 'MaxTau': (1.0, 0.0, 10.0, False, False)}
    _op_class = edgeOp
    def __init__(self, edgeE, MaxTau, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class edge(SpectralModel):
    def __init__(self, edgeE=None, MaxTau=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([edgeComponent(edgeE, MaxTau, name, grad_method, eps)])


class eplogparOp(XspecNumericGradOp):
    modname = 'eplogpar'
    optype = 'add'

class eplogparComponent(SpectralComponent):
    _comp_name = 'eplogpar'
    _config = {'Ep': (0.1, 1e-10, 10000.0, False, False), 'beta': (0.2, -4.0, 4.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = eplogparOp
    def __init__(self, Ep, beta, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class eplogpar(SpectralModel):
    def __init__(self, Ep=None, beta=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([eplogparComponent(Ep, beta, norm, name, grad_method, eps)])


class eqpairOp(XspecNumericGradOp):
    modname = 'eqpair'
    optype = 'add'

class eqpairComponent(SpectralComponent):
    _comp_name = 'eqpair'
    _config = {'l_hovl_s': (1.0, 1e-06, 1000000.0, False, False), 'l_bb': (100.0, 0.0, 10000.0, False, False), 'kT_bb': (200.0, 1.0, 400000.0, True, False), 'l_ntol_h': (0.5, 0.0, 0.9999, False, False), 'tau_p': (0.1, 0.0001, 10.0, True, False), 'radius': (10000000.0, 100000.0, 1e+16, True, False), 'g_min': (1.3, 1.2, 1000.0, True, False), 'g_max': (1000.0, 5.0, 10000.0, True, False), 'G_inj': (2.0, 0.0, 5.0, True, False), 'pairinj': (0.0, 0.0, 1.0, True, False), 'cosIncl': (0.5, 0.05, 0.95, True, False), 'Refl': (1.0, 0.0, 2.0, True, False), 'Fe_abund': (1.0, 0.1, 10.0, True, False), 'Ab_met': (1.0, 0.1, 10.0, True, False), 'T_disk': (1000000.0, 10000.0, 1000000.0, True, False), 'xi': (0.0, 0.0, 5000.0, False, False), 'Beta': (-10.0, -10.0, 10.0, True, False), 'Rin': (10.0, 6.001, 10000.0, True, False), 'Rout': (1000.0, 0.0, 1000000.0, True, False), 'redshift': (0.0, 0.0, 4.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = eqpairOp
    def __init__(self, l_hovl_s, l_bb, kT_bb, l_ntol_h, tau_p, radius, g_min, g_max, G_inj, pairinj, cosIncl, Refl, Fe_abund, Ab_met, T_disk, xi, Beta, Rin, Rout, redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class eqpair(SpectralModel):
    def __init__(self, l_hovl_s=None, l_bb=None, kT_bb=None, l_ntol_h=None, tau_p=None, radius=None, g_min=None, g_max=None, G_inj=None, pairinj=None, cosIncl=None, Refl=None, Fe_abund=None, Ab_met=None, T_disk=None, xi=None, Beta=None, Rin=None, Rout=None, redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([eqpairComponent(l_hovl_s, l_bb, kT_bb, l_ntol_h, tau_p, radius, g_min, g_max, G_inj, pairinj, cosIncl, Refl, Fe_abund, Ab_met, T_disk, xi, Beta, Rin, Rout, redshift, norm, name, grad_method, eps)])


class eqthermOp(XspecNumericGradOp):
    modname = 'eqtherm'
    optype = 'add'

class eqthermComponent(SpectralComponent):
    _comp_name = 'eqtherm'
    _config = {'l_hovl_s': (1.0, 1e-06, 1000000.0, False, False), 'l_bb': (100.0, 0.0, 10000.0, False, False), 'kT_bb': (200.0, 1.0, 400000.0, True, False), 'l_ntol_h': (0.5, 0.0, 0.9999, False, False), 'tau_p': (0.1, 0.0001, 10.0, True, False), 'radius': (10000000.0, 100000.0, 1e+16, True, False), 'g_min': (1.3, 1.2, 1000.0, True, False), 'g_max': (1000.0, 5.0, 10000.0, True, False), 'G_inj': (2.0, 0.0, 5.0, True, False), 'pairinj': (0.0, 0.0, 1.0, True, False), 'cosIncl': (0.5, 0.05, 0.95, True, False), 'Refl': (1.0, 0.0, 2.0, True, False), 'Fe_abund': (1.0, 0.1, 10.0, True, False), 'Ab_met': (1.0, 0.1, 10.0, True, False), 'T_disk': (1000000.0, 10000.0, 1000000.0, True, False), 'xi': (0.0, 0.0, 5000.0, False, False), 'Beta': (-10.0, -10.0, 10.0, True, False), 'Rin': (10.0, 6.001, 10000.0, True, False), 'Rout': (1000.0, 0.0, 1000000.0, True, False), 'redshift': (0.0, 0.0, 4.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = eqthermOp
    def __init__(self, l_hovl_s, l_bb, kT_bb, l_ntol_h, tau_p, radius, g_min, g_max, G_inj, pairinj, cosIncl, Refl, Fe_abund, Ab_met, T_disk, xi, Beta, Rin, Rout, redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class eqtherm(SpectralModel):
    def __init__(self, l_hovl_s=None, l_bb=None, kT_bb=None, l_ntol_h=None, tau_p=None, radius=None, g_min=None, g_max=None, G_inj=None, pairinj=None, cosIncl=None, Refl=None, Fe_abund=None, Ab_met=None, T_disk=None, xi=None, Beta=None, Rin=None, Rout=None, redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([eqthermComponent(l_hovl_s, l_bb, kT_bb, l_ntol_h, tau_p, radius, g_min, g_max, G_inj, pairinj, cosIncl, Refl, Fe_abund, Ab_met, T_disk, xi, Beta, Rin, Rout, redshift, norm, name, grad_method, eps)])


class equilOp(XspecNumericGradOp):
    modname = 'equil'
    optype = 'add'

class equilComponent(SpectralComponent):
    _comp_name = 'equil'
    _config = {'kT': (1.0, 0.0808, 79.9, False, False), 'Abundanc': (1.0, 0.0, 10000.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = equilOp
    def __init__(self, kT, Abundanc, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class equil(SpectralModel):
    def __init__(self, kT=None, Abundanc=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([equilComponent(kT, Abundanc, Redshift, norm, name, grad_method, eps)])


class expabsOp(XspecNumericGradOp):
    modname = 'expabs'
    optype = 'mul'

class expabsComponent(SpectralComponent):
    _comp_name = 'expabs'
    _config = {'LowECut': (2.0, 0.0, 200.0, False, False)}
    _op_class = expabsOp
    def __init__(self, LowECut, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class expabs(SpectralModel):
    def __init__(self, LowECut=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([expabsComponent(LowECut, name, grad_method, eps)])


class expdecOp(XspecNumericGradOp):
    modname = 'expdec'
    optype = 'add'

class expdecComponent(SpectralComponent):
    _comp_name = 'expdec'
    _config = {'factor': (1.0, 0.0, 100.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = expdecOp
    def __init__(self, factor, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class expdec(SpectralModel):
    def __init__(self, factor=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([expdecComponent(factor, norm, name, grad_method, eps)])


class expfacOp(XspecNumericGradOp):
    modname = 'expfac'
    optype = 'mul'

class expfacComponent(SpectralComponent):
    _comp_name = 'expfac'
    _config = {'Ampl': (1.0, 0.0, 1000000.0, False, False), 'Factor': (1.0, 0.0, 1000000.0, False, False), 'StartE': (0.5, 0.0, 1000000.0, True, False)}
    _op_class = expfacOp
    def __init__(self, Ampl, Factor, StartE, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class expfac(SpectralModel):
    def __init__(self, Ampl=None, Factor=None, StartE=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([expfacComponent(Ampl, Factor, StartE, name, grad_method, eps)])


class ezdiskbbOp(XspecNumericGradOp):
    modname = 'ezdiskbb'
    optype = 'add'

class ezdiskbbComponent(SpectralComponent):
    _comp_name = 'ezdiskbb'
    _config = {'T_max': (1.0, 0.01, 100.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = ezdiskbbOp
    def __init__(self, T_max, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class ezdiskbb(SpectralModel):
    def __init__(self, T_max=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([ezdiskbbComponent(T_max, norm, name, grad_method, eps)])


class gabsOp(XspecNumericGradOp):
    modname = 'gabs'
    optype = 'mul'

class gabsComponent(SpectralComponent):
    _comp_name = 'gabs'
    _config = {'LineE': (1.0, 0.0, 1000000.0, False, False), 'Sigma': (0.01, 0.0, 20.0, False, False), 'Strength': (1.0, 0.0, 1000000.0, False, False)}
    _op_class = gabsOp
    def __init__(self, LineE, Sigma, Strength, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class gabs(SpectralModel):
    def __init__(self, LineE=None, Sigma=None, Strength=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([gabsComponent(LineE, Sigma, Strength, name, grad_method, eps)])


class gademOp(XspecNumericGradOp):
    modname = 'gadem'
    optype = 'add'

class gademComponent(SpectralComponent):
    _comp_name = 'gadem'
    _config = {'Tmean': (4.0, 0.01, 20.0, True, False), 'Tsigma': (0.1, 0.01, 100.0, False, False), 'nH': (1.0, 1e-06, 1e+20, True, False), 'abundanc': (1.0, 0.0, 10.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'switch': (2, None, None, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = gademOp
    def __init__(self, Tmean, Tsigma, nH, abundanc, Redshift, switch, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class gadem(SpectralModel):
    def __init__(self, Tmean=None, Tsigma=None, nH=None, abundanc=None, Redshift=None, switch=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([gademComponent(Tmean, Tsigma, nH, abundanc, Redshift, switch, norm, name, grad_method, eps)])


class gaussianOp(XspecNumericGradOp):
    modname = 'gaussian'
    optype = 'add'

class gaussianComponent(SpectralComponent):
    _comp_name = 'gaussian'
    _config = {'LineE': (6.5, 0.0, 1000000.0, False, False), 'Sigma': (0.1, 0.0, 20.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = gaussianOp
    def __init__(self, LineE, Sigma, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class gaussian(SpectralModel):
    def __init__(self, LineE=None, Sigma=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([gaussianComponent(LineE, Sigma, norm, name, grad_method, eps)])


class gneiOp(XspecNumericGradOp):
    modname = 'gnei'
    optype = 'add'

class gneiComponent(SpectralComponent):
    _comp_name = 'gnei'
    _config = {'kT': (1.0, 0.0808, 79.9, False, False), 'Abundanc': (1.0, 0.0, 10000.0, True, False), 'Tau': (100000000000.0, 100000000.0, 50000000000000.0, False, False), 'meankT': (1.0, 0.0808, 79.9, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = gneiOp
    def __init__(self, kT, Abundanc, Tau, meankT, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class gnei(SpectralModel):
    def __init__(self, kT=None, Abundanc=None, Tau=None, meankT=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([gneiComponent(kT, Abundanc, Tau, meankT, Redshift, norm, name, grad_method, eps)])


class gradOp(XspecNumericGradOp):
    modname = 'grad'
    optype = 'add'

class gradComponent(SpectralComponent):
    _comp_name = 'grad'
    _config = {'D': (10.0, 0.0, 10000.0, True, False), 'i': (0.0, 0.0, 90.0, True, False), 'Mass': (1.0, 0.0, 100.0, False, False), 'Mdot': (1.0, 0.0, 100.0, False, False), 'TclovTef': (1.7, 1.0, 10.0, True, False), 'refflag': (1.0, -1.0, 1.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = gradOp
    def __init__(self, D, i, Mass, Mdot, TclovTef, refflag, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class grad(SpectralModel):
    def __init__(self, D=None, i=None, Mass=None, Mdot=None, TclovTef=None, refflag=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([gradComponent(D, i, Mass, Mdot, TclovTef, refflag, norm, name, grad_method, eps)])


class grbcompOp(XspecNumericGradOp):
    modname = 'grbcomp'
    optype = 'add'

class grbcompComponent(SpectralComponent):
    _comp_name = 'grbcomp'
    _config = {'kTs': (1.0, 0.0, 20.0, False, False), 'gamma': (3.0, 0.0, 10.0, False, False), 'kTe': (100.0, 0.2, 2000.0, False, False), 'tau': (5.0, 0.0, 200.0, False, False), 'beta': (0.2, 0.0, 1.0, False, False), 'fbflag': (0.0, 0.0, 1.0, True, False), 'log_A': (5.0, -8.0, 8.0, True, False), 'z': (0.0, 0.0, 10.0, True, False), 'a_boost': (5.0, 0.0, 30.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = grbcompOp
    def __init__(self, kTs, gamma, kTe, tau, beta, fbflag, log_A, z, a_boost, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class grbcomp(SpectralModel):
    def __init__(self, kTs=None, gamma=None, kTe=None, tau=None, beta=None, fbflag=None, log_A=None, z=None, a_boost=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([grbcompComponent(kTs, gamma, kTe, tau, beta, fbflag, log_A, z, a_boost, norm, name, grad_method, eps)])


class grbjetOp(XspecNumericGradOp):
    modname = 'grbjet'
    optype = 'add'

class grbjetComponent(SpectralComponent):
    _comp_name = 'grbjet'
    _config = {'thobs': (5.0, 0.0, 30.0, True, False), 'thjet': (10.0, 2.0, 20.0, True, False), 'gamma': (200.0, 1.0, 500.0, False, False), 'r12': (1.0, 0.1, 100.0, True, False), 'p1': (0.0, -2.0, 1.0, False, False), 'p2': (1.5, 1.1, 10.0, False, False), 'E0': (1.0, 0.1, 1000.0, False, False), 'delta': (0.2, 0.01, 1.5, True, False), 'index_pl': (0.8, 0.0, 1.5, True, False), 'ecut': (20.0, 0.1, 1000.0, True, False), 'ktbb': (1.0, 0.1, 1000.0, True, False), 'model': (1, None, None, True, False), 'redshift': (2.0, 0.001, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = grbjetOp
    def __init__(self, thobs, thjet, gamma, r12, p1, p2, E0, delta, index_pl, ecut, ktbb, model, redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class grbjet(SpectralModel):
    def __init__(self, thobs=None, thjet=None, gamma=None, r12=None, p1=None, p2=None, E0=None, delta=None, index_pl=None, ecut=None, ktbb=None, model=None, redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([grbjetComponent(thobs, thjet, gamma, r12, p1, p2, E0, delta, index_pl, ecut, ktbb, model, redshift, norm, name, grad_method, eps)])


class grbmOp(XspecNumericGradOp):
    modname = 'grbm'
    optype = 'add'

class grbmComponent(SpectralComponent):
    _comp_name = 'grbm'
    _config = {'alpha': (-1.0, -10.0, 5.0, False, False), 'beta': (-2.0, -10.0, 10.0, False, False), 'tem': (300.0, 10.0, 10000.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = grbmOp
    def __init__(self, alpha, beta, tem, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class grbm(SpectralModel):
    def __init__(self, alpha=None, beta=None, tem=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([grbmComponent(alpha, beta, tem, norm, name, grad_method, eps)])


class gsmoothOp(XspecNumericGradOp):
    modname = 'gsmooth'
    optype = 'con'

class gsmoothComponent(SpectralComponent):
    _comp_name = 'gsmooth'
    _config = {'Sig_6keV': (1.0, 0.0, 20.0, False, False), 'Index': (0.0, -1.0, 1.0, True, False)}
    _op_class = gsmoothOp
    def __init__(self, Sig_6keV, Index, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class gsmooth(SpectralModel):
    def __init__(self, Sig_6keV=None, Index=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([gsmoothComponent(Sig_6keV, Index, name, grad_method, eps)])


class hatmOp(XspecNumericGradOp):
    modname = 'hatm'
    optype = 'add'

class hatmComponent(SpectralComponent):
    _comp_name = 'hatm'
    _config = {'T': (3.0, 0.5, 10.0, False, False), 'NSmass': (1.4, 0.6, 2.8, False, False), 'NSrad': (10.0, 5.0, 23.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = hatmOp
    def __init__(self, T, NSmass, NSrad, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class hatm(SpectralModel):
    def __init__(self, T=None, NSmass=None, NSrad=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([hatmComponent(T, NSmass, NSrad, norm, name, grad_method, eps)])


class heilinOp(XspecNumericGradOp):
    modname = 'heilin'
    optype = 'mul'

class heilinComponent(SpectralComponent):
    _comp_name = 'heilin'
    _config = {'nHeI': (1e-05, 0.0, 1000000.0, False, False), 'b': (10.0, 1.0, 1000000.0, False, False), 'z': (0.0, -0.001, 100000.0, False, False)}
    _op_class = heilinOp
    def __init__(self, nHeI, b, z, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class heilin(SpectralModel):
    def __init__(self, nHeI=None, b=None, z=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([heilinComponent(nHeI, b, z, name, grad_method, eps)])


class highecutOp(XspecNumericGradOp):
    modname = 'highecut'
    optype = 'mul'

class highecutComponent(SpectralComponent):
    _comp_name = 'highecut'
    _config = {'cutoffE': (10.0, 0.0001, 1000000.0, False, False), 'foldE': (15.0, 0.0001, 1000000.0, False, False)}
    _op_class = highecutOp
    def __init__(self, cutoffE, foldE, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class highecut(SpectralModel):
    def __init__(self, cutoffE=None, foldE=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([highecutComponent(cutoffE, foldE, name, grad_method, eps)])


class hreflOp(XspecNumericGradOp):
    modname = 'hrefl'
    optype = 'mul'

class hreflComponent(SpectralComponent):
    _comp_name = 'hrefl'
    _config = {'thetamin': (0.0, 0.0, 90.0, True, False), 'thetamax': (90.0, 0.0, 90.0, True, False), 'thetaobs': (60.0, 0.0, 90.0, False, False), 'Feabun': (1.0, 0.0, 200.0, True, False), 'FeKedge': (7.11, 7.0, 10.0, True, False), 'Escfrac': (1.0, 0.0, 1000.0, False, False), 'covfac': (1.0, 0.0, 1000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False)}
    _op_class = hreflOp
    def __init__(self, thetamin, thetamax, thetaobs, Feabun, FeKedge, Escfrac, covfac, Redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class hrefl(SpectralModel):
    def __init__(self, thetamin=None, thetamax=None, thetaobs=None, Feabun=None, FeKedge=None, Escfrac=None, covfac=None, Redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([hreflComponent(thetamin, thetamax, thetaobs, Feabun, FeKedge, Escfrac, covfac, Redshift, name, grad_method, eps)])


class ireflectOp(XspecNumericGradOp):
    modname = 'ireflect'
    optype = 'con'

class ireflectComponent(SpectralComponent):
    _comp_name = 'ireflect'
    _config = {'rel_refl': (0.0, -1.0, 1000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'abund': (1.0, 0.0, 1000000.0, True, False), 'Fe_abund': (1.0, 0.0, 1000000.0, True, False), 'cosIncl': (0.45, 0.05, 0.95, True, False), 'T_disk': (30000.0, 10000.0, 1000000.0, True, False), 'xi': (1.0, 0.0, 5000.0, False, False)}
    _op_class = ireflectOp
    def __init__(self, rel_refl, Redshift, abund, Fe_abund, cosIncl, T_disk, xi, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class ireflect(SpectralModel):
    def __init__(self, rel_refl=None, Redshift=None, abund=None, Fe_abund=None, cosIncl=None, T_disk=None, xi=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([ireflectComponent(rel_refl, Redshift, abund, Fe_abund, cosIncl, T_disk, xi, name, grad_method, eps)])


class ismabsOp(XspecNumericGradOp):
    modname = 'ismabs'
    optype = 'mul'

class ismabsComponent(SpectralComponent):
    _comp_name = 'ismabs'
    _config = {'H': (0.1, 0.0, 1000000.0, False, False), 'He_II': (0.0, 0.0, 1000000.0, True, False), 'C_I': (33.1, 0.0, 1000000.0, False, False), 'C_II': (0.0, 0.0, 1000000.0, True, False), 'C_III': (0.0, 0.0, 1000000.0, True, False), 'N_I': (8.32, 0.0, 1000000.0, False, False), 'N_II': (0.0, 0.0, 1000000.0, True, False), 'N_III': (0.0, 0.0, 1000000.0, True, False), 'O_I': (67.6, 0.0, 1000000.0, False, False), 'O_II': (0.0, 0.0, 1000000.0, True, False), 'O_III': (0.0, 0.0, 1000000.0, True, False), 'Ne_I': (12.0, 0.0, 1000000.0, False, False), 'Ne_II': (0.0, 0.0, 1000000.0, True, False), 'Ne_III': (0.0, 0.0, 1000000.0, True, False), 'Mg_I': (3.8, 0.0, 1000000.0, False, False), 'Mg_II': (0.0, 0.0, 1000000.0, True, False), 'Mg_III': (0.0, 0.0, 1000000.0, True, False), 'Si_I': (3.35, 0.0, 1000000.0, False, False), 'Si_II': (0.0, 0.0, 1000000.0, True, False), 'Si_III': (0.0, 0.0, 1000000.0, True, False), 'S_I': (2.14, 0.0, 1000000.0, False, False), 'S_II': (0.0, 0.0, 1000000.0, True, False), 'S_III': (0.0, 0.0, 1000000.0, True, False), 'Ar_I': (0.25, 0.0, 1000000.0, False, False), 'Ar_II': (0.0, 0.0, 1000000.0, True, False), 'Ar_III': (0.0, 0.0, 1000000.0, True, False), 'Ca_I': (0.22, 0.0, 1000000.0, False, False), 'Ca_II': (0.0, 0.0, 1000000.0, True, False), 'Ca_III': (0.0, 0.0, 1000000.0, True, False), 'Fe': (3.16, 0.0, 1000000.0, False, False), 'redshift': (0.0, -1.0, 10.0, True, False)}
    _op_class = ismabsOp
    def __init__(self, H, He_II, C_I, C_II, C_III, N_I, N_II, N_III, O_I, O_II, O_III, Ne_I, Ne_II, Ne_III, Mg_I, Mg_II, Mg_III, Si_I, Si_II, Si_III, S_I, S_II, S_III, Ar_I, Ar_II, Ar_III, Ca_I, Ca_II, Ca_III, Fe, redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class ismabs(SpectralModel):
    def __init__(self, H=None, He_II=None, C_I=None, C_II=None, C_III=None, N_I=None, N_II=None, N_III=None, O_I=None, O_II=None, O_III=None, Ne_I=None, Ne_II=None, Ne_III=None, Mg_I=None, Mg_II=None, Mg_III=None, Si_I=None, Si_II=None, Si_III=None, S_I=None, S_II=None, S_III=None, Ar_I=None, Ar_II=None, Ar_III=None, Ca_I=None, Ca_II=None, Ca_III=None, Fe=None, redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([ismabsComponent(H, He_II, C_I, C_II, C_III, N_I, N_II, N_III, O_I, O_II, O_III, Ne_I, Ne_II, Ne_III, Mg_I, Mg_II, Mg_III, Si_I, Si_II, Si_III, S_I, S_II, S_III, Ar_I, Ar_II, Ar_III, Ca_I, Ca_II, Ca_III, Fe, redshift, name, grad_method, eps)])


class ismdustOp(XspecNumericGradOp):
    modname = 'ismdust'
    optype = 'mul'

class ismdustComponent(SpectralComponent):
    _comp_name = 'ismdust'
    _config = {'msil': (1.0, 0.0, 100000.0, False, False), 'mgra': (1.0, 0.0, 100000.0, False, False), 'redshift': (0.0, -1.0, 10.0, True, False)}
    _op_class = ismdustOp
    def __init__(self, msil, mgra, redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class ismdust(SpectralModel):
    def __init__(self, msil=None, mgra=None, redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([ismdustComponent(msil, mgra, redshift, name, grad_method, eps)])


class jetOp(XspecNumericGradOp):
    modname = 'jet'
    optype = 'add'

class jetComponent(SpectralComponent):
    _comp_name = 'jet'
    _config = {'mass': (1000000000.0, 1.0, 10000000000.0, True, False), 'Dco': (3350.6, 1.0, 100000000.0, True, False), 'log_mdot': (-1.0, -5.0, 2.0, False, False), 'thetaobs': (3.0, 0.0, 90.0, True, False), 'BulkG': (13.0, 1.0, 100.0, True, False), 'phi': (0.1, 0.01, 100.0, True, False), 'zdiss': (1275.0, 10.0, 10000.0, True, False), 'B': (2.6, 0.01, 15.0, True, False), 'logPrel': (43.3, 40.0, 48.0, True, False), 'gmin_inj': (1.0, 1.0, 1000.0, True, False), 'gbreak': (300.0, 10.0, 10000.0, True, False), 'gmax': (3000.0, 1000.0, 1000000.0, True, False), 's1': (1.0, -1.0, 1.0, True, False), 's2': (2.7, 1.0, 5.0, True, False), 'z': (0.0, 0.0, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = jetOp
    def __init__(self, mass, Dco, log_mdot, thetaobs, BulkG, phi, zdiss, B, logPrel, gmin_inj, gbreak, gmax, s1, s2, z, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class jet(SpectralModel):
    def __init__(self, mass=None, Dco=None, log_mdot=None, thetaobs=None, BulkG=None, phi=None, zdiss=None, B=None, logPrel=None, gmin_inj=None, gbreak=None, gmax=None, s1=None, s2=None, z=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([jetComponent(mass, Dco, log_mdot, thetaobs, BulkG, phi, zdiss, B, logPrel, gmin_inj, gbreak, gmax, s1, s2, z, norm, name, grad_method, eps)])


class kdblurOp(XspecNumericGradOp):
    modname = 'kdblur'
    optype = 'con'

class kdblurComponent(SpectralComponent):
    _comp_name = 'kdblur'
    _config = {'Index': (3.0, -10.0, 10.0, True, False), 'Rin_G': (4.5, 1.235, 400.0, True, False), 'Rout_G': (100.0, 1.235, 400.0, True, False), 'Incl': (30.0, 0.0, 90.0, False, False)}
    _op_class = kdblurOp
    def __init__(self, Index, Rin_G, Rout_G, Incl, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class kdblur(SpectralModel):
    def __init__(self, Index=None, Rin_G=None, Rout_G=None, Incl=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([kdblurComponent(Index, Rin_G, Rout_G, Incl, name, grad_method, eps)])


class kdblur2Op(XspecNumericGradOp):
    modname = 'kdblur2'
    optype = 'con'

class kdblur2Component(SpectralComponent):
    _comp_name = 'kdblur2'
    _config = {'Index': (3.0, -10.0, 10.0, True, False), 'Rin_G': (4.5, 1.235, 400.0, True, False), 'Rout_G': (100.0, 1.235, 400.0, True, False), 'Incl': (30.0, 0.0, 90.0, False, False), 'Rbreak': (20.0, 1.235, 400.0, True, False), 'Index1': (3.0, -10.0, 10.0, True, False)}
    _op_class = kdblur2Op
    def __init__(self, Index, Rin_G, Rout_G, Incl, Rbreak, Index1, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class kdblur2(SpectralModel):
    def __init__(self, Index=None, Rin_G=None, Rout_G=None, Incl=None, Rbreak=None, Index1=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([kdblur2Component(Index, Rin_G, Rout_G, Incl, Rbreak, Index1, name, grad_method, eps)])


class kerrbbOp(XspecNumericGradOp):
    modname = 'kerrbb'
    optype = 'add'

class kerrbbComponent(SpectralComponent):
    _comp_name = 'kerrbb'
    _config = {'eta': (0.0, 0.0, 1.0, True, False), 'a': (0.0, -1.0, 0.9999, False, False), 'i': (30.0, 0.0, 85.0, True, False), 'Mbh': (1.0, 0.0, 100.0, False, False), 'Mdd': (1.0, 0.0, 1000.0, False, False), 'Dbh': (10.0, 0.0, 10000.0, True, False), 'hd': (1.7, 1.0, 10.0, True, False), 'rflag': (1, None, None, True, False), 'lflag': (0, None, None, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = kerrbbOp
    def __init__(self, eta, a, i, Mbh, Mdd, Dbh, hd, rflag, lflag, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class kerrbb(SpectralModel):
    def __init__(self, eta=None, a=None, i=None, Mbh=None, Mdd=None, Dbh=None, hd=None, rflag=None, lflag=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([kerrbbComponent(eta, a, i, Mbh, Mdd, Dbh, hd, rflag, lflag, norm, name, grad_method, eps)])


class kerrconvOp(XspecNumericGradOp):
    modname = 'kerrconv'
    optype = 'con'

class kerrconvComponent(SpectralComponent):
    _comp_name = 'kerrconv'
    _config = {'Index1': (3.0, -10.0, 10.0, True, False), 'Index2': (3.0, -10.0, 10.0, True, False), 'r_br_g': (6.0, 1.0, 400.0, True, False), 'a': (0.998, 0.0, 0.998, False, False), 'Incl': (30.0, 0.0, 90.0, True, False), 'Rin_ms': (1.0, 1.0, 400.0, True, False), 'Rout_ms': (400.0, 1.0, 400.0, True, False)}
    _op_class = kerrconvOp
    def __init__(self, Index1, Index2, r_br_g, a, Incl, Rin_ms, Rout_ms, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class kerrconv(SpectralModel):
    def __init__(self, Index1=None, Index2=None, r_br_g=None, a=None, Incl=None, Rin_ms=None, Rout_ms=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([kerrconvComponent(Index1, Index2, r_br_g, a, Incl, Rin_ms, Rout_ms, name, grad_method, eps)])


class kerrdOp(XspecNumericGradOp):
    modname = 'kerrd'
    optype = 'add'

class kerrdComponent(SpectralComponent):
    _comp_name = 'kerrd'
    _config = {'distance': (1.0, 0.01, 1000.0, True, False), 'TcoloTeff': (1.5, 1.0, 2.0, True, False), 'M': (1.0, 0.1, 100.0, False, False), 'Mdot': (1.0, 0.01, 100.0, False, False), 'Incl': (30.0, 0.0, 90.0, True, False), 'Rin': (1.235, 1.235, 100.0, True, False), 'Rout': (100000.0, 10000.0, 100000000.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = kerrdOp
    def __init__(self, distance, TcoloTeff, M, Mdot, Incl, Rin, Rout, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class kerrd(SpectralModel):
    def __init__(self, distance=None, TcoloTeff=None, M=None, Mdot=None, Incl=None, Rin=None, Rout=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([kerrdComponent(distance, TcoloTeff, M, Mdot, Incl, Rin, Rout, norm, name, grad_method, eps)])


class kerrdiskOp(XspecNumericGradOp):
    modname = 'kerrdisk'
    optype = 'add'

class kerrdiskComponent(SpectralComponent):
    _comp_name = 'kerrdisk'
    _config = {'lineE': (6.4, 0.1, 100.0, True, False), 'Index1': (3.0, -10.0, 10.0, True, False), 'Index2': (3.0, -10.0, 10.0, True, False), 'r_br_g': (6.0, 1.0, 400.0, True, False), 'a': (0.998, 0.01, 0.998, False, False), 'Incl': (30.0, 0.0, 90.0, True, False), 'Rin_ms': (1.0, 1.0, 400.0, True, False), 'Rout_ms': (400.0, 1.0, 400.0, True, False), 'z': (0.0, 0.0, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = kerrdiskOp
    def __init__(self, lineE, Index1, Index2, r_br_g, a, Incl, Rin_ms, Rout_ms, z, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class kerrdisk(SpectralModel):
    def __init__(self, lineE=None, Index1=None, Index2=None, r_br_g=None, a=None, Incl=None, Rin_ms=None, Rout_ms=None, z=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([kerrdiskComponent(lineE, Index1, Index2, r_br_g, a, Incl, Rin_ms, Rout_ms, z, norm, name, grad_method, eps)])


class kyconvOp(XspecNumericGradOp):
    modname = 'kyconv'
    optype = 'con'

class kyconvComponent(SpectralComponent):
    _comp_name = 'kyconv'
    _config = {'a': (0.9982, 0.0, 1.0, False, False), 'theta_o': (30.0, 0.0, 89.0, False, False), 'rin': (1.0, 1.0, 1000.0, True, False), 'ms': (1.0, 0.0, 1.0, True, False), 'rout': (400.0, 1.0, 1000.0, True, False), 'alpha': (3.0, -20.0, 20.0, True, False), 'beta': (3.0, -20.0, 20.0, True, False), 'rb': (400.0, 1.0, 1000.0, True, False), 'zshift': (0.0, -0.999, 10.0, True, False), 'limb': (0.0, 0.0, 2.0, True, False), 'ne_loc': (100.0, 3.0, 5000.0, True, False), 'normal': (1.0, -1.0, 100.0, True, False)}
    _op_class = kyconvOp
    def __init__(self, a, theta_o, rin, ms, rout, alpha, beta, rb, zshift, limb, ne_loc, normal, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class kyconv(SpectralModel):
    def __init__(self, a=None, theta_o=None, rin=None, ms=None, rout=None, alpha=None, beta=None, rb=None, zshift=None, limb=None, ne_loc=None, normal=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([kyconvComponent(a, theta_o, rin, ms, rout, alpha, beta, rb, zshift, limb, ne_loc, normal, name, grad_method, eps)])


class kyrlineOp(XspecNumericGradOp):
    modname = 'kyrline'
    optype = 'add'

class kyrlineComponent(SpectralComponent):
    _comp_name = 'kyrline'
    _config = {'a': (0.9982, 0.0, 1.0, False, False), 'theta_o': (30.0, 0.0, 89.0, False, False), 'rin': (1.0, 1.0, 1000.0, True, False), 'ms': (1.0, 0.0, 1.0, True, False), 'rout': (400.0, 1.0, 1000.0, True, False), 'Erest': (6.4, 1.0, 99.0, True, False), 'alpha': (3.0, -20.0, 20.0, True, False), 'beta': (3.0, -20.0, 20.0, True, False), 'rb': (400.0, 1.0, 1000.0, True, False), 'zshift': (0.0, -0.999, 10.0, True, False), 'limb': (1.0, 0.0, 2.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = kyrlineOp
    def __init__(self, a, theta_o, rin, ms, rout, Erest, alpha, beta, rb, zshift, limb, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class kyrline(SpectralModel):
    def __init__(self, a=None, theta_o=None, rin=None, ms=None, rout=None, Erest=None, alpha=None, beta=None, rb=None, zshift=None, limb=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([kyrlineComponent(a, theta_o, rin, ms, rout, Erest, alpha, beta, rb, zshift, limb, norm, name, grad_method, eps)])


class laorOp(XspecNumericGradOp):
    modname = 'laor'
    optype = 'add'

class laorComponent(SpectralComponent):
    _comp_name = 'laor'
    _config = {'lineE': (6.4, 0.0, 100.0, False, False), 'Index': (3.0, -10.0, 10.0, True, False), 'Rin_G': (1.235, 1.235, 400.0, True, False), 'Rout_G': (400.0, 1.235, 400.0, True, False), 'Incl': (30.0, 0.0, 90.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = laorOp
    def __init__(self, lineE, Index, Rin_G, Rout_G, Incl, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class laor(SpectralModel):
    def __init__(self, lineE=None, Index=None, Rin_G=None, Rout_G=None, Incl=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([laorComponent(lineE, Index, Rin_G, Rout_G, Incl, norm, name, grad_method, eps)])


class laor2Op(XspecNumericGradOp):
    modname = 'laor2'
    optype = 'add'

class laor2Component(SpectralComponent):
    _comp_name = 'laor2'
    _config = {'lineE': (6.4, 0.0, 100.0, False, False), 'Index': (3.0, -10.0, 10.0, True, False), 'Rin_G': (1.235, 1.235, 400.0, True, False), 'Rout_G': (400.0, 1.235, 400.0, True, False), 'Incl': (30.0, 0.0, 90.0, True, False), 'Rbreak': (20.0, 1.235, 400.0, True, False), 'Index1': (3.0, -10.0, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = laor2Op
    def __init__(self, lineE, Index, Rin_G, Rout_G, Incl, Rbreak, Index1, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class laor2(SpectralModel):
    def __init__(self, lineE=None, Index=None, Rin_G=None, Rout_G=None, Incl=None, Rbreak=None, Index1=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([laor2Component(lineE, Index, Rin_G, Rout_G, Incl, Rbreak, Index1, norm, name, grad_method, eps)])


class log10conOp(XspecNumericGradOp):
    modname = 'log10con'
    optype = 'mul'

class log10conComponent(SpectralComponent):
    _comp_name = 'log10con'
    _config = {'log10fac': (0.0, -20.0, 20.0, False, False)}
    _op_class = log10conOp
    def __init__(self, log10fac, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class log10con(SpectralModel):
    def __init__(self, log10fac=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([log10conComponent(log10fac, name, grad_method, eps)])


class logconstOp(XspecNumericGradOp):
    modname = 'logconst'
    optype = 'mul'

class logconstComponent(SpectralComponent):
    _comp_name = 'logconst'
    _config = {'logfact': (0.0, -20.0, 20.0, False, False)}
    _op_class = logconstOp
    def __init__(self, logfact, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class logconst(SpectralModel):
    def __init__(self, logfact=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([logconstComponent(logfact, name, grad_method, eps)])


class logparOp(XspecNumericGradOp):
    modname = 'logpar'
    optype = 'add'

class logparComponent(SpectralComponent):
    _comp_name = 'logpar'
    _config = {'alpha': (1.5, 0.0, 4.0, False, False), 'beta': (0.2, -4.0, 4.0, False, False), 'pivotE': (1.0, None, None, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = logparOp
    def __init__(self, alpha, beta, pivotE, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class logpar(SpectralModel):
    def __init__(self, alpha=None, beta=None, pivotE=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([logparComponent(alpha, beta, pivotE, norm, name, grad_method, eps)])


class lorentzOp(XspecNumericGradOp):
    modname = 'lorentz'
    optype = 'add'

class lorentzComponent(SpectralComponent):
    _comp_name = 'lorentz'
    _config = {'LineE': (6.5, 0.0, 1000000.0, False, False), 'Width': (0.1, 0.0, 20.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = lorentzOp
    def __init__(self, LineE, Width, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class lorentz(SpectralModel):
    def __init__(self, LineE=None, Width=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([lorentzComponent(LineE, Width, norm, name, grad_method, eps)])


class lsmoothOp(XspecNumericGradOp):
    modname = 'lsmooth'
    optype = 'con'

class lsmoothComponent(SpectralComponent):
    _comp_name = 'lsmooth'
    _config = {'Sig_6keV': (1.0, 0.0, 20.0, False, False), 'Index': (0.0, -1.0, 1.0, True, False)}
    _op_class = lsmoothOp
    def __init__(self, Sig_6keV, Index, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class lsmooth(SpectralModel):
    def __init__(self, Sig_6keV=None, Index=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([lsmoothComponent(Sig_6keV, Index, name, grad_method, eps)])


class lymanOp(XspecNumericGradOp):
    modname = 'lyman'
    optype = 'mul'

class lymanComponent(SpectralComponent):
    _comp_name = 'lyman'
    _config = {'n': (1e-05, 0.0, 1000000.0, False, False), 'b': (10.0, 1.0, 1000000.0, False, False), 'z': (0.0, -0.001, 100000.0, False, False), 'ZA': (1.0, 1.0, 2.0, False, False)}
    _op_class = lymanOp
    def __init__(self, n, b, z, ZA, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class lyman(SpectralModel):
    def __init__(self, n=None, b=None, z=None, ZA=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([lymanComponent(n, b, z, ZA, name, grad_method, eps)])


class mekaOp(XspecNumericGradOp):
    modname = 'meka'
    optype = 'add'

class mekaComponent(SpectralComponent):
    _comp_name = 'meka'
    _config = {'kT': (1.0, 0.001, 100.0, False, False), 'nH': (1.0, 1e-06, 1e+20, True, False), 'Abundanc': (1.0, 0.0, 1000.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = mekaOp
    def __init__(self, kT, nH, Abundanc, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class meka(SpectralModel):
    def __init__(self, kT=None, nH=None, Abundanc=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([mekaComponent(kT, nH, Abundanc, Redshift, norm, name, grad_method, eps)])


class mekalOp(XspecNumericGradOp):
    modname = 'mekal'
    optype = 'add'

class mekalComponent(SpectralComponent):
    _comp_name = 'mekal'
    _config = {'kT': (1.0, 0.0808, 79.9, False, False), 'nH': (1.0, 1e-06, 1e+20, True, False), 'Abundanc': (1.0, 0.0, 1000.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'switch': (1, 0.0, 1.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = mekalOp
    def __init__(self, kT, nH, Abundanc, Redshift, switch, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class mekal(SpectralModel):
    def __init__(self, kT=None, nH=None, Abundanc=None, Redshift=None, switch=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([mekalComponent(kT, nH, Abundanc, Redshift, switch, norm, name, grad_method, eps)])


class mkcflowOp(XspecNumericGradOp):
    modname = 'mkcflow'
    optype = 'add'

class mkcflowComponent(SpectralComponent):
    _comp_name = 'mkcflow'
    _config = {'lowT': (0.1, 0.0808, 79.9, False, False), 'highT': (4.0, 0.0808, 79.9, False, False), 'Abundanc': (1.0, 0.0, 5.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'switch': (1, 0.0, 1.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = mkcflowOp
    def __init__(self, lowT, highT, Abundanc, Redshift, switch, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class mkcflow(SpectralModel):
    def __init__(self, lowT=None, highT=None, Abundanc=None, Redshift=None, switch=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([mkcflowComponent(lowT, highT, Abundanc, Redshift, switch, norm, name, grad_method, eps)])


class neiOp(XspecNumericGradOp):
    modname = 'nei'
    optype = 'add'

class neiComponent(SpectralComponent):
    _comp_name = 'nei'
    _config = {'kT': (1.0, 0.0808, 79.9, False, False), 'Abundanc': (1.0, 0.0, 10000.0, True, False), 'Tau': (100000000000.0, 100000000.0, 50000000000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = neiOp
    def __init__(self, kT, Abundanc, Tau, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class nei(SpectralModel):
    def __init__(self, kT=None, Abundanc=None, Tau=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([neiComponent(kT, Abundanc, Tau, Redshift, norm, name, grad_method, eps)])


class nlapecOp(XspecNumericGradOp):
    modname = 'nlapec'
    optype = 'add'

class nlapecComponent(SpectralComponent):
    _comp_name = 'nlapec'
    _config = {'kT': (1.0, 0.008, 64.0, False, False), 'Abundanc': (1.0, 0.0, 5.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = nlapecOp
    def __init__(self, kT, Abundanc, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class nlapec(SpectralModel):
    def __init__(self, kT=None, Abundanc=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([nlapecComponent(kT, Abundanc, Redshift, norm, name, grad_method, eps)])


class notchOp(XspecNumericGradOp):
    modname = 'notch'
    optype = 'mul'

class notchComponent(SpectralComponent):
    _comp_name = 'notch'
    _config = {'LineE': (3.5, 0.0, 20.0, False, False), 'Width': (1.0, 0.0, 20.0, False, False), 'CvrFract': (1.0, 0.0, 1.0, True, False)}
    _op_class = notchOp
    def __init__(self, LineE, Width, CvrFract, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class notch(SpectralModel):
    def __init__(self, LineE=None, Width=None, CvrFract=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([notchComponent(LineE, Width, CvrFract, name, grad_method, eps)])


class npshockOp(XspecNumericGradOp):
    modname = 'npshock'
    optype = 'add'

class npshockComponent(SpectralComponent):
    _comp_name = 'npshock'
    _config = {'kT_a': (1.0, 0.0808, 79.9, False, False), 'kT_b': (0.5, 0.01, 79.9, False, False), 'Abundanc': (1.0, 0.0, 10000.0, True, False), 'Tau_l': (0.0, 0.0, 50000000000000.0, True, False), 'Tau_u': (100000000000.0, 100000000.0, 50000000000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = npshockOp
    def __init__(self, kT_a, kT_b, Abundanc, Tau_l, Tau_u, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class npshock(SpectralModel):
    def __init__(self, kT_a=None, kT_b=None, Abundanc=None, Tau_l=None, Tau_u=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([npshockComponent(kT_a, kT_b, Abundanc, Tau_l, Tau_u, Redshift, norm, name, grad_method, eps)])


class nsaOp(XspecNumericGradOp):
    modname = 'nsa'
    optype = 'add'

class nsaComponent(SpectralComponent):
    _comp_name = 'nsa'
    _config = {'LogT_eff': (6.0, 5.0, 7.0, False, False), 'M_ns': (1.4, 0.5, 2.5, False, False), 'R_ns': (10.0, 5.0, 20.0, False, False), 'MagField': (0.0, 0.0, 50000000000000.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = nsaOp
    def __init__(self, LogT_eff, M_ns, R_ns, MagField, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class nsa(SpectralModel):
    def __init__(self, LogT_eff=None, M_ns=None, R_ns=None, MagField=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([nsaComponent(LogT_eff, M_ns, R_ns, MagField, norm, name, grad_method, eps)])


class nsagravOp(XspecNumericGradOp):
    modname = 'nsagrav'
    optype = 'add'

class nsagravComponent(SpectralComponent):
    _comp_name = 'nsagrav'
    _config = {'LogT_eff': (6.0, 5.5, 6.5, False, False), 'NSmass': (1.4, 0.3, 2.5, False, False), 'NSrad': (10.0, 6.0, 20.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = nsagravOp
    def __init__(self, LogT_eff, NSmass, NSrad, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class nsagrav(SpectralModel):
    def __init__(self, LogT_eff=None, NSmass=None, NSrad=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([nsagravComponent(LogT_eff, NSmass, NSrad, norm, name, grad_method, eps)])


class nsatmosOp(XspecNumericGradOp):
    modname = 'nsatmos'
    optype = 'add'

class nsatmosComponent(SpectralComponent):
    _comp_name = 'nsatmos'
    _config = {'LogT_eff': (6.0, 5.0, 6.5, False, False), 'M_ns': (1.4, 0.5, 3.0, False, False), 'R_ns': (10.0, 5.0, 30.0, False, False), 'dist': (10.0, 0.1, 100.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = nsatmosOp
    def __init__(self, LogT_eff, M_ns, R_ns, dist, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class nsatmos(SpectralModel):
    def __init__(self, LogT_eff=None, M_ns=None, R_ns=None, dist=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([nsatmosComponent(LogT_eff, M_ns, R_ns, dist, norm, name, grad_method, eps)])


class nsmaxOp(XspecNumericGradOp):
    modname = 'nsmax'
    optype = 'add'

class nsmaxComponent(SpectralComponent):
    _comp_name = 'nsmax'
    _config = {'logTeff': (6.0, 5.5, 6.8, False, False), 'redshift': (0.1, 1e-05, 2.0, False, False), 'specfile': (1200, None, None, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = nsmaxOp
    def __init__(self, logTeff, redshift, specfile, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class nsmax(SpectralModel):
    def __init__(self, logTeff=None, redshift=None, specfile=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([nsmaxComponent(logTeff, redshift, specfile, norm, name, grad_method, eps)])


class nsmaxgOp(XspecNumericGradOp):
    modname = 'nsmaxg'
    optype = 'add'

class nsmaxgComponent(SpectralComponent):
    _comp_name = 'nsmaxg'
    _config = {'logTeff': (6.0, 5.5, 6.9, False, False), 'M_ns': (1.4, 0.5, 3.0, False, False), 'R_ns': (10.0, 5.0, 30.0, False, False), 'dist': (1.0, 0.01, 100.0, False, False), 'specfile': (1200, None, None, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = nsmaxgOp
    def __init__(self, logTeff, M_ns, R_ns, dist, specfile, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class nsmaxg(SpectralModel):
    def __init__(self, logTeff=None, M_ns=None, R_ns=None, dist=None, specfile=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([nsmaxgComponent(logTeff, M_ns, R_ns, dist, specfile, norm, name, grad_method, eps)])


class nsxOp(XspecNumericGradOp):
    modname = 'nsx'
    optype = 'add'

class nsxComponent(SpectralComponent):
    _comp_name = 'nsx'
    _config = {'logTeff': (6.0, 5.5, 6.7, False, False), 'M_ns': (1.4, 0.5, 3.0, False, False), 'R_ns': (10.0, 5.0, 30.0, False, False), 'dist': (1.0, 0.01, 100.0, False, False), 'specfile': (6, None, None, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = nsxOp
    def __init__(self, logTeff, M_ns, R_ns, dist, specfile, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class nsx(SpectralModel):
    def __init__(self, logTeff=None, M_ns=None, R_ns=None, dist=None, specfile=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([nsxComponent(logTeff, M_ns, R_ns, dist, specfile, norm, name, grad_method, eps)])


class nteeaOp(XspecNumericGradOp):
    modname = 'nteea'
    optype = 'add'

class nteeaComponent(SpectralComponent):
    _comp_name = 'nteea'
    _config = {'l_nth': (100.0, 0.0, 10000.0, False, False), 'l_bb': (100.0, 0.0, 10000.0, False, False), 'f_refl': (0.0, 0.0, 4.0, False, False), 'kT_bb': (10.0, 1.0, 100.0, True, False), 'g_max': (1000.0, 5.0, 10000.0, True, False), 'l_th': (0.0, 0.0, 10000.0, True, False), 'tau_p': (0.0, 0.0, 10.0, True, False), 'G_inj': (0.0, 0.0, 5.0, True, False), 'g_min': (1.3, 1.0, 1000.0, True, False), 'g_0': (1.3, 1.0, 5.0, True, False), 'radius': (10000000000000.0, 100000.0, 1e+16, True, False), 'pair_esc': (0.0, 0.0, 1.0, True, False), 'cosIncl': (0.45, 0.05, 0.95, False, False), 'Fe_abund': (1.0, 0.1, 10.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = nteeaOp
    def __init__(self, l_nth, l_bb, f_refl, kT_bb, g_max, l_th, tau_p, G_inj, g_min, g_0, radius, pair_esc, cosIncl, Fe_abund, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class nteea(SpectralModel):
    def __init__(self, l_nth=None, l_bb=None, f_refl=None, kT_bb=None, g_max=None, l_th=None, tau_p=None, G_inj=None, g_min=None, g_0=None, radius=None, pair_esc=None, cosIncl=None, Fe_abund=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([nteeaComponent(l_nth, l_bb, f_refl, kT_bb, g_max, l_th, tau_p, G_inj, g_min, g_0, radius, pair_esc, cosIncl, Fe_abund, Redshift, norm, name, grad_method, eps)])


class nthCompOp(XspecNumericGradOp):
    modname = 'nthComp'
    optype = 'add'

class nthCompComponent(SpectralComponent):
    _comp_name = 'nthComp'
    _config = {'Gamma': (1.7, 1.001, 10.0, False, False), 'kT_e': (100.0, 1.0, 1000.0, False, False), 'kT_bb': (0.1, 0.001, 10.0, True, False), 'inp_type': (0.0, 0.0, 1.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = nthCompOp
    def __init__(self, Gamma, kT_e, kT_bb, inp_type, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class nthComp(SpectralModel):
    def __init__(self, Gamma=None, kT_e=None, kT_bb=None, inp_type=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([nthCompComponent(Gamma, kT_e, kT_bb, inp_type, Redshift, norm, name, grad_method, eps)])


class olivineabsOp(XspecNumericGradOp):
    modname = 'olivineabs'
    optype = 'mul'

class olivineabsComponent(SpectralComponent):
    _comp_name = 'olivineabs'
    _config = {'moliv': (1.0, 0.0, 100000.0, False, False), 'redshift': (0.0, -1.0, 10.0, True, False)}
    _op_class = olivineabsOp
    def __init__(self, moliv, redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class olivineabs(SpectralModel):
    def __init__(self, moliv=None, redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([olivineabsComponent(moliv, redshift, name, grad_method, eps)])


class optxagnOp(XspecNumericGradOp):
    modname = 'optxagn'
    optype = 'add'

class optxagnComponent(SpectralComponent):
    _comp_name = 'optxagn'
    _config = {'mass': (10000000.0, 1.0, 1000000000.0, True, False), 'dist': (100.0, 0.01, 1000000000.0, True, False), 'logLoLEdd': (-1.0, -10.0, 2.0, False, False), 'astar': (0.0, 0.0, 0.998, True, False), 'rcor': (10.0, 1.0, 100.0, False, False), 'logrout': (5.0, 3.0, 7.0, True, False), 'kT_e': (0.2, 0.01, 10.0, False, False), 'tau': (10.0, 0.1, 100.0, False, False), 'Gamma': (2.1, 0.5, 10.0, False, False), 'fpl': (0.0001, 0.0, 0.1, False, False), 'fcol': (2.4, 1.0, 5.0, True, False), 'tscat': (100000.0, 10000.0, 100000.0, True, False), 'Redshift': (0.0, 0.0, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = optxagnOp
    def __init__(self, mass, dist, logLoLEdd, astar, rcor, logrout, kT_e, tau, Gamma, fpl, fcol, tscat, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class optxagn(SpectralModel):
    def __init__(self, mass=None, dist=None, logLoLEdd=None, astar=None, rcor=None, logrout=None, kT_e=None, tau=None, Gamma=None, fpl=None, fcol=None, tscat=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([optxagnComponent(mass, dist, logLoLEdd, astar, rcor, logrout, kT_e, tau, Gamma, fpl, fcol, tscat, Redshift, norm, name, grad_method, eps)])


class optxagnfOp(XspecNumericGradOp):
    modname = 'optxagnf'
    optype = 'add'

class optxagnfComponent(SpectralComponent):
    _comp_name = 'optxagnf'
    _config = {'mass': (10000000.0, 1.0, 1000000000.0, True, False), 'dist': (100.0, 0.01, 1000000000.0, True, False), 'logLoLEdd': (-1.0, -10.0, 2.0, False, False), 'astar': (0.0, 0.0, 0.998, True, False), 'rcor': (10.0, 1.0, 100.0, False, False), 'logrout': (5.0, 3.0, 7.0, True, False), 'kT_e': (0.2, 0.01, 10.0, False, False), 'tau': (10.0, 0.1, 100.0, False, False), 'Gamma': (2.1, 1.05, 10.0, False, False), 'fpl': (0.0001, 0.0, 1.0, False, False), 'Redshift': (0.0, 0.0, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = optxagnfOp
    def __init__(self, mass, dist, logLoLEdd, astar, rcor, logrout, kT_e, tau, Gamma, fpl, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class optxagnf(SpectralModel):
    def __init__(self, mass=None, dist=None, logLoLEdd=None, astar=None, rcor=None, logrout=None, kT_e=None, tau=None, Gamma=None, fpl=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([optxagnfComponent(mass, dist, logLoLEdd, astar, rcor, logrout, kT_e, tau, Gamma, fpl, Redshift, norm, name, grad_method, eps)])


class partcovOp(XspecNumericGradOp):
    modname = 'partcov'
    optype = 'con'

class partcovComponent(SpectralComponent):
    _comp_name = 'partcov'
    _config = {'CvrFract': (0.5, 0.0, 1.0, False, False)}
    _op_class = partcovOp
    def __init__(self, CvrFract, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class partcov(SpectralModel):
    def __init__(self, CvrFract=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([partcovComponent(CvrFract, name, grad_method, eps)])


class pcfabsOp(XspecNumericGradOp):
    modname = 'pcfabs'
    optype = 'mul'

class pcfabsComponent(SpectralComponent):
    _comp_name = 'pcfabs'
    _config = {'nH': (1.0, 0.0, 1000000.0, False, False), 'CvrFract': (0.5, 0.0, 1.0, False, False)}
    _op_class = pcfabsOp
    def __init__(self, nH, CvrFract, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class pcfabs(SpectralModel):
    def __init__(self, nH=None, CvrFract=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([pcfabsComponent(nH, CvrFract, name, grad_method, eps)])


class pegpwrlwOp(XspecNumericGradOp):
    modname = 'pegpwrlw'
    optype = 'add'

class pegpwrlwComponent(SpectralComponent):
    _comp_name = 'pegpwrlw'
    _config = {'PhoIndex': (1.0, -3.0, 10.0, False, False), 'eMin': (2.0, -100.0, 10000000000.0, True, False), 'eMax': (10.0, -100.0, 10000000000.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = pegpwrlwOp
    def __init__(self, PhoIndex, eMin, eMax, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class pegpwrlw(SpectralModel):
    def __init__(self, PhoIndex=None, eMin=None, eMax=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([pegpwrlwComponent(PhoIndex, eMin, eMax, norm, name, grad_method, eps)])


class pexmonOp(XspecNumericGradOp):
    modname = 'pexmon'
    optype = 'add'

class pexmonComponent(SpectralComponent):
    _comp_name = 'pexmon'
    _config = {'PhoIndex': (2.0, 1.1, 2.5, False, False), 'foldE': (1000.0, 1.0, 1000000.0, True, False), 'rel_refl': (-1.0, -1000000.0, 1000000.0, True, False), 'redshift': (0.0, 0.0, 4.0, True, False), 'abund': (1.0, 0.0, 1000000.0, True, False), 'Fe_abund': (1.0, 0.0, 100.0, True, False), 'Incl': (60.0, 0.0, 85.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = pexmonOp
    def __init__(self, PhoIndex, foldE, rel_refl, redshift, abund, Fe_abund, Incl, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class pexmon(SpectralModel):
    def __init__(self, PhoIndex=None, foldE=None, rel_refl=None, redshift=None, abund=None, Fe_abund=None, Incl=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([pexmonComponent(PhoIndex, foldE, rel_refl, redshift, abund, Fe_abund, Incl, norm, name, grad_method, eps)])


class pexravOp(XspecNumericGradOp):
    modname = 'pexrav'
    optype = 'add'

class pexravComponent(SpectralComponent):
    _comp_name = 'pexrav'
    _config = {'PhoIndex': (2.0, -10.0, 10.0, False, False), 'foldE': (100.0, 1.0, 1000000.0, False, False), 'rel_refl': (0.0, 0.0, 1000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'abund': (1.0, 0.0, 1000000.0, True, False), 'Fe_abund': (1.0, 0.0, 1000000.0, True, False), 'cosIncl': (0.45, 0.05, 0.95, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = pexravOp
    def __init__(self, PhoIndex, foldE, rel_refl, Redshift, abund, Fe_abund, cosIncl, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class pexrav(SpectralModel):
    def __init__(self, PhoIndex=None, foldE=None, rel_refl=None, Redshift=None, abund=None, Fe_abund=None, cosIncl=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([pexravComponent(PhoIndex, foldE, rel_refl, Redshift, abund, Fe_abund, cosIncl, norm, name, grad_method, eps)])


class pexrivOp(XspecNumericGradOp):
    modname = 'pexriv'
    optype = 'add'

class pexrivComponent(SpectralComponent):
    _comp_name = 'pexriv'
    _config = {'PhoIndex': (2.0, -10.0, 10.0, False, False), 'foldE': (100.0, 1.0, 1000000.0, False, False), 'rel_refl': (0.0, 0.0, 1000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'abund': (1.0, 0.0, 1000000.0, True, False), 'Fe_abund': (1.0, 0.0, 1000000.0, True, False), 'cosIncl': (0.45, 0.05, 0.95, True, False), 'T_disk': (30000.0, 10000.0, 1000000.0, True, False), 'xi': (1.0, 0.0, 5000.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = pexrivOp
    def __init__(self, PhoIndex, foldE, rel_refl, Redshift, abund, Fe_abund, cosIncl, T_disk, xi, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class pexriv(SpectralModel):
    def __init__(self, PhoIndex=None, foldE=None, rel_refl=None, Redshift=None, abund=None, Fe_abund=None, cosIncl=None, T_disk=None, xi=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([pexrivComponent(PhoIndex, foldE, rel_refl, Redshift, abund, Fe_abund, cosIncl, T_disk, xi, norm, name, grad_method, eps)])


class phabsOp(XspecNumericGradOp):
    modname = 'phabs'
    optype = 'mul'

class phabsComponent(SpectralComponent):
    _comp_name = 'phabs'
    _config = {'nH': (1.0, 0.0, 1000000.0, False, False)}
    _op_class = phabsOp
    def __init__(self, nH, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class phabs(SpectralModel):
    def __init__(self, nH=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([phabsComponent(nH, name, grad_method, eps)])


class plabsOp(XspecNumericGradOp):
    modname = 'plabs'
    optype = 'mul'

class plabsComponent(SpectralComponent):
    _comp_name = 'plabs'
    _config = {'index': (2.0, 0.0, 5.0, False, False), 'coef': (1.0, 0.0, 100.0, False, False)}
    _op_class = plabsOp
    def __init__(self, index, coef, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class plabs(SpectralModel):
    def __init__(self, index=None, coef=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([plabsComponent(index, coef, name, grad_method, eps)])


class plcabsOp(XspecNumericGradOp):
    modname = 'plcabs'
    optype = 'add'

class plcabsComponent(SpectralComponent):
    _comp_name = 'plcabs'
    _config = {'nH': (1.0, 0.0, 1000000.0, False, False), 'nmax': (1.0, None, None, True, False), 'FeAbun': (1.0, 0.0, 10.0, True, False), 'FeKedge': (7.11, 7.0, 10.0, True, False), 'PhoIndex': (2.0, -3.0, 10.0, False, False), 'HighECut': (95.0, 0.01, 200.0, True, False), 'foldE': (100.0, 1.0, 1000000.0, True, False), 'acrit': (1.0, 0.0, 1.0, True, False), 'FAST': (0.0, None, None, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = plcabsOp
    def __init__(self, nH, nmax, FeAbun, FeKedge, PhoIndex, HighECut, foldE, acrit, FAST, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class plcabs(SpectralModel):
    def __init__(self, nH=None, nmax=None, FeAbun=None, FeKedge=None, PhoIndex=None, HighECut=None, foldE=None, acrit=None, FAST=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([plcabsComponent(nH, nmax, FeAbun, FeKedge, PhoIndex, HighECut, foldE, acrit, FAST, Redshift, norm, name, grad_method, eps)])


class polconstOp(XspecNumericGradOp):
    modname = 'polconst'
    optype = 'mul'

class polconstComponent(SpectralComponent):
    _comp_name = 'polconst'
    _config = {'A': (1.0, 0.0, 1.0, False, False), 'psi': (45.0, -90.0, 90.0, False, False)}
    _op_class = polconstOp
    def __init__(self, A, psi, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class polconst(SpectralModel):
    def __init__(self, A=None, psi=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([polconstComponent(A, psi, name, grad_method, eps)])


class pollinOp(XspecNumericGradOp):
    modname = 'pollin'
    optype = 'mul'

class pollinComponent(SpectralComponent):
    _comp_name = 'pollin'
    _config = {'A1': (1.0, 0.0, 1.0, False, False), 'Aslope': (0.0, -5.0, 5.0, False, False), 'psi1': (45.0, -90.0, 90.0, False, False), 'psislope': (0.0, -5.0, 5.0, False, False)}
    _op_class = pollinOp
    def __init__(self, A1, Aslope, psi1, psislope, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class pollin(SpectralModel):
    def __init__(self, A1=None, Aslope=None, psi1=None, psislope=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([pollinComponent(A1, Aslope, psi1, psislope, name, grad_method, eps)])


class polpowOp(XspecNumericGradOp):
    modname = 'polpow'
    optype = 'mul'

class polpowComponent(SpectralComponent):
    _comp_name = 'polpow'
    _config = {'Anorm': (1.0, 0.0, 1.0, False, False), 'Aindex': (0.0, -5.0, 5.0, False, False), 'psinorm': (45.0, -90.0, 90.0, False, False), 'psiindex': (0.0, -5.0, 5.0, False, False)}
    _op_class = polpowOp
    def __init__(self, Anorm, Aindex, psinorm, psiindex, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class polpow(SpectralModel):
    def __init__(self, Anorm=None, Aindex=None, psinorm=None, psiindex=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([polpowComponent(Anorm, Aindex, psinorm, psiindex, name, grad_method, eps)])


class posmOp(XspecNumericGradOp):
    modname = 'posm'
    optype = 'add'

class posmComponent(SpectralComponent):
    _comp_name = 'posm'
    _config = {'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = posmOp
    def __init__(self, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class posm(SpectralModel):
    def __init__(self, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([posmComponent(norm, name, grad_method, eps)])


class powerlawOp(XspecNumericGradOp):
    modname = 'powerlaw'
    optype = 'add'

class powerlawComponent(SpectralComponent):
    _comp_name = 'powerlaw'
    _config = {'PhoIndex': (1.0, -3.0, 10.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = powerlawOp
    def __init__(self, PhoIndex, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class powerlaw(SpectralModel):
    def __init__(self, PhoIndex=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([powerlawComponent(PhoIndex, norm, name, grad_method, eps)])


class pshockOp(XspecNumericGradOp):
    modname = 'pshock'
    optype = 'add'

class pshockComponent(SpectralComponent):
    _comp_name = 'pshock'
    _config = {'kT': (1.0, 0.0808, 79.9, False, False), 'Abundanc': (1.0, 0.0, 10000.0, True, False), 'Tau_l': (0.0, 0.0, 50000000000000.0, True, False), 'Tau_u': (100000000000.0, 100000000.0, 50000000000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = pshockOp
    def __init__(self, kT, Abundanc, Tau_l, Tau_u, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class pshock(SpectralModel):
    def __init__(self, kT=None, Abundanc=None, Tau_l=None, Tau_u=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([pshockComponent(kT, Abundanc, Tau_l, Tau_u, Redshift, norm, name, grad_method, eps)])


class pwabOp(XspecNumericGradOp):
    modname = 'pwab'
    optype = 'mul'

class pwabComponent(SpectralComponent):
    _comp_name = 'pwab'
    _config = {'nHmin': (1.0, 1e-07, 1000000.0, False, False), 'nHmax': (2.0, 1e-07, 1000000.0, False, False), 'beta': (1.0, -10.0, 20.0, True, False)}
    _op_class = pwabOp
    def __init__(self, nHmin, nHmax, beta, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class pwab(SpectralModel):
    def __init__(self, nHmin=None, nHmax=None, beta=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([pwabComponent(nHmin, nHmax, beta, name, grad_method, eps)])


class qsosedOp(XspecNumericGradOp):
    modname = 'qsosed'
    optype = 'add'

class qsosedComponent(SpectralComponent):
    _comp_name = 'qsosed'
    _config = {'mass': (10000000.0, 100000.0, 10000000000.0, True, False), 'dist': (100.0, 0.01, 1000000000.0, True, False), 'logmdot': (-1.0, -1.65, 0.39, False, False), 'astar': (0.0, -1.0, 0.998, True, False), 'cosi': (0.5, 0.05, 1.0, True, False), 'redshift': (0.0, 0.0, 5.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = qsosedOp
    def __init__(self, mass, dist, logmdot, astar, cosi, redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class qsosed(SpectralModel):
    def __init__(self, mass=None, dist=None, logmdot=None, astar=None, cosi=None, redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([qsosedComponent(mass, dist, logmdot, astar, cosi, redshift, norm, name, grad_method, eps)])


class raymondOp(XspecNumericGradOp):
    modname = 'raymond'
    optype = 'add'

class raymondComponent(SpectralComponent):
    _comp_name = 'raymond'
    _config = {'kT': (1.0, 0.008, 64.0, False, False), 'Abundanc': (1.0, 0.0, 5.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = raymondOp
    def __init__(self, kT, Abundanc, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class raymond(SpectralModel):
    def __init__(self, kT=None, Abundanc=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([raymondComponent(kT, Abundanc, Redshift, norm, name, grad_method, eps)])


class rdblurOp(XspecNumericGradOp):
    modname = 'rdblur'
    optype = 'con'

class rdblurComponent(SpectralComponent):
    _comp_name = 'rdblur'
    _config = {'Betor10': (-2.0, -10.0, 20.0, True, False), 'Rin_M': (10.0, 6.0, 10000.0, True, False), 'Rout_M': (1000.0, 0.0, 10000000.0, True, False), 'Incl': (30.0, 0.0, 90.0, False, False)}
    _op_class = rdblurOp
    def __init__(self, Betor10, Rin_M, Rout_M, Incl, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class rdblur(SpectralModel):
    def __init__(self, Betor10=None, Rin_M=None, Rout_M=None, Incl=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([rdblurComponent(Betor10, Rin_M, Rout_M, Incl, name, grad_method, eps)])


class reddenOp(XspecNumericGradOp):
    modname = 'redden'
    optype = 'mul'

class reddenComponent(SpectralComponent):
    _comp_name = 'redden'
    _config = {'E_BmV': (0.05, 0.0, 10.0, False, False)}
    _op_class = reddenOp
    def __init__(self, E_BmV, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class redden(SpectralModel):
    def __init__(self, E_BmV=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([reddenComponent(E_BmV, name, grad_method, eps)])


class redgeOp(XspecNumericGradOp):
    modname = 'redge'
    optype = 'add'

class redgeComponent(SpectralComponent):
    _comp_name = 'redge'
    _config = {'edge': (1.4, 0.001, 100.0, False, False), 'kT': (1.0, 0.001, 100.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = redgeOp
    def __init__(self, edge, kT, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class redge(SpectralModel):
    def __init__(self, edge=None, kT=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([redgeComponent(edge, kT, norm, name, grad_method, eps)])


class reflectOp(XspecNumericGradOp):
    modname = 'reflect'
    optype = 'con'

class reflectComponent(SpectralComponent):
    _comp_name = 'reflect'
    _config = {'rel_refl': (0.0, -1.0, 1000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'abund': (1.0, 0.0, 1000000.0, True, False), 'Fe_abund': (1.0, 0.0, 1000000.0, True, False), 'cosIncl': (0.45, 0.05, 0.95, True, False)}
    _op_class = reflectOp
    def __init__(self, rel_refl, Redshift, abund, Fe_abund, cosIncl, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class reflect(SpectralModel):
    def __init__(self, rel_refl=None, Redshift=None, abund=None, Fe_abund=None, cosIncl=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([reflectComponent(rel_refl, Redshift, abund, Fe_abund, cosIncl, name, grad_method, eps)])


class refschOp(XspecNumericGradOp):
    modname = 'refsch'
    optype = 'add'

class refschComponent(SpectralComponent):
    _comp_name = 'refsch'
    _config = {'PhoIndex': (2.0, -10.0, 10.0, False, False), 'foldE': (100.0, 1.0, 1000000.0, False, False), 'rel_refl': (0.0, 0.0, 2.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'abund': (1.0, 0.5, 10.0, True, False), 'Fe_abund': (1.0, 0.1, 10.0, True, False), 'Incl': (30.0, 19.0, 87.0, True, False), 'T_disk': (30000.0, 10000.0, 1000000.0, True, False), 'xi': (1.0, 0.0, 5000.0, False, False), 'Betor10': (-2.0, -10.0, 20.0, True, False), 'Rin': (10.0, 6.0, 10000.0, True, False), 'Rout': (1000.0, 0.0, 10000000.0, True, False), 'accuracy': (30.0, 30.0, 100000.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = refschOp
    def __init__(self, PhoIndex, foldE, rel_refl, Redshift, abund, Fe_abund, Incl, T_disk, xi, Betor10, Rin, Rout, accuracy, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class refsch(SpectralModel):
    def __init__(self, PhoIndex=None, foldE=None, rel_refl=None, Redshift=None, abund=None, Fe_abund=None, Incl=None, T_disk=None, xi=None, Betor10=None, Rin=None, Rout=None, accuracy=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([refschComponent(PhoIndex, foldE, rel_refl, Redshift, abund, Fe_abund, Incl, T_disk, xi, Betor10, Rin, Rout, accuracy, norm, name, grad_method, eps)])


class rfxconvOp(XspecNumericGradOp):
    modname = 'rfxconv'
    optype = 'con'

class rfxconvComponent(SpectralComponent):
    _comp_name = 'rfxconv'
    _config = {'rel_refl': (-1.0, -1.0, 1000000.0, False, False), 'redshift': (0.0, 0.0, 4.0, True, False), 'Fe_abund': (1.0, 0.5, 3.0, True, False), 'cosIncl': (0.5, 0.05, 0.95, True, False), 'log_xi': (1.0, 1.0, 6.0, False, False)}
    _op_class = rfxconvOp
    def __init__(self, rel_refl, redshift, Fe_abund, cosIncl, log_xi, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class rfxconv(SpectralModel):
    def __init__(self, rel_refl=None, redshift=None, Fe_abund=None, cosIncl=None, log_xi=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([rfxconvComponent(rel_refl, redshift, Fe_abund, cosIncl, log_xi, name, grad_method, eps)])


class rgsxsrcOp(XspecNumericGradOp):
    modname = 'rgsxsrc'
    optype = 'con'

class rgsxsrcComponent(SpectralComponent):
    _comp_name = 'rgsxsrc'
    _config = {'order': (-1.0, -3.0, -1.0, True, False)}
    _op_class = rgsxsrcOp
    def __init__(self, order, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class rgsxsrc(SpectralModel):
    def __init__(self, order=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([rgsxsrcComponent(order, name, grad_method, eps)])


class rneiOp(XspecNumericGradOp):
    modname = 'rnei'
    optype = 'add'

class rneiComponent(SpectralComponent):
    _comp_name = 'rnei'
    _config = {'kT': (0.5, 0.0808, 79.9, False, False), 'kT_init': (1.0, 0.0808, 79.9, False, False), 'Abundanc': (1.0, 0.0, 10000.0, True, False), 'Tau': (100000000000.0, 100000000.0, 50000000000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = rneiOp
    def __init__(self, kT, kT_init, Abundanc, Tau, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class rnei(SpectralModel):
    def __init__(self, kT=None, kT_init=None, Abundanc=None, Tau=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([rneiComponent(kT, kT_init, Abundanc, Tau, Redshift, norm, name, grad_method, eps)])


class sedovOp(XspecNumericGradOp):
    modname = 'sedov'
    optype = 'add'

class sedovComponent(SpectralComponent):
    _comp_name = 'sedov'
    _config = {'kT_a': (1.0, 0.0808, 79.9, False, False), 'kT_b': (0.5, 0.01, 79.9, False, False), 'Abundanc': (1.0, 0.0, 10000.0, True, False), 'Tau': (100000000000.0, 100000000.0, 50000000000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = sedovOp
    def __init__(self, kT_a, kT_b, Abundanc, Tau, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class sedov(SpectralModel):
    def __init__(self, kT_a=None, kT_b=None, Abundanc=None, Tau=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([sedovComponent(kT_a, kT_b, Abundanc, Tau, Redshift, norm, name, grad_method, eps)])


class simplOp(XspecNumericGradOp):
    modname = 'simpl'
    optype = 'con'

class simplComponent(SpectralComponent):
    _comp_name = 'simpl'
    _config = {'Gamma': (2.3, 1.0, 5.0, False, False), 'FracSctr': (0.05, 0.0, 1.0, False, False), 'UpScOnly': (1.0, 0.0, 100.0, True, False)}
    _op_class = simplOp
    def __init__(self, Gamma, FracSctr, UpScOnly, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class simpl(SpectralModel):
    def __init__(self, Gamma=None, FracSctr=None, UpScOnly=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([simplComponent(Gamma, FracSctr, UpScOnly, name, grad_method, eps)])


class sirfOp(XspecNumericGradOp):
    modname = 'sirf'
    optype = 'add'

class sirfComponent(SpectralComponent):
    _comp_name = 'sirf'
    _config = {'tin': (1.0, 0.01, 1000.0, False, False), 'rin': (0.01, 1e-06, 10.0, False, False), 'rout': (100.0, 0.1, 100000000.0, False, False), 'theta': (22.9, 0.0, 90.0, False, False), 'incl': (0.0, -90.0, 90.0, True, False), 'valpha': (-0.5, -1.5, 5.0, True, False), 'gamma': (1.333, 0.5, 10.0, True, False), 'mdot': (1000.0, 0.5, 10000000.0, True, False), 'irrad': (2.0, 0.0, 20.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = sirfOp
    def __init__(self, tin, rin, rout, theta, incl, valpha, gamma, mdot, irrad, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class sirf(SpectralModel):
    def __init__(self, tin=None, rin=None, rout=None, theta=None, incl=None, valpha=None, gamma=None, mdot=None, irrad=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([sirfComponent(tin, rin, rout, theta, incl, valpha, gamma, mdot, irrad, norm, name, grad_method, eps)])


class slimbhOp(XspecNumericGradOp):
    modname = 'slimbh'
    optype = 'add'

class slimbhComponent(SpectralComponent):
    _comp_name = 'slimbh'
    _config = {'M': (10.0, 0.0, 1000.0, True, False), 'a': (0.0, 0.0, 0.999, False, False), 'lumin': (0.5, 0.05, 1.0, False, False), 'alpha': (0.1, 0.005, 0.1, True, False), 'inc': (60.0, 0.0, 85.0, True, False), 'D': (10.0, 0.0, 10000.0, True, False), 'f_hard': (-1.0, -10.0, 10.0, True, False), 'lflag': (1, None, None, True, False), 'vflag': (1, None, None, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = slimbhOp
    def __init__(self, M, a, lumin, alpha, inc, D, f_hard, lflag, vflag, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class slimbh(SpectralModel):
    def __init__(self, M=None, a=None, lumin=None, alpha=None, inc=None, D=None, f_hard=None, lflag=None, vflag=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([slimbhComponent(M, a, lumin, alpha, inc, D, f_hard, lflag, vflag, norm, name, grad_method, eps)])


class smaugOp(XspecNumericGradOp):
    modname = 'smaug'
    optype = 'add'

class smaugComponent(SpectralComponent):
    _comp_name = 'smaug'
    _config = {'kT_cc': (1.0, 0.08, 100.0, False, False), 'kT_dt': (1.0, 0.0, 100.0, False, False), 'kT_ix': (0.0, 0.0, 10.0, True, False), 'kT_ir': (0.1, 0.0001, 1.0, True, False), 'kT_cx': (0.5, 0.0, 10.0, False, False), 'kT_cr': (0.1, 0.0001, 20.0, False, False), 'kT_tx': (0.0, 0.0, 10.0, True, False), 'kT_tr': (0.5, 0.0001, 3.0, True, False), 'nH_cc': (1.0, 1e-06, 3.0, True, False), 'nH_ff': (1.0, 0.0, 1.0, True, False), 'nH_cx': (0.5, 0.0, 10.0, False, False), 'nH_cr': (0.1, 0.0001, 2.0, False, False), 'nH_gx': (0.0, 0.0, 10.0, True, False), 'nH_gr': (0.002, 0.0001, 20.0, True, False), 'Ab_cc': (1.0, 0.0, 5.0, True, False), 'Ab_xx': (0.0, 0.0, 10.0, True, False), 'Ab_rr': (0.1, 0.0001, 1.0, True, False), 'redshift': (0.01, 0.0001, 10.0, True, False), 'meshpts': (10.0, 1.0, 10000.0, True, False), 'rcutoff': (2.0, 1.0, 3.0, True, False), 'mode': (1.0, 0.0, 2.0, True, False), 'itype': (2.0, 1.0, 4.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = smaugOp
    def __init__(self, kT_cc, kT_dt, kT_ix, kT_ir, kT_cx, kT_cr, kT_tx, kT_tr, nH_cc, nH_ff, nH_cx, nH_cr, nH_gx, nH_gr, Ab_cc, Ab_xx, Ab_rr, redshift, meshpts, rcutoff, mode, itype, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class smaug(SpectralModel):
    def __init__(self, kT_cc=None, kT_dt=None, kT_ix=None, kT_ir=None, kT_cx=None, kT_cr=None, kT_tx=None, kT_tr=None, nH_cc=None, nH_ff=None, nH_cx=None, nH_cr=None, nH_gx=None, nH_gr=None, Ab_cc=None, Ab_xx=None, Ab_rr=None, redshift=None, meshpts=None, rcutoff=None, mode=None, itype=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([smaugComponent(kT_cc, kT_dt, kT_ix, kT_ir, kT_cx, kT_cr, kT_tx, kT_tr, nH_cc, nH_ff, nH_cx, nH_cr, nH_gx, nH_gr, Ab_cc, Ab_xx, Ab_rr, redshift, meshpts, rcutoff, mode, itype, norm, name, grad_method, eps)])


class smedgeOp(XspecNumericGradOp):
    modname = 'smedge'
    optype = 'mul'

class smedgeComponent(SpectralComponent):
    _comp_name = 'smedge'
    _config = {'edgeE': (7.0, 0.1, 100.0, False, False), 'MaxTau': (1.0, 0.0, 10.0, False, False), 'index': (-2.67, -10.0, 10.0, True, False), 'width': (10.0, 0.01, 100.0, False, False)}
    _op_class = smedgeOp
    def __init__(self, edgeE, MaxTau, index, width, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class smedge(SpectralModel):
    def __init__(self, edgeE=None, MaxTau=None, index=None, width=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([smedgeComponent(edgeE, MaxTau, index, width, name, grad_method, eps)])


class snapecOp(XspecNumericGradOp):
    modname = 'snapec'
    optype = 'add'

class snapecComponent(SpectralComponent):
    _comp_name = 'snapec'
    _config = {'kT': (1.0, 0.008, 64.0, False, False), 'N_SNe': (1.0, 0.0, 1e+20, False, False), 'R': (1.0, 0.0, 1e+20, False, False), 'SNIModelIndex': (1.0, 0.0, 125.0, True, False), 'SNIIModelIndex': (1.0, 0.0, 125.0, True, False), 'redshift': (0.0, 0.0, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = snapecOp
    def __init__(self, kT, N_SNe, R, SNIModelIndex, SNIIModelIndex, redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class snapec(SpectralModel):
    def __init__(self, kT=None, N_SNe=None, R=None, SNIModelIndex=None, SNIIModelIndex=None, redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([snapecComponent(kT, N_SNe, R, SNIModelIndex, SNIIModelIndex, redshift, norm, name, grad_method, eps)])


class spexpcutOp(XspecNumericGradOp):
    modname = 'spexpcut'
    optype = 'mul'

class spexpcutComponent(SpectralComponent):
    _comp_name = 'spexpcut'
    _config = {'Ecut': (10.0, 0.0, 1000000.0, False, False), 'alpha': (1.0, -5.0, 5.0, False, False)}
    _op_class = spexpcutOp
    def __init__(self, Ecut, alpha, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class spexpcut(SpectralModel):
    def __init__(self, Ecut=None, alpha=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([spexpcutComponent(Ecut, alpha, name, grad_method, eps)])


class splineOp(XspecNumericGradOp):
    modname = 'spline'
    optype = 'mul'

class splineComponent(SpectralComponent):
    _comp_name = 'spline'
    _config = {'Estart': (0.1, 0.0, 100.0, False, False), 'Ystart': (1.0, -1000000.0, 1000000.0, False, False), 'Yend': (1.0, -1000000.0, 1000000.0, False, False), 'YPstart': (0.0, -1000000.0, 1000000.0, False, False), 'YPend': (0.0, -1000000.0, 1000000.0, False, False), 'Eend': (15.0, 0.0, 100.0, False, False)}
    _op_class = splineOp
    def __init__(self, Estart, Ystart, Yend, YPstart, YPend, Eend, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class spline(SpectralModel):
    def __init__(self, Estart=None, Ystart=None, Yend=None, YPstart=None, YPend=None, Eend=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([splineComponent(Estart, Ystart, Yend, YPstart, YPend, Eend, name, grad_method, eps)])


class srcutOp(XspecNumericGradOp):
    modname = 'srcut'
    optype = 'add'

class srcutComponent(SpectralComponent):
    _comp_name = 'srcut'
    _config = {'alpha': (0.5, 1e-05, 1.0, False, False), 'break_': (2.42e+17, 10000000000.0, 1e+25, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = srcutOp
    def __init__(self, alpha, break_, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class srcut(SpectralModel):
    def __init__(self, alpha=None, break_=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([srcutComponent(alpha, break_, norm, name, grad_method, eps)])


class srescOp(XspecNumericGradOp):
    modname = 'sresc'
    optype = 'add'

class srescComponent(SpectralComponent):
    _comp_name = 'sresc'
    _config = {'alpha': (0.5, 1e-05, 1.0, False, False), 'rolloff': (2.42e+17, 10000000000.0, 1e+25, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = srescOp
    def __init__(self, alpha, rolloff, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class sresc(SpectralModel):
    def __init__(self, alpha=None, rolloff=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([srescComponent(alpha, rolloff, norm, name, grad_method, eps)])


class ssaOp(XspecNumericGradOp):
    modname = 'ssa'
    optype = 'add'

class ssaComponent(SpectralComponent):
    _comp_name = 'ssa'
    _config = {'te': (0.1, 0.01, 0.5, False, False), 'y': (0.7, 0.0001, 1000.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = ssaOp
    def __init__(self, te, y, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class ssa(SpectralModel):
    def __init__(self, te=None, y=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([ssaComponent(te, y, norm, name, grad_method, eps)])


class stepOp(XspecNumericGradOp):
    modname = 'step'
    optype = 'add'

class stepComponent(SpectralComponent):
    _comp_name = 'step'
    _config = {'Energy': (6.5, 0.0, 100.0, False, False), 'Sigma': (0.1, 0.0, 20.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = stepOp
    def __init__(self, Energy, Sigma, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class step(SpectralModel):
    def __init__(self, Energy=None, Sigma=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([stepComponent(Energy, Sigma, norm, name, grad_method, eps)])


class swind1Op(XspecNumericGradOp):
    modname = 'swind1'
    optype = 'mul'

class swind1Component(SpectralComponent):
    _comp_name = 'swind1'
    _config = {'column': (6.0, 3.0, 50.0, False, False), 'log_xi': (2.5, 2.1, 4.1, False, False), 'sigma': (0.1, 0.0, 0.5, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False)}
    _op_class = swind1Op
    def __init__(self, column, log_xi, sigma, Redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class swind1(SpectralModel):
    def __init__(self, column=None, log_xi=None, sigma=None, Redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([swind1Component(column, log_xi, sigma, Redshift, name, grad_method, eps)])


class tapecOp(XspecNumericGradOp):
    modname = 'tapec'
    optype = 'add'

class tapecComponent(SpectralComponent):
    _comp_name = 'tapec'
    _config = {'kT': (1.0, 0.008, 64.0, False, False), 'kTi': (1.0, 0.008, 64.0, False, False), 'Abundanc': (1.0, 0.0, 5.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = tapecOp
    def __init__(self, kT, kTi, Abundanc, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class tapec(SpectralModel):
    def __init__(self, kT=None, kTi=None, Abundanc=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([tapecComponent(kT, kTi, Abundanc, Redshift, norm, name, grad_method, eps)])


class thcompOp(XspecNumericGradOp):
    modname = 'thcomp'
    optype = 'con'

class thcompComponent(SpectralComponent):
    _comp_name = 'thcomp'
    _config = {'Gamma_tau': (1.7, 1.001, 10.0, False, False), 'kT_e': (50.0, 0.5, 150.0, False, False), 'cov_frac': (1.0, 0.0, 1.0, False, False), 'z': (0.0, 0.0, 5.0, True, False)}
    _op_class = thcompOp
    def __init__(self, Gamma_tau, kT_e, cov_frac, z, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class thcomp(SpectralModel):
    def __init__(self, Gamma_tau=None, kT_e=None, cov_frac=None, z=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([thcompComponent(Gamma_tau, kT_e, cov_frac, z, name, grad_method, eps)])


class uvredOp(XspecNumericGradOp):
    modname = 'uvred'
    optype = 'mul'

class uvredComponent(SpectralComponent):
    _comp_name = 'uvred'
    _config = {'E_BmV': (0.05, 0.0, 10.0, False, False)}
    _op_class = uvredOp
    def __init__(self, E_BmV, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class uvred(SpectralModel):
    def __init__(self, E_BmV=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([uvredComponent(E_BmV, name, grad_method, eps)])


class vapecOp(XspecNumericGradOp):
    modname = 'vapec'
    optype = 'add'

class vapecComponent(SpectralComponent):
    _comp_name = 'vapec'
    _config = {'kT': (6.5, 0.0808, 68.447, False, False), 'He': (1.0, 0.0, 1000.0, True, False), 'C': (1.0, 0.0, 1000.0, True, False), 'N': (1.0, 0.0, 1000.0, True, False), 'O': (1.0, 0.0, 1000.0, True, False), 'Ne': (1.0, 0.0, 1000.0, True, False), 'Mg': (1.0, 0.0, 1000.0, True, False), 'Al': (1.0, 0.0, 1000.0, True, False), 'Si': (1.0, 0.0, 1000.0, True, False), 'S': (1.0, 0.0, 1000.0, True, False), 'Ar': (1.0, 0.0, 1000.0, True, False), 'Ca': (1.0, 0.0, 1000.0, True, False), 'Fe': (1.0, 0.0, 1000.0, True, False), 'Ni': (1.0, 0.0, 1000.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vapecOp
    def __init__(self, kT, He, C, N, O, Ne, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vapec(SpectralModel):
    def __init__(self, kT=None, He=None, C=None, N=None, O=None, Ne=None, Mg=None, Al=None, Si=None, S=None, Ar=None, Ca=None, Fe=None, Ni=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vapecComponent(kT, He, C, N, O, Ne, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, norm, name, grad_method, eps)])


class varabsOp(XspecNumericGradOp):
    modname = 'varabs'
    optype = 'mul'

class varabsComponent(SpectralComponent):
    _comp_name = 'varabs'
    _config = {'H': (1.0, 0.0, 10000.0, True, False), 'He': (1.0, 0.0, 10000.0, True, False), 'C': (1.0, 0.0, 10000.0, True, False), 'N': (1.0, 0.0, 10000.0, True, False), 'O': (1.0, 0.0, 10000.0, True, False), 'Ne': (1.0, 0.0, 10000.0, True, False), 'Na': (1.0, 0.0, 10000.0, True, False), 'Mg': (1.0, 0.0, 10000.0, True, False), 'Al': (1.0, 0.0, 10000.0, True, False), 'Si': (1.0, 0.0, 10000.0, True, False), 'S': (1.0, 0.0, 10000.0, True, False), 'Cl': (1.0, 0.0, 10000.0, True, False), 'Ar': (1.0, 0.0, 10000.0, True, False), 'Ca': (1.0, 0.0, 10000.0, True, False), 'Cr': (1.0, 0.0, 10000.0, True, False), 'Fe': (1.0, 0.0, 10000.0, True, False), 'Co': (1.0, 0.0, 10000.0, True, False), 'Ni': (1.0, 0.0, 10000.0, True, False)}
    _op_class = varabsOp
    def __init__(self, H, He, C, N, O, Ne, Na, Mg, Al, Si, S, Cl, Ar, Ca, Cr, Fe, Co, Ni, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class varabs(SpectralModel):
    def __init__(self, H=None, He=None, C=None, N=None, O=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, S=None, Cl=None, Ar=None, Ca=None, Cr=None, Fe=None, Co=None, Ni=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([varabsComponent(H, He, C, N, O, Ne, Na, Mg, Al, Si, S, Cl, Ar, Ca, Cr, Fe, Co, Ni, name, grad_method, eps)])


class vashiftOp(XspecNumericGradOp):
    modname = 'vashift'
    optype = 'con'

class vashiftComponent(SpectralComponent):
    _comp_name = 'vashift'
    _config = {'Velocity': (0.0, -10000.0, 10000.0, True, False)}
    _op_class = vashiftOp
    def __init__(self, Velocity, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vashift(SpectralModel):
    def __init__(self, Velocity=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vashiftComponent(Velocity, name, grad_method, eps)])


class vbremssOp(XspecNumericGradOp):
    modname = 'vbremss'
    optype = 'add'

class vbremssComponent(SpectralComponent):
    _comp_name = 'vbremss'
    _config = {'kT': (3.0, 0.0001, 200.0, False, False), 'HeovrH': (1.0, 0.0, 100.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vbremssOp
    def __init__(self, kT, HeovrH, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vbremss(SpectralModel):
    def __init__(self, kT=None, HeovrH=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vbremssComponent(kT, HeovrH, norm, name, grad_method, eps)])


class vcphOp(XspecNumericGradOp):
    modname = 'vcph'
    optype = 'add'

class vcphComponent(SpectralComponent):
    _comp_name = 'vcph'
    _config = {'peakT': (2.2, 0.1, 100.0, False, False), 'He': (1.0, 0.0, 1000.0, True, False), 'C': (1.0, 0.0, 1000.0, True, False), 'N': (1.0, 0.0, 1000.0, True, False), 'O': (1.0, 0.0, 1000.0, True, False), 'Ne': (1.0, 0.0, 1000.0, True, False), 'Na': (1.0, 0.0, 1000.0, True, False), 'Mg': (1.0, 0.0, 1000.0, True, False), 'Al': (1.0, 0.0, 1000.0, True, False), 'Si': (1.0, 0.0, 1000.0, True, False), 'S': (1.0, 0.0, 1000.0, True, False), 'Ar': (1.0, 0.0, 1000.0, True, False), 'Ca': (1.0, 0.0, 1000.0, True, False), 'Fe': (1.0, 0.0, 1000.0, True, False), 'Ni': (1.0, 0.0, 1000.0, True, False), 'Redshift': (0.0, 0.0, 50.0, True, False), 'switch': (1, None, None, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vcphOp
    def __init__(self, peakT, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vcph(SpectralModel):
    def __init__(self, peakT=None, He=None, C=None, N=None, O=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, S=None, Ar=None, Ca=None, Fe=None, Ni=None, Redshift=None, switch=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vcphComponent(peakT, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch, norm, name, grad_method, eps)])


class vequilOp(XspecNumericGradOp):
    modname = 'vequil'
    optype = 'add'

class vequilComponent(SpectralComponent):
    _comp_name = 'vequil'
    _config = {'kT': (1.0, 0.0808, 79.9, False, False), 'He': (1.0, 0.0, 10000.0, True, False), 'C': (1.0, 0.0, 10000.0, True, False), 'N': (1.0, 0.0, 10000.0, True, False), 'O': (1.0, 0.0, 10000.0, True, False), 'Ne': (1.0, 0.0, 10000.0, True, False), 'Mg': (1.0, 0.0, 10000.0, True, False), 'Si': (1.0, 0.0, 10000.0, True, False), 'S': (1.0, 0.0, 10000.0, True, False), 'Ar': (1.0, 0.0, 10000.0, True, False), 'Ca': (1.0, 0.0, 10000.0, True, False), 'Fe': (1.0, 0.0, 10000.0, True, False), 'Ni': (1.0, 0.0, 10000.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vequilOp
    def __init__(self, kT, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vequil(SpectralModel):
    def __init__(self, kT=None, He=None, C=None, N=None, O=None, Ne=None, Mg=None, Si=None, S=None, Ar=None, Ca=None, Fe=None, Ni=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vequilComponent(kT, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Redshift, norm, name, grad_method, eps)])


class vgademOp(XspecNumericGradOp):
    modname = 'vgadem'
    optype = 'add'

class vgademComponent(SpectralComponent):
    _comp_name = 'vgadem'
    _config = {'Tmean': (4.0, 0.01, 20.0, True, False), 'Tsigma': (0.1, 0.01, 100.0, False, False), 'nH': (1.0, 1e-06, 1e+20, True, False), 'He': (1.0, 0.0, 10.0, True, False), 'C': (1.0, 0.0, 10.0, True, False), 'N': (1.0, 0.0, 10.0, True, False), 'O': (1.0, 0.0, 10.0, True, False), 'Ne': (1.0, 0.0, 10.0, True, False), 'Na': (1.0, 0.0, 10.0, True, False), 'Mg': (1.0, 0.0, 10.0, True, False), 'Al': (1.0, 0.0, 10.0, True, False), 'Si': (1.0, 0.0, 10.0, True, False), 'S': (1.0, 0.0, 10.0, True, False), 'Ar': (1.0, 0.0, 10.0, True, False), 'Ca': (1.0, 0.0, 10.0, True, False), 'Fe': (1.0, 0.0, 10.0, True, False), 'Ni': (1.0, 0.0, 10.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'switch': (2, None, None, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vgademOp
    def __init__(self, Tmean, Tsigma, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vgadem(SpectralModel):
    def __init__(self, Tmean=None, Tsigma=None, nH=None, He=None, C=None, N=None, O=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, S=None, Ar=None, Ca=None, Fe=None, Ni=None, Redshift=None, switch=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vgademComponent(Tmean, Tsigma, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch, norm, name, grad_method, eps)])


class vgneiOp(XspecNumericGradOp):
    modname = 'vgnei'
    optype = 'add'

class vgneiComponent(SpectralComponent):
    _comp_name = 'vgnei'
    _config = {'kT': (1.0, 0.0808, 79.9, False, False), 'H': (1.0, 0.0, 1.0, True, False), 'He': (1.0, 0.0, 10000.0, True, False), 'C': (1.0, 0.0, 10000.0, True, False), 'N': (1.0, 0.0, 10000.0, True, False), 'O': (1.0, 0.0, 10000.0, True, False), 'Ne': (1.0, 0.0, 10000.0, True, False), 'Mg': (1.0, 0.0, 10000.0, True, False), 'Si': (1.0, 0.0, 10000.0, True, False), 'S': (1.0, 0.0, 10000.0, True, False), 'Ar': (1.0, 0.0, 10000.0, True, False), 'Ca': (1.0, 0.0, 10000.0, True, False), 'Fe': (1.0, 0.0, 10000.0, True, False), 'Ni': (1.0, 0.0, 10000.0, True, False), 'Tau': (100000000000.0, 100000000.0, 50000000000000.0, False, False), 'meankT': (1.0, 0.0808, 79.9, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vgneiOp
    def __init__(self, kT, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau, meankT, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vgnei(SpectralModel):
    def __init__(self, kT=None, H=None, He=None, C=None, N=None, O=None, Ne=None, Mg=None, Si=None, S=None, Ar=None, Ca=None, Fe=None, Ni=None, Tau=None, meankT=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vgneiComponent(kT, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau, meankT, Redshift, norm, name, grad_method, eps)])


class vmcflowOp(XspecNumericGradOp):
    modname = 'vmcflow'
    optype = 'add'

class vmcflowComponent(SpectralComponent):
    _comp_name = 'vmcflow'
    _config = {'lowT': (0.1, 0.0808, 79.9, False, False), 'highT': (4.0, 0.0808, 79.9, False, False), 'He': (1.0, 0.0, 1000.0, True, False), 'C': (1.0, 0.0, 1000.0, True, False), 'N': (1.0, 0.0, 1000.0, True, False), 'O': (1.0, 0.0, 1000.0, True, False), 'Ne': (1.0, 0.0, 1000.0, True, False), 'Na': (1.0, 0.0, 1000.0, True, False), 'Mg': (1.0, 0.0, 1000.0, True, False), 'Al': (1.0, 0.0, 1000.0, True, False), 'Si': (1.0, 0.0, 1000.0, True, False), 'S': (1.0, 0.0, 1000.0, True, False), 'Ar': (1.0, 0.0, 1000.0, True, False), 'Ca': (1.0, 0.0, 1000.0, True, False), 'Fe': (1.0, 0.0, 1000.0, True, False), 'Ni': (1.0, 0.0, 1000.0, True, False), 'Redshift': (0.0, 0.0, 10.0, True, False), 'switch': (1, None, None, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vmcflowOp
    def __init__(self, lowT, highT, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vmcflow(SpectralModel):
    def __init__(self, lowT=None, highT=None, He=None, C=None, N=None, O=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, S=None, Ar=None, Ca=None, Fe=None, Ni=None, Redshift=None, switch=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vmcflowComponent(lowT, highT, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch, norm, name, grad_method, eps)])


class vmekaOp(XspecNumericGradOp):
    modname = 'vmeka'
    optype = 'add'

class vmekaComponent(SpectralComponent):
    _comp_name = 'vmeka'
    _config = {'kT': (1.0, 0.001, 100.0, False, False), 'nH': (1.0, 1e-06, 1e+20, True, False), 'He': (1.0, 0.0, 1000.0, True, False), 'C': (1.0, 0.0, 1000.0, True, False), 'N': (1.0, 0.0, 1000.0, True, False), 'O': (1.0, 0.0, 1000.0, True, False), 'Ne': (1.0, 0.0, 1000.0, True, False), 'Na': (1.0, 0.0, 1000.0, True, False), 'Mg': (1.0, 0.0, 1000.0, True, False), 'Al': (1.0, 0.0, 1000.0, True, False), 'Si': (1.0, 0.0, 1000.0, True, False), 'S': (1.0, 0.0, 1000.0, True, False), 'Ar': (1.0, 0.0, 1000.0, True, False), 'Ca': (1.0, 0.0, 1000.0, True, False), 'Fe': (1.0, 0.0, 1000.0, True, False), 'Ni': (1.0, 0.0, 1000.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vmekaOp
    def __init__(self, kT, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vmeka(SpectralModel):
    def __init__(self, kT=None, nH=None, He=None, C=None, N=None, O=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, S=None, Ar=None, Ca=None, Fe=None, Ni=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vmekaComponent(kT, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, norm, name, grad_method, eps)])


class vmekalOp(XspecNumericGradOp):
    modname = 'vmekal'
    optype = 'add'

class vmekalComponent(SpectralComponent):
    _comp_name = 'vmekal'
    _config = {'kT': (1.0, 0.0808, 79.9, False, False), 'nH': (1.0, 1e-06, 1e+20, True, False), 'He': (1.0, 0.0, 1000.0, True, False), 'C': (1.0, 0.0, 1000.0, True, False), 'N': (1.0, 0.0, 1000.0, True, False), 'O': (1.0, 0.0, 1000.0, True, False), 'Ne': (1.0, 0.0, 1000.0, True, False), 'Na': (1.0, 0.0, 1000.0, True, False), 'Mg': (1.0, 0.0, 1000.0, True, False), 'Al': (1.0, 0.0, 1000.0, True, False), 'Si': (1.0, 0.0, 1000.0, True, False), 'S': (1.0, 0.0, 1000.0, True, False), 'Ar': (1.0, 0.0, 1000.0, True, False), 'Ca': (1.0, 0.0, 1000.0, True, False), 'Fe': (1.0, 0.0, 1000.0, True, False), 'Ni': (1.0, 0.0, 1000.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'switch': (1, None, None, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vmekalOp
    def __init__(self, kT, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vmekal(SpectralModel):
    def __init__(self, kT=None, nH=None, He=None, C=None, N=None, O=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, S=None, Ar=None, Ca=None, Fe=None, Ni=None, Redshift=None, switch=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vmekalComponent(kT, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch, norm, name, grad_method, eps)])


class vmshiftOp(XspecNumericGradOp):
    modname = 'vmshift'
    optype = 'con'

class vmshiftComponent(SpectralComponent):
    _comp_name = 'vmshift'
    _config = {'Velocity': (0.0, -10000.0, 10000.0, True, False)}
    _op_class = vmshiftOp
    def __init__(self, Velocity, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vmshift(SpectralModel):
    def __init__(self, Velocity=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vmshiftComponent(Velocity, name, grad_method, eps)])


class vneiOp(XspecNumericGradOp):
    modname = 'vnei'
    optype = 'add'

class vneiComponent(SpectralComponent):
    _comp_name = 'vnei'
    _config = {'kT': (1.0, 0.0808, 79.9, False, False), 'H': (1.0, 0.0, 1.0, True, False), 'He': (1.0, 0.0, 10000.0, True, False), 'C': (1.0, 0.0, 10000.0, True, False), 'N': (1.0, 0.0, 10000.0, True, False), 'O': (1.0, 0.0, 10000.0, True, False), 'Ne': (1.0, 0.0, 10000.0, True, False), 'Mg': (1.0, 0.0, 10000.0, True, False), 'Si': (1.0, 0.0, 10000.0, True, False), 'S': (1.0, 0.0, 10000.0, True, False), 'Ar': (1.0, 0.0, 10000.0, True, False), 'Ca': (1.0, 0.0, 10000.0, True, False), 'Fe': (1.0, 0.0, 10000.0, True, False), 'Ni': (1.0, 0.0, 10000.0, True, False), 'Tau': (100000000000.0, 100000000.0, 50000000000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vneiOp
    def __init__(self, kT, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vnei(SpectralModel):
    def __init__(self, kT=None, H=None, He=None, C=None, N=None, O=None, Ne=None, Mg=None, Si=None, S=None, Ar=None, Ca=None, Fe=None, Ni=None, Tau=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vneiComponent(kT, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau, Redshift, norm, name, grad_method, eps)])


class vnpshockOp(XspecNumericGradOp):
    modname = 'vnpshock'
    optype = 'add'

class vnpshockComponent(SpectralComponent):
    _comp_name = 'vnpshock'
    _config = {'kT_a': (1.0, 0.0808, 79.9, False, False), 'kT_b': (0.5, 0.01, 79.9, False, False), 'H': (1.0, 0.0, 1.0, True, False), 'He': (1.0, 0.0, 10000.0, True, False), 'C': (1.0, 0.0, 10000.0, True, False), 'N': (1.0, 0.0, 10000.0, True, False), 'O': (1.0, 0.0, 10000.0, True, False), 'Ne': (1.0, 0.0, 10000.0, True, False), 'Mg': (1.0, 0.0, 10000.0, True, False), 'Si': (1.0, 0.0, 10000.0, True, False), 'S': (1.0, 0.0, 10000.0, True, False), 'Ar': (1.0, 0.0, 10000.0, True, False), 'Ca': (1.0, 0.0, 10000.0, True, False), 'Fe': (1.0, 0.0, 10000.0, True, False), 'Ni': (1.0, 0.0, 10000.0, True, False), 'Tau_l': (0.0, 0.0, 50000000000000.0, True, False), 'Tau_u': (100000000000.0, 100000000.0, 50000000000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vnpshockOp
    def __init__(self, kT_a, kT_b, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau_l, Tau_u, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vnpshock(SpectralModel):
    def __init__(self, kT_a=None, kT_b=None, H=None, He=None, C=None, N=None, O=None, Ne=None, Mg=None, Si=None, S=None, Ar=None, Ca=None, Fe=None, Ni=None, Tau_l=None, Tau_u=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vnpshockComponent(kT_a, kT_b, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau_l, Tau_u, Redshift, norm, name, grad_method, eps)])


class voigtOp(XspecNumericGradOp):
    modname = 'voigt'
    optype = 'add'

class voigtComponent(SpectralComponent):
    _comp_name = 'voigt'
    _config = {'LineE': (6.5, 0.0, 1000000.0, False, False), 'Sigma': (0.01, 0.0, 20.0, False, False), 'Gamma': (0.01, 0.0, 20.0, False, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = voigtOp
    def __init__(self, LineE, Sigma, Gamma, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class voigt(SpectralModel):
    def __init__(self, LineE=None, Sigma=None, Gamma=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([voigtComponent(LineE, Sigma, Gamma, norm, name, grad_method, eps)])


class vphabsOp(XspecNumericGradOp):
    modname = 'vphabs'
    optype = 'mul'

class vphabsComponent(SpectralComponent):
    _comp_name = 'vphabs'
    _config = {'nH': (1.0, 0.0, 1000000.0, False, False), 'He': (1.0, 0.0, 1000.0, True, False), 'C': (1.0, 0.0, 1000.0, True, False), 'N': (1.0, 0.0, 1000.0, True, False), 'O': (1.0, 0.0, 1000.0, True, False), 'Ne': (1.0, 0.0, 1000.0, True, False), 'Na': (1.0, 0.0, 1000.0, True, False), 'Mg': (1.0, 0.0, 1000.0, True, False), 'Al': (1.0, 0.0, 1000.0, True, False), 'Si': (1.0, 0.0, 1000.0, True, False), 'S': (1.0, 0.0, 1000.0, True, False), 'Cl': (1.0, 0.0, 1000.0, True, False), 'Ar': (1.0, 0.0, 1000.0, True, False), 'Ca': (1.0, 0.0, 1000.0, True, False), 'Cr': (1.0, 0.0, 1000.0, True, False), 'Fe': (1.0, 0.0, 1000.0, True, False), 'Co': (1.0, 0.0, 1000.0, True, False), 'Ni': (1.0, 0.0, 1000.0, True, False)}
    _op_class = vphabsOp
    def __init__(self, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Cl, Ar, Ca, Cr, Fe, Co, Ni, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vphabs(SpectralModel):
    def __init__(self, nH=None, He=None, C=None, N=None, O=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, S=None, Cl=None, Ar=None, Ca=None, Cr=None, Fe=None, Co=None, Ni=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vphabsComponent(nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Cl, Ar, Ca, Cr, Fe, Co, Ni, name, grad_method, eps)])


class vpshockOp(XspecNumericGradOp):
    modname = 'vpshock'
    optype = 'add'

class vpshockComponent(SpectralComponent):
    _comp_name = 'vpshock'
    _config = {'kT': (1.0, 0.0808, 79.9, False, False), 'H': (1.0, 0.0, 1.0, True, False), 'He': (1.0, 0.0, 10000.0, True, False), 'C': (1.0, 0.0, 10000.0, True, False), 'N': (1.0, 0.0, 10000.0, True, False), 'O': (1.0, 0.0, 10000.0, True, False), 'Ne': (1.0, 0.0, 10000.0, True, False), 'Mg': (1.0, 0.0, 10000.0, True, False), 'Si': (1.0, 0.0, 10000.0, True, False), 'S': (1.0, 0.0, 10000.0, True, False), 'Ar': (1.0, 0.0, 10000.0, True, False), 'Ca': (1.0, 0.0, 10000.0, True, False), 'Fe': (1.0, 0.0, 10000.0, True, False), 'Ni': (1.0, 0.0, 10000.0, True, False), 'Tau_l': (0.0, 0.0, 50000000000000.0, True, False), 'Tau_u': (100000000000.0, 100000000.0, 50000000000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vpshockOp
    def __init__(self, kT, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau_l, Tau_u, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vpshock(SpectralModel):
    def __init__(self, kT=None, H=None, He=None, C=None, N=None, O=None, Ne=None, Mg=None, Si=None, S=None, Ar=None, Ca=None, Fe=None, Ni=None, Tau_l=None, Tau_u=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vpshockComponent(kT, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau_l, Tau_u, Redshift, norm, name, grad_method, eps)])


class vraymondOp(XspecNumericGradOp):
    modname = 'vraymond'
    optype = 'add'

class vraymondComponent(SpectralComponent):
    _comp_name = 'vraymond'
    _config = {'kT': (6.5, 0.0808, 79.9, False, False), 'He': (1.0, 0.0, 10000.0, True, False), 'C': (1.0, 0.0, 10000.0, True, False), 'N': (1.0, 0.0, 10000.0, True, False), 'O': (1.0, 0.0, 10000.0, True, False), 'Ne': (1.0, 0.0, 10000.0, True, False), 'Mg': (1.0, 0.0, 10000.0, True, False), 'Si': (1.0, 0.0, 10000.0, True, False), 'S': (1.0, 0.0, 10000.0, True, False), 'Ar': (1.0, 0.0, 10000.0, True, False), 'Ca': (1.0, 0.0, 10000.0, True, False), 'Fe': (1.0, 0.0, 10000.0, True, False), 'Ni': (1.0, 0.0, 10000.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vraymondOp
    def __init__(self, kT, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vraymond(SpectralModel):
    def __init__(self, kT=None, He=None, C=None, N=None, O=None, Ne=None, Mg=None, Si=None, S=None, Ar=None, Ca=None, Fe=None, Ni=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vraymondComponent(kT, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Redshift, norm, name, grad_method, eps)])


class vrneiOp(XspecNumericGradOp):
    modname = 'vrnei'
    optype = 'add'

class vrneiComponent(SpectralComponent):
    _comp_name = 'vrnei'
    _config = {'kT': (0.5, 0.0808, 79.9, False, False), 'kT_init': (1.0, 0.0808, 79.9, False, False), 'H': (1.0, 0.0, 1.0, True, False), 'He': (1.0, 0.0, 10000.0, True, False), 'C': (1.0, 0.0, 10000.0, True, False), 'N': (1.0, 0.0, 10000.0, True, False), 'O': (1.0, 0.0, 10000.0, True, False), 'Ne': (1.0, 0.0, 10000.0, True, False), 'Mg': (1.0, 0.0, 10000.0, True, False), 'Si': (1.0, 0.0, 10000.0, True, False), 'S': (1.0, 0.0, 10000.0, True, False), 'Ar': (1.0, 0.0, 10000.0, True, False), 'Ca': (1.0, 0.0, 10000.0, True, False), 'Fe': (1.0, 0.0, 10000.0, True, False), 'Ni': (1.0, 0.0, 10000.0, True, False), 'Tau': (100000000000.0, 100000000.0, 50000000000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vrneiOp
    def __init__(self, kT, kT_init, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vrnei(SpectralModel):
    def __init__(self, kT=None, kT_init=None, H=None, He=None, C=None, N=None, O=None, Ne=None, Mg=None, Si=None, S=None, Ar=None, Ca=None, Fe=None, Ni=None, Tau=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vrneiComponent(kT, kT_init, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau, Redshift, norm, name, grad_method, eps)])


class vsedovOp(XspecNumericGradOp):
    modname = 'vsedov'
    optype = 'add'

class vsedovComponent(SpectralComponent):
    _comp_name = 'vsedov'
    _config = {'kT_a': (1.0, 0.0808, 79.9, False, False), 'kT_b': (0.5, 0.01, 79.9, False, False), 'H': (1.0, 0.0, 1.0, True, False), 'He': (1.0, 0.0, 10000.0, True, False), 'C': (1.0, 0.0, 10000.0, True, False), 'N': (1.0, 0.0, 10000.0, True, False), 'O': (1.0, 0.0, 10000.0, True, False), 'Ne': (1.0, 0.0, 10000.0, True, False), 'Mg': (1.0, 0.0, 10000.0, True, False), 'Si': (1.0, 0.0, 10000.0, True, False), 'S': (1.0, 0.0, 10000.0, True, False), 'Ar': (1.0, 0.0, 10000.0, True, False), 'Ca': (1.0, 0.0, 10000.0, True, False), 'Fe': (1.0, 0.0, 10000.0, True, False), 'Ni': (1.0, 0.0, 10000.0, True, False), 'Tau': (100000000000.0, 100000000.0, 50000000000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vsedovOp
    def __init__(self, kT_a, kT_b, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vsedov(SpectralModel):
    def __init__(self, kT_a=None, kT_b=None, H=None, He=None, C=None, N=None, O=None, Ne=None, Mg=None, Si=None, S=None, Ar=None, Ca=None, Fe=None, Ni=None, Tau=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vsedovComponent(kT_a, kT_b, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau, Redshift, norm, name, grad_method, eps)])


class vtapecOp(XspecNumericGradOp):
    modname = 'vtapec'
    optype = 'add'

class vtapecComponent(SpectralComponent):
    _comp_name = 'vtapec'
    _config = {'kT': (6.5, 0.0808, 68.447, False, False), 'kTi': (6.5, 0.0808, 68.447, False, False), 'He': (1.0, 0.0, 1000.0, True, False), 'C': (1.0, 0.0, 1000.0, True, False), 'N': (1.0, 0.0, 1000.0, True, False), 'O': (1.0, 0.0, 1000.0, True, False), 'Ne': (1.0, 0.0, 1000.0, True, False), 'Mg': (1.0, 0.0, 1000.0, True, False), 'Al': (1.0, 0.0, 1000.0, True, False), 'Si': (1.0, 0.0, 1000.0, True, False), 'S': (1.0, 0.0, 1000.0, True, False), 'Ar': (1.0, 0.0, 1000.0, True, False), 'Ca': (1.0, 0.0, 1000.0, True, False), 'Fe': (1.0, 0.0, 1000.0, True, False), 'Ni': (1.0, 0.0, 1000.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vtapecOp
    def __init__(self, kT, kTi, He, C, N, O, Ne, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vtapec(SpectralModel):
    def __init__(self, kT=None, kTi=None, He=None, C=None, N=None, O=None, Ne=None, Mg=None, Al=None, Si=None, S=None, Ar=None, Ca=None, Fe=None, Ni=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vtapecComponent(kT, kTi, He, C, N, O, Ne, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, norm, name, grad_method, eps)])


class vvapecOp(XspecNumericGradOp):
    modname = 'vvapec'
    optype = 'add'

class vvapecComponent(SpectralComponent):
    _comp_name = 'vvapec'
    _config = {'kT': (6.5, 0.0808, 68.447, False, False), 'H': (1.0, 0.0, 1000.0, True, False), 'He': (1.0, 0.0, 1000.0, True, False), 'Li': (1.0, 0.0, 1000.0, True, False), 'Be': (1.0, 0.0, 1000.0, True, False), 'B': (1.0, 0.0, 1000.0, True, False), 'C': (1.0, 0.0, 1000.0, True, False), 'N': (1.0, 0.0, 1000.0, True, False), 'O': (1.0, 0.0, 1000.0, True, False), 'F': (1.0, 0.0, 1000.0, True, False), 'Ne': (1.0, 0.0, 1000.0, True, False), 'Na': (1.0, 0.0, 1000.0, True, False), 'Mg': (1.0, 0.0, 1000.0, True, False), 'Al': (1.0, 0.0, 1000.0, True, False), 'Si': (1.0, 0.0, 1000.0, True, False), 'P': (1.0, 0.0, 1000.0, True, False), 'S': (1.0, 0.0, 1000.0, True, False), 'Cl': (1.0, 0.0, 1000.0, True, False), 'Ar': (1.0, 0.0, 1000.0, True, False), 'K': (1.0, 0.0, 1000.0, True, False), 'Ca': (1.0, 0.0, 1000.0, True, False), 'Sc': (1.0, 0.0, 1000.0, True, False), 'Ti': (1.0, 0.0, 1000.0, True, False), 'V': (1.0, 0.0, 1000.0, True, False), 'Cr': (1.0, 0.0, 1000.0, True, False), 'Mn': (1.0, 0.0, 1000.0, True, False), 'Fe': (1.0, 0.0, 1000.0, True, False), 'Co': (1.0, 0.0, 1000.0, True, False), 'Ni': (1.0, 0.0, 1000.0, True, False), 'Cu': (1.0, 0.0, 1000.0, True, False), 'Zn': (1.0, 0.0, 1000.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vvapecOp
    def __init__(self, kT, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vvapec(SpectralModel):
    def __init__(self, kT=None, H=None, He=None, Li=None, Be=None, B=None, C=None, N=None, O=None, F=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, P=None, S=None, Cl=None, Ar=None, K=None, Ca=None, Sc=None, Ti=None, V=None, Cr=None, Mn=None, Fe=None, Co=None, Ni=None, Cu=None, Zn=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vvapecComponent(kT, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Redshift, norm, name, grad_method, eps)])


class vvgneiOp(XspecNumericGradOp):
    modname = 'vvgnei'
    optype = 'add'

class vvgneiComponent(SpectralComponent):
    _comp_name = 'vvgnei'
    _config = {'kT': (1.0, 0.0808, 79.9, False, False), 'H': (1.0, 0.0, 1000.0, True, False), 'He': (1.0, 0.0, 1000.0, True, False), 'Li': (1.0, 0.0, 1000.0, True, False), 'Be': (1.0, 0.0, 1000.0, True, False), 'B': (1.0, 0.0, 1000.0, True, False), 'C': (1.0, 0.0, 1000.0, True, False), 'N': (1.0, 0.0, 1000.0, True, False), 'O': (1.0, 0.0, 1000.0, True, False), 'F': (1.0, 0.0, 1000.0, True, False), 'Ne': (1.0, 0.0, 1000.0, True, False), 'Na': (1.0, 0.0, 1000.0, True, False), 'Mg': (1.0, 0.0, 1000.0, True, False), 'Al': (1.0, 0.0, 1000.0, True, False), 'Si': (1.0, 0.0, 1000.0, True, False), 'P': (1.0, 0.0, 1000.0, True, False), 'S': (1.0, 0.0, 1000.0, True, False), 'Cl': (1.0, 0.0, 1000.0, True, False), 'Ar': (1.0, 0.0, 1000.0, True, False), 'K': (1.0, 0.0, 1000.0, True, False), 'Ca': (1.0, 0.0, 1000.0, True, False), 'Sc': (1.0, 0.0, 1000.0, True, False), 'Ti': (1.0, 0.0, 1000.0, True, False), 'V': (1.0, 0.0, 1000.0, True, False), 'Cr': (1.0, 0.0, 1000.0, True, False), 'Mn': (1.0, 0.0, 1000.0, True, False), 'Fe': (1.0, 0.0, 1000.0, True, False), 'Co': (1.0, 0.0, 1000.0, True, False), 'Ni': (1.0, 0.0, 1000.0, True, False), 'Cu': (1.0, 0.0, 1000.0, True, False), 'Zn': (1.0, 0.0, 1000.0, True, False), 'Tau': (100000000000.0, 100000000.0, 50000000000000.0, False, False), 'meankT': (1.0, 0.0808, 79.9, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vvgneiOp
    def __init__(self, kT, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau, meankT, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vvgnei(SpectralModel):
    def __init__(self, kT=None, H=None, He=None, Li=None, Be=None, B=None, C=None, N=None, O=None, F=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, P=None, S=None, Cl=None, Ar=None, K=None, Ca=None, Sc=None, Ti=None, V=None, Cr=None, Mn=None, Fe=None, Co=None, Ni=None, Cu=None, Zn=None, Tau=None, meankT=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vvgneiComponent(kT, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau, meankT, Redshift, norm, name, grad_method, eps)])


class vvneiOp(XspecNumericGradOp):
    modname = 'vvnei'
    optype = 'add'

class vvneiComponent(SpectralComponent):
    _comp_name = 'vvnei'
    _config = {'kT': (1.0, 0.0808, 79.9, False, False), 'H': (1.0, 0.0, 1000.0, True, False), 'He': (1.0, 0.0, 1000.0, True, False), 'Li': (1.0, 0.0, 1000.0, True, False), 'Be': (1.0, 0.0, 1000.0, True, False), 'B': (1.0, 0.0, 1000.0, True, False), 'C': (1.0, 0.0, 1000.0, True, False), 'N': (1.0, 0.0, 1000.0, True, False), 'O': (1.0, 0.0, 1000.0, True, False), 'F': (1.0, 0.0, 1000.0, True, False), 'Ne': (1.0, 0.0, 1000.0, True, False), 'Na': (1.0, 0.0, 1000.0, True, False), 'Mg': (1.0, 0.0, 1000.0, True, False), 'Al': (1.0, 0.0, 1000.0, True, False), 'Si': (1.0, 0.0, 1000.0, True, False), 'P': (1.0, 0.0, 1000.0, True, False), 'S': (1.0, 0.0, 1000.0, True, False), 'Cl': (1.0, 0.0, 1000.0, True, False), 'Ar': (1.0, 0.0, 1000.0, True, False), 'K': (1.0, 0.0, 1000.0, True, False), 'Ca': (1.0, 0.0, 1000.0, True, False), 'Sc': (1.0, 0.0, 1000.0, True, False), 'Ti': (1.0, 0.0, 1000.0, True, False), 'V': (1.0, 0.0, 1000.0, True, False), 'Cr': (1.0, 0.0, 1000.0, True, False), 'Mn': (1.0, 0.0, 1000.0, True, False), 'Fe': (1.0, 0.0, 1000.0, True, False), 'Co': (1.0, 0.0, 1000.0, True, False), 'Ni': (1.0, 0.0, 1000.0, True, False), 'Cu': (1.0, 0.0, 1000.0, True, False), 'Zn': (1.0, 0.0, 1000.0, True, False), 'Tau': (100000000000.0, 100000000.0, 50000000000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vvneiOp
    def __init__(self, kT, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vvnei(SpectralModel):
    def __init__(self, kT=None, H=None, He=None, Li=None, Be=None, B=None, C=None, N=None, O=None, F=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, P=None, S=None, Cl=None, Ar=None, K=None, Ca=None, Sc=None, Ti=None, V=None, Cr=None, Mn=None, Fe=None, Co=None, Ni=None, Cu=None, Zn=None, Tau=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vvneiComponent(kT, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau, Redshift, norm, name, grad_method, eps)])


class vvnpshockOp(XspecNumericGradOp):
    modname = 'vvnpshock'
    optype = 'add'

class vvnpshockComponent(SpectralComponent):
    _comp_name = 'vvnpshock'
    _config = {'kT_a': (1.0, 0.0808, 79.9, False, False), 'kT_b': (0.5, 0.01, 79.9, False, False), 'H': (1.0, 0.0, 1000.0, True, False), 'He': (1.0, 0.0, 1000.0, True, False), 'Li': (1.0, 0.0, 1000.0, True, False), 'Be': (1.0, 0.0, 1000.0, True, False), 'B': (1.0, 0.0, 1000.0, True, False), 'C': (1.0, 0.0, 1000.0, True, False), 'N': (1.0, 0.0, 1000.0, True, False), 'O': (1.0, 0.0, 1000.0, True, False), 'F': (1.0, 0.0, 1000.0, True, False), 'Ne': (1.0, 0.0, 1000.0, True, False), 'Na': (1.0, 0.0, 1000.0, True, False), 'Mg': (1.0, 0.0, 1000.0, True, False), 'Al': (1.0, 0.0, 1000.0, True, False), 'Si': (1.0, 0.0, 1000.0, True, False), 'P': (1.0, 0.0, 1000.0, True, False), 'S': (1.0, 0.0, 1000.0, True, False), 'Cl': (1.0, 0.0, 1000.0, True, False), 'Ar': (1.0, 0.0, 1000.0, True, False), 'K': (1.0, 0.0, 1000.0, True, False), 'Ca': (1.0, 0.0, 1000.0, True, False), 'Sc': (1.0, 0.0, 1000.0, True, False), 'Ti': (1.0, 0.0, 1000.0, True, False), 'V': (1.0, 0.0, 1000.0, True, False), 'Cr': (1.0, 0.0, 1000.0, True, False), 'Mn': (1.0, 0.0, 1000.0, True, False), 'Fe': (1.0, 0.0, 1000.0, True, False), 'Co': (1.0, 0.0, 1000.0, True, False), 'Ni': (1.0, 0.0, 1000.0, True, False), 'Cu': (1.0, 0.0, 1000.0, True, False), 'Zn': (1.0, 0.0, 1000.0, True, False), 'Tau_l': (0.0, 0.0, 50000000000000.0, True, False), 'Tau_u': (100000000000.0, 100000000.0, 50000000000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vvnpshockOp
    def __init__(self, kT_a, kT_b, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau_l, Tau_u, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vvnpshock(SpectralModel):
    def __init__(self, kT_a=None, kT_b=None, H=None, He=None, Li=None, Be=None, B=None, C=None, N=None, O=None, F=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, P=None, S=None, Cl=None, Ar=None, K=None, Ca=None, Sc=None, Ti=None, V=None, Cr=None, Mn=None, Fe=None, Co=None, Ni=None, Cu=None, Zn=None, Tau_l=None, Tau_u=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vvnpshockComponent(kT_a, kT_b, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau_l, Tau_u, Redshift, norm, name, grad_method, eps)])


class vvpshockOp(XspecNumericGradOp):
    modname = 'vvpshock'
    optype = 'add'

class vvpshockComponent(SpectralComponent):
    _comp_name = 'vvpshock'
    _config = {'kT': (1.0, 0.0808, 79.9, False, False), 'H': (1.0, 0.0, 1000.0, True, False), 'He': (1.0, 0.0, 1000.0, True, False), 'Li': (1.0, 0.0, 1000.0, True, False), 'Be': (1.0, 0.0, 1000.0, True, False), 'B': (1.0, 0.0, 1000.0, True, False), 'C': (1.0, 0.0, 1000.0, True, False), 'N': (1.0, 0.0, 1000.0, True, False), 'O': (1.0, 0.0, 1000.0, True, False), 'F': (1.0, 0.0, 1000.0, True, False), 'Ne': (1.0, 0.0, 1000.0, True, False), 'Na': (1.0, 0.0, 1000.0, True, False), 'Mg': (1.0, 0.0, 1000.0, True, False), 'Al': (1.0, 0.0, 1000.0, True, False), 'Si': (1.0, 0.0, 1000.0, True, False), 'P': (1.0, 0.0, 1000.0, True, False), 'S': (1.0, 0.0, 1000.0, True, False), 'Cl': (1.0, 0.0, 1000.0, True, False), 'Ar': (1.0, 0.0, 1000.0, True, False), 'K': (1.0, 0.0, 1000.0, True, False), 'Ca': (1.0, 0.0, 1000.0, True, False), 'Sc': (1.0, 0.0, 1000.0, True, False), 'Ti': (1.0, 0.0, 1000.0, True, False), 'V': (1.0, 0.0, 1000.0, True, False), 'Cr': (1.0, 0.0, 1000.0, True, False), 'Mn': (1.0, 0.0, 1000.0, True, False), 'Fe': (1.0, 0.0, 1000.0, True, False), 'Co': (1.0, 0.0, 1000.0, True, False), 'Ni': (1.0, 0.0, 1000.0, True, False), 'Cu': (1.0, 0.0, 1000.0, True, False), 'Zn': (1.0, 0.0, 1000.0, True, False), 'Tau_l': (0.0, 0.0, 50000000000000.0, True, False), 'Tau_u': (100000000000.0, 100000000.0, 50000000000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vvpshockOp
    def __init__(self, kT, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau_l, Tau_u, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vvpshock(SpectralModel):
    def __init__(self, kT=None, H=None, He=None, Li=None, Be=None, B=None, C=None, N=None, O=None, F=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, P=None, S=None, Cl=None, Ar=None, K=None, Ca=None, Sc=None, Ti=None, V=None, Cr=None, Mn=None, Fe=None, Co=None, Ni=None, Cu=None, Zn=None, Tau_l=None, Tau_u=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vvpshockComponent(kT, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau_l, Tau_u, Redshift, norm, name, grad_method, eps)])


class vvrneiOp(XspecNumericGradOp):
    modname = 'vvrnei'
    optype = 'add'

class vvrneiComponent(SpectralComponent):
    _comp_name = 'vvrnei'
    _config = {'kT': (0.5, 0.0808, 79.9, False, False), 'kT_init': (1.0, 0.0808, 79.9, False, False), 'H': (1.0, 0.0, 1000.0, True, False), 'He': (1.0, 0.0, 1000.0, True, False), 'Li': (1.0, 0.0, 1000.0, True, False), 'Be': (1.0, 0.0, 1000.0, True, False), 'B': (1.0, 0.0, 1000.0, True, False), 'C': (1.0, 0.0, 1000.0, True, False), 'N': (1.0, 0.0, 1000.0, True, False), 'O': (1.0, 0.0, 1000.0, True, False), 'F': (1.0, 0.0, 1000.0, True, False), 'Ne': (1.0, 0.0, 1000.0, True, False), 'Na': (1.0, 0.0, 1000.0, True, False), 'Mg': (1.0, 0.0, 1000.0, True, False), 'Al': (1.0, 0.0, 1000.0, True, False), 'Si': (1.0, 0.0, 1000.0, True, False), 'P': (1.0, 0.0, 1000.0, True, False), 'S': (1.0, 0.0, 1000.0, True, False), 'Cl': (1.0, 0.0, 1000.0, True, False), 'Ar': (1.0, 0.0, 1000.0, True, False), 'K': (1.0, 0.0, 1000.0, True, False), 'Ca': (1.0, 0.0, 1000.0, True, False), 'Sc': (1.0, 0.0, 1000.0, True, False), 'Ti': (1.0, 0.0, 1000.0, True, False), 'V': (1.0, 0.0, 1000.0, True, False), 'Cr': (1.0, 0.0, 1000.0, True, False), 'Mn': (1.0, 0.0, 1000.0, True, False), 'Fe': (1.0, 0.0, 1000.0, True, False), 'Co': (1.0, 0.0, 1000.0, True, False), 'Ni': (1.0, 0.0, 1000.0, True, False), 'Cu': (1.0, 0.0, 1000.0, True, False), 'Zn': (1.0, 0.0, 1000.0, True, False), 'Tau': (100000000000.0, 100000000.0, 50000000000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vvrneiOp
    def __init__(self, kT, kT_init, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vvrnei(SpectralModel):
    def __init__(self, kT=None, kT_init=None, H=None, He=None, Li=None, Be=None, B=None, C=None, N=None, O=None, F=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, P=None, S=None, Cl=None, Ar=None, K=None, Ca=None, Sc=None, Ti=None, V=None, Cr=None, Mn=None, Fe=None, Co=None, Ni=None, Cu=None, Zn=None, Tau=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vvrneiComponent(kT, kT_init, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau, Redshift, norm, name, grad_method, eps)])


class vvsedovOp(XspecNumericGradOp):
    modname = 'vvsedov'
    optype = 'add'

class vvsedovComponent(SpectralComponent):
    _comp_name = 'vvsedov'
    _config = {'kT_a': (1.0, 0.0808, 79.9, False, False), 'kT_b': (0.5, 0.01, 79.9, False, False), 'H': (1.0, 0.0, 1000.0, True, False), 'He': (1.0, 0.0, 1000.0, True, False), 'Li': (1.0, 0.0, 1000.0, True, False), 'Be': (1.0, 0.0, 1000.0, True, False), 'B': (1.0, 0.0, 1000.0, True, False), 'C': (1.0, 0.0, 1000.0, True, False), 'N': (1.0, 0.0, 1000.0, True, False), 'O': (1.0, 0.0, 1000.0, True, False), 'F': (1.0, 0.0, 1000.0, True, False), 'Ne': (1.0, 0.0, 1000.0, True, False), 'Na': (1.0, 0.0, 1000.0, True, False), 'Mg': (1.0, 0.0, 1000.0, True, False), 'Al': (1.0, 0.0, 1000.0, True, False), 'Si': (1.0, 0.0, 1000.0, True, False), 'P': (1.0, 0.0, 1000.0, True, False), 'S': (1.0, 0.0, 1000.0, True, False), 'Cl': (1.0, 0.0, 1000.0, True, False), 'Ar': (1.0, 0.0, 1000.0, True, False), 'K': (1.0, 0.0, 1000.0, True, False), 'Ca': (1.0, 0.0, 1000.0, True, False), 'Sc': (1.0, 0.0, 1000.0, True, False), 'Ti': (1.0, 0.0, 1000.0, True, False), 'V': (1.0, 0.0, 1000.0, True, False), 'Cr': (1.0, 0.0, 1000.0, True, False), 'Mn': (1.0, 0.0, 1000.0, True, False), 'Fe': (1.0, 0.0, 1000.0, True, False), 'Co': (1.0, 0.0, 1000.0, True, False), 'Ni': (1.0, 0.0, 1000.0, True, False), 'Cu': (1.0, 0.0, 1000.0, True, False), 'Zn': (1.0, 0.0, 1000.0, True, False), 'Tau': (100000000000.0, 100000000.0, 50000000000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vvsedovOp
    def __init__(self, kT_a, kT_b, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vvsedov(SpectralModel):
    def __init__(self, kT_a=None, kT_b=None, H=None, He=None, Li=None, Be=None, B=None, C=None, N=None, O=None, F=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, P=None, S=None, Cl=None, Ar=None, K=None, Ca=None, Sc=None, Ti=None, V=None, Cr=None, Mn=None, Fe=None, Co=None, Ni=None, Cu=None, Zn=None, Tau=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vvsedovComponent(kT_a, kT_b, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau, Redshift, norm, name, grad_method, eps)])


class vvtapecOp(XspecNumericGradOp):
    modname = 'vvtapec'
    optype = 'add'

class vvtapecComponent(SpectralComponent):
    _comp_name = 'vvtapec'
    _config = {'kT': (6.5, 0.0808, 68.447, False, False), 'kTi': (6.5, 0.0808, 68.447, False, False), 'H': (1.0, 0.0, 1000.0, True, False), 'He': (1.0, 0.0, 1000.0, True, False), 'Li': (1.0, 0.0, 1000.0, True, False), 'Be': (1.0, 0.0, 1000.0, True, False), 'B': (1.0, 0.0, 1000.0, True, False), 'C': (1.0, 0.0, 1000.0, True, False), 'N': (1.0, 0.0, 1000.0, True, False), 'O': (1.0, 0.0, 1000.0, True, False), 'F': (1.0, 0.0, 1000.0, True, False), 'Ne': (1.0, 0.0, 1000.0, True, False), 'Na': (1.0, 0.0, 1000.0, True, False), 'Mg': (1.0, 0.0, 1000.0, True, False), 'Al': (1.0, 0.0, 1000.0, True, False), 'Si': (1.0, 0.0, 1000.0, True, False), 'P': (1.0, 0.0, 1000.0, True, False), 'S': (1.0, 0.0, 1000.0, True, False), 'Cl': (1.0, 0.0, 1000.0, True, False), 'Ar': (1.0, 0.0, 1000.0, True, False), 'K': (1.0, 0.0, 1000.0, True, False), 'Ca': (1.0, 0.0, 1000.0, True, False), 'Sc': (1.0, 0.0, 1000.0, True, False), 'Ti': (1.0, 0.0, 1000.0, True, False), 'V': (1.0, 0.0, 1000.0, True, False), 'Cr': (1.0, 0.0, 1000.0, True, False), 'Mn': (1.0, 0.0, 1000.0, True, False), 'Fe': (1.0, 0.0, 1000.0, True, False), 'Co': (1.0, 0.0, 1000.0, True, False), 'Ni': (1.0, 0.0, 1000.0, True, False), 'Cu': (1.0, 0.0, 1000.0, True, False), 'Zn': (1.0, 0.0, 1000.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vvtapecOp
    def __init__(self, kT, kTi, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vvtapec(SpectralModel):
    def __init__(self, kT=None, kTi=None, H=None, He=None, Li=None, Be=None, B=None, C=None, N=None, O=None, F=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, P=None, S=None, Cl=None, Ar=None, K=None, Ca=None, Sc=None, Ti=None, V=None, Cr=None, Mn=None, Fe=None, Co=None, Ni=None, Cu=None, Zn=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vvtapecComponent(kT, kTi, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Redshift, norm, name, grad_method, eps)])


class vvwdemOp(XspecNumericGradOp):
    modname = 'vvwdem'
    optype = 'add'

class vvwdemComponent(SpectralComponent):
    _comp_name = 'vvwdem'
    _config = {'Tmax': (1.0, 0.01, 20.0, False, False), 'beta': (0.1, 0.01, 1.0, False, False), 'inv_slope': (0.25, -1.0, 10.0, False, False), 'nH': (1.0, 1e-06, 1e+20, True, False), 'H': (1.0, 0.0, 1000.0, True, False), 'He': (1.0, 0.0, 1000.0, True, False), 'Li': (1.0, 0.0, 1000.0, True, False), 'Be': (1.0, 0.0, 1000.0, True, False), 'B': (1.0, 0.0, 1000.0, True, False), 'C': (1.0, 0.0, 1000.0, True, False), 'N': (1.0, 0.0, 1000.0, True, False), 'O': (1.0, 0.0, 1000.0, True, False), 'F': (1.0, 0.0, 1000.0, True, False), 'Ne': (1.0, 0.0, 1000.0, True, False), 'Na': (1.0, 0.0, 1000.0, True, False), 'Mg': (1.0, 0.0, 1000.0, True, False), 'Al': (1.0, 0.0, 1000.0, True, False), 'Si': (1.0, 0.0, 1000.0, True, False), 'P': (1.0, 0.0, 1000.0, True, False), 'S': (1.0, 0.0, 1000.0, True, False), 'Cl': (1.0, 0.0, 1000.0, True, False), 'Ar': (1.0, 0.0, 1000.0, True, False), 'K': (1.0, 0.0, 1000.0, True, False), 'Ca': (1.0, 0.0, 1000.0, True, False), 'Sc': (1.0, 0.0, 1000.0, True, False), 'Ti': (1.0, 0.0, 1000.0, True, False), 'V': (1.0, 0.0, 1000.0, True, False), 'Cr': (1.0, 0.0, 1000.0, True, False), 'Mn': (1.0, 0.0, 1000.0, True, False), 'Fe': (1.0, 0.0, 1000.0, True, False), 'Co': (1.0, 0.0, 1000.0, True, False), 'Ni': (1.0, 0.0, 1000.0, True, False), 'Cu': (1.0, 0.0, 1000.0, True, False), 'Zn': (1.0, 0.0, 1000.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'switch': (2, None, None, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vvwdemOp
    def __init__(self, Tmax, beta, inv_slope, nH, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Redshift, switch, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vvwdem(SpectralModel):
    def __init__(self, Tmax=None, beta=None, inv_slope=None, nH=None, H=None, He=None, Li=None, Be=None, B=None, C=None, N=None, O=None, F=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, P=None, S=None, Cl=None, Ar=None, K=None, Ca=None, Sc=None, Ti=None, V=None, Cr=None, Mn=None, Fe=None, Co=None, Ni=None, Cu=None, Zn=None, Redshift=None, switch=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vvwdemComponent(Tmax, beta, inv_slope, nH, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Redshift, switch, norm, name, grad_method, eps)])


class vwdemOp(XspecNumericGradOp):
    modname = 'vwdem'
    optype = 'add'

class vwdemComponent(SpectralComponent):
    _comp_name = 'vwdem'
    _config = {'Tmax': (1.0, 0.01, 20.0, False, False), 'beta': (0.1, 0.01, 1.0, False, False), 'inv_slope': (0.25, -1.0, 10.0, False, False), 'nH': (1.0, 1e-06, 1e+20, True, False), 'He': (1.0, 0.0, 10.0, True, False), 'C': (1.0, 0.0, 10.0, True, False), 'N': (1.0, 0.0, 10.0, True, False), 'O': (1.0, 0.0, 10.0, True, False), 'Ne': (1.0, 0.0, 10.0, True, False), 'Na': (1.0, 0.0, 10.0, True, False), 'Mg': (1.0, 0.0, 10.0, True, False), 'Al': (1.0, 0.0, 10.0, True, False), 'Si': (1.0, 0.0, 10.0, True, False), 'S': (1.0, 0.0, 10.0, True, False), 'Ar': (1.0, 0.0, 10.0, True, False), 'Ca': (1.0, 0.0, 10.0, True, False), 'Fe': (1.0, 0.0, 10.0, True, False), 'Ni': (1.0, 0.0, 10.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'switch': (2, None, None, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = vwdemOp
    def __init__(self, Tmax, beta, inv_slope, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class vwdem(SpectralModel):
    def __init__(self, Tmax=None, beta=None, inv_slope=None, nH=None, He=None, C=None, N=None, O=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, S=None, Ar=None, Ca=None, Fe=None, Ni=None, Redshift=None, switch=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([vwdemComponent(Tmax, beta, inv_slope, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch, norm, name, grad_method, eps)])


class wabsOp(XspecNumericGradOp):
    modname = 'wabs'
    optype = 'mul'

class wabsComponent(SpectralComponent):
    _comp_name = 'wabs'
    _config = {'nH': (1.0, 0.0, 1000000.0, False, False)}
    _op_class = wabsOp
    def __init__(self, nH, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class wabs(SpectralModel):
    def __init__(self, nH=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([wabsComponent(nH, name, grad_method, eps)])


class wdemOp(XspecNumericGradOp):
    modname = 'wdem'
    optype = 'add'

class wdemComponent(SpectralComponent):
    _comp_name = 'wdem'
    _config = {'Tmax': (1.0, 0.01, 20.0, False, False), 'beta': (0.1, 0.01, 1.0, False, False), 'inv_slope': (0.25, -1.0, 10.0, False, False), 'nH': (1.0, 1e-06, 1e+20, True, False), 'abundanc': (1.0, 0.0, 10.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'switch': (2, None, None, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = wdemOp
    def __init__(self, Tmax, beta, inv_slope, nH, abundanc, Redshift, switch, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class wdem(SpectralModel):
    def __init__(self, Tmax=None, beta=None, inv_slope=None, nH=None, abundanc=None, Redshift=None, switch=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([wdemComponent(Tmax, beta, inv_slope, nH, abundanc, Redshift, switch, norm, name, grad_method, eps)])


class wndabsOp(XspecNumericGradOp):
    modname = 'wndabs'
    optype = 'mul'

class wndabsComponent(SpectralComponent):
    _comp_name = 'wndabs'
    _config = {'nH': (1.0, 0.0, 20.0, False, False), 'WindowE': (1.0, 0.03, 20.0, False, False)}
    _op_class = wndabsOp
    def __init__(self, nH, WindowE, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class wndabs(SpectralModel):
    def __init__(self, nH=None, WindowE=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([wndabsComponent(nH, WindowE, name, grad_method, eps)])


class xilconvOp(XspecNumericGradOp):
    modname = 'xilconv'
    optype = 'con'

class xilconvComponent(SpectralComponent):
    _comp_name = 'xilconv'
    _config = {'rel_refl': (-1.0, -1.0, 1000000.0, False, False), 'redshift': (0.0, 0.0, 4.0, True, False), 'Fe_abund': (1.0, 0.5, 3.0, True, False), 'cosIncl': (0.5, 0.05, 0.95, True, False), 'log_xi': (1.0, 1.0, 6.0, False, False), 'cutoff': (300.0, 20.0, 300.0, True, False)}
    _op_class = xilconvOp
    def __init__(self, rel_refl, redshift, Fe_abund, cosIncl, log_xi, cutoff, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class xilconv(SpectralModel):
    def __init__(self, rel_refl=None, redshift=None, Fe_abund=None, cosIncl=None, log_xi=None, cutoff=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([xilconvComponent(rel_refl, redshift, Fe_abund, cosIncl, log_xi, cutoff, name, grad_method, eps)])


class xionOp(XspecNumericGradOp):
    modname = 'xion'
    optype = 'mul'

class xionComponent(SpectralComponent):
    _comp_name = 'xion'
    _config = {'height': (5.0, 0.0, 100.0, False, False), 'lxovrld': (0.3, 0.02, 100.0, False, False), 'rate': (0.05, 0.001, 1.0, False, False), 'cosAng': (0.9, 0.0, 1.0, False, False), 'inner': (3.0, 2.0, 1000.0, False, False), 'outer': (100.0, 2.1, 100000.0, False, False), 'index': (2.0, 1.6, 2.2, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'Feabun': (1.0, 0.0, 5.0, True, False), 'E_cut': (150.0, 20.0, 300.0, False, False), 'Ref_type': (1.0, 1.0, 3.0, True, False), 'Rel_smear': (4.0, 1.0, 4.0, True, False), 'Geometry': (1.0, 1.0, 4.0, True, False)}
    _op_class = xionOp
    def __init__(self, height, lxovrld, rate, cosAng, inner, outer, index, Redshift, Feabun, E_cut, Ref_type, Rel_smear, Geometry, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class xion(SpectralModel):
    def __init__(self, height=None, lxovrld=None, rate=None, cosAng=None, inner=None, outer=None, index=None, Redshift=None, Feabun=None, E_cut=None, Ref_type=None, Rel_smear=None, Geometry=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([xionComponent(height, lxovrld, rate, cosAng, inner, outer, index, Redshift, Feabun, E_cut, Ref_type, Rel_smear, Geometry, name, grad_method, eps)])


class xscatOp(XspecNumericGradOp):
    modname = 'xscat'
    optype = 'mul'

class xscatComponent(SpectralComponent):
    _comp_name = 'xscat'
    _config = {'NH': (1.0, 0.0, 1000.0, False, False), 'Xpos': (0.5, 0.0, 0.999, False, False), 'Rext': (10.0, 0.0, 240.0, True, False), 'DustModel': (1, None, None, True, False)}
    _op_class = xscatOp
    def __init__(self, NH, Xpos, Rext, DustModel, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class xscat(SpectralModel):
    def __init__(self, NH=None, Xpos=None, Rext=None, DustModel=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([xscatComponent(NH, Xpos, Rext, DustModel, name, grad_method, eps)])


class zTBabsOp(XspecNumericGradOp):
    modname = 'zTBabs'
    optype = 'mul'

class zTBabsComponent(SpectralComponent):
    _comp_name = 'zTBabs'
    _config = {'nH': (1.0, 0.0, 1000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False)}
    _op_class = zTBabsOp
    def __init__(self, nH, Redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zTBabs(SpectralModel):
    def __init__(self, nH=None, Redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zTBabsComponent(nH, Redshift, name, grad_method, eps)])


class zagaussOp(XspecNumericGradOp):
    modname = 'zagauss'
    optype = 'add'

class zagaussComponent(SpectralComponent):
    _comp_name = 'zagauss'
    _config = {'LineE': (10.0, 0.0, 1000000.0, False, False), 'Sigma': (1.0, 0.0, 1000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = zagaussOp
    def __init__(self, LineE, Sigma, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zagauss(SpectralModel):
    def __init__(self, LineE=None, Sigma=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zagaussComponent(LineE, Sigma, Redshift, norm, name, grad_method, eps)])


class zashiftOp(XspecNumericGradOp):
    modname = 'zashift'
    optype = 'con'

class zashiftComponent(SpectralComponent):
    _comp_name = 'zashift'
    _config = {'Redshift': (0.0, -0.999, 10.0, True, False)}
    _op_class = zashiftOp
    def __init__(self, Redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zashift(SpectralModel):
    def __init__(self, Redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zashiftComponent(Redshift, name, grad_method, eps)])


class zbabsOp(XspecNumericGradOp):
    modname = 'zbabs'
    optype = 'mul'

class zbabsComponent(SpectralComponent):
    _comp_name = 'zbabs'
    _config = {'nH': (0.0001, 0.0, 1000000.0, False, False), 'nHeI': (1e-05, 0.0, 1000000.0, False, False), 'nHeII': (1e-06, 0.0, 1000000.0, False, False), 'z': (0.0, 0.0, 1000000.0, False, False)}
    _op_class = zbabsOp
    def __init__(self, nH, nHeI, nHeII, z, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zbabs(SpectralModel):
    def __init__(self, nH=None, nHeI=None, nHeII=None, z=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zbabsComponent(nH, nHeI, nHeII, z, name, grad_method, eps)])


class zbbodyOp(XspecNumericGradOp):
    modname = 'zbbody'
    optype = 'add'

class zbbodyComponent(SpectralComponent):
    _comp_name = 'zbbody'
    _config = {'kT': (3.0, 0.0001, 200.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = zbbodyOp
    def __init__(self, kT, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zbbody(SpectralModel):
    def __init__(self, kT=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zbbodyComponent(kT, Redshift, norm, name, grad_method, eps)])


class zbknpowerOp(XspecNumericGradOp):
    modname = 'zbknpower'
    optype = 'add'

class zbknpowerComponent(SpectralComponent):
    _comp_name = 'zbknpower'
    _config = {'PhoIndx1': (1.0, -3.0, 10.0, False, False), 'BreakE': (5.0, 0.0, 1000000.0, False, False), 'PhoIndx2': (2.0, -3.0, 10.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = zbknpowerOp
    def __init__(self, PhoIndx1, BreakE, PhoIndx2, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zbknpower(SpectralModel):
    def __init__(self, PhoIndx1=None, BreakE=None, PhoIndx2=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zbknpowerComponent(PhoIndx1, BreakE, PhoIndx2, Redshift, norm, name, grad_method, eps)])


class zbremssOp(XspecNumericGradOp):
    modname = 'zbremss'
    optype = 'add'

class zbremssComponent(SpectralComponent):
    _comp_name = 'zbremss'
    _config = {'kT': (7.0, 0.0001, 200.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = zbremssOp
    def __init__(self, kT, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zbremss(SpectralModel):
    def __init__(self, kT=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zbremssComponent(kT, Redshift, norm, name, grad_method, eps)])


class zcutoffplOp(XspecNumericGradOp):
    modname = 'zcutoffpl'
    optype = 'add'

class zcutoffplComponent(SpectralComponent):
    _comp_name = 'zcutoffpl'
    _config = {'PhoIndex': (1.0, -3.0, 10.0, False, False), 'HighECut': (15.0, 0.01, 500.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = zcutoffplOp
    def __init__(self, PhoIndex, HighECut, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zcutoffpl(SpectralModel):
    def __init__(self, PhoIndex=None, HighECut=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zcutoffplComponent(PhoIndex, HighECut, Redshift, norm, name, grad_method, eps)])


class zdustOp(XspecNumericGradOp):
    modname = 'zdust'
    optype = 'mul'

class zdustComponent(SpectralComponent):
    _comp_name = 'zdust'
    _config = {'method': (1, None, None, True, False), 'E_BmV': (0.1, 0.0, 100.0, False, False), 'Rv': (3.1, 0.0, 10.0, True, False), 'Redshift': (0.0, 0.0, 20.0, True, False)}
    _op_class = zdustOp
    def __init__(self, method, E_BmV, Rv, Redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zdust(SpectralModel):
    def __init__(self, method=None, E_BmV=None, Rv=None, Redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zdustComponent(method, E_BmV, Rv, Redshift, name, grad_method, eps)])


class zedgeOp(XspecNumericGradOp):
    modname = 'zedge'
    optype = 'mul'

class zedgeComponent(SpectralComponent):
    _comp_name = 'zedge'
    _config = {'edgeE': (7.0, 0.0, 100.0, False, False), 'MaxTau': (1.0, 0.0, 10.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False)}
    _op_class = zedgeOp
    def __init__(self, edgeE, MaxTau, Redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zedge(SpectralModel):
    def __init__(self, edgeE=None, MaxTau=None, Redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zedgeComponent(edgeE, MaxTau, Redshift, name, grad_method, eps)])


class zgaussOp(XspecNumericGradOp):
    modname = 'zgauss'
    optype = 'add'

class zgaussComponent(SpectralComponent):
    _comp_name = 'zgauss'
    _config = {'LineE': (6.5, 0.0, 1000000.0, False, False), 'Sigma': (0.1, 0.0, 20.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = zgaussOp
    def __init__(self, LineE, Sigma, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zgauss(SpectralModel):
    def __init__(self, LineE=None, Sigma=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zgaussComponent(LineE, Sigma, Redshift, norm, name, grad_method, eps)])


class zhighectOp(XspecNumericGradOp):
    modname = 'zhighect'
    optype = 'mul'

class zhighectComponent(SpectralComponent):
    _comp_name = 'zhighect'
    _config = {'cutoffE': (10.0, 0.0001, 200.0, False, False), 'foldE': (15.0, 0.0001, 200.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False)}
    _op_class = zhighectOp
    def __init__(self, cutoffE, foldE, Redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zhighect(SpectralModel):
    def __init__(self, cutoffE=None, foldE=None, Redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zhighectComponent(cutoffE, foldE, Redshift, name, grad_method, eps)])


class zigmOp(XspecNumericGradOp):
    modname = 'zigm'
    optype = 'mul'

class zigmComponent(SpectralComponent):
    _comp_name = 'zigm'
    _config = {'redshift': (0.0, None, None, True, False), 'model': (0, None, None, True, False), 'lyman_limit': (1, None, None, True, False)}
    _op_class = zigmOp
    def __init__(self, redshift, model, lyman_limit, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zigm(SpectralModel):
    def __init__(self, redshift=None, model=None, lyman_limit=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zigmComponent(redshift, model, lyman_limit, name, grad_method, eps)])


class zkerrbbOp(XspecNumericGradOp):
    modname = 'zkerrbb'
    optype = 'add'

class zkerrbbComponent(SpectralComponent):
    _comp_name = 'zkerrbb'
    _config = {'eta': (0.0, 0.0, 1.0, True, False), 'a': (0.5, -0.99, 0.9999, False, False), 'i': (30.0, 0.0, 85.0, True, False), 'Mbh': (10000000.0, 3.0, 10000000000.0, False, False), 'Mdd': (1.0, 1e-05, 100000.0, False, False), 'z': (0.01, 0.0, 10.0, True, False), 'fcol': (2.0, -100.0, 100.0, True, False), 'rflag': (1, None, None, True, False), 'lflag': (1, None, None, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = zkerrbbOp
    def __init__(self, eta, a, i, Mbh, Mdd, z, fcol, rflag, lflag, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zkerrbb(SpectralModel):
    def __init__(self, eta=None, a=None, i=None, Mbh=None, Mdd=None, z=None, fcol=None, rflag=None, lflag=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zkerrbbComponent(eta, a, i, Mbh, Mdd, z, fcol, rflag, lflag, norm, name, grad_method, eps)])


class zlogparOp(XspecNumericGradOp):
    modname = 'zlogpar'
    optype = 'add'

class zlogparComponent(SpectralComponent):
    _comp_name = 'zlogpar'
    _config = {'alpha': (1.5, 0.0, 4.0, False, False), 'beta': (0.2, -4.0, 4.0, False, False), 'pivotE': (1.0, None, None, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = zlogparOp
    def __init__(self, alpha, beta, pivotE, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zlogpar(SpectralModel):
    def __init__(self, alpha=None, beta=None, pivotE=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zlogparComponent(alpha, beta, pivotE, Redshift, norm, name, grad_method, eps)])


class zmshiftOp(XspecNumericGradOp):
    modname = 'zmshift'
    optype = 'con'

class zmshiftComponent(SpectralComponent):
    _comp_name = 'zmshift'
    _config = {'Redshift': (0.0, -0.999, 10.0, True, False)}
    _op_class = zmshiftOp
    def __init__(self, Redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zmshift(SpectralModel):
    def __init__(self, Redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zmshiftComponent(Redshift, name, grad_method, eps)])


class zpcfabsOp(XspecNumericGradOp):
    modname = 'zpcfabs'
    optype = 'mul'

class zpcfabsComponent(SpectralComponent):
    _comp_name = 'zpcfabs'
    _config = {'nH': (1.0, 0.0, 1000000.0, False, False), 'CvrFract': (0.5, 0.0, 1.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False)}
    _op_class = zpcfabsOp
    def __init__(self, nH, CvrFract, Redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zpcfabs(SpectralModel):
    def __init__(self, nH=None, CvrFract=None, Redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zpcfabsComponent(nH, CvrFract, Redshift, name, grad_method, eps)])


class zphabsOp(XspecNumericGradOp):
    modname = 'zphabs'
    optype = 'mul'

class zphabsComponent(SpectralComponent):
    _comp_name = 'zphabs'
    _config = {'nH': (1.0, 0.0, 1000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False)}
    _op_class = zphabsOp
    def __init__(self, nH, Redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zphabs(SpectralModel):
    def __init__(self, nH=None, Redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zphabsComponent(nH, Redshift, name, grad_method, eps)])


class zpowerlwOp(XspecNumericGradOp):
    modname = 'zpowerlw'
    optype = 'add'

class zpowerlwComponent(SpectralComponent):
    _comp_name = 'zpowerlw'
    _config = {'PhoIndex': (1.0, -3.0, 10.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False), 'norm': (1, 1e-10, 10000000000.0, False, False)}
    _op_class = zpowerlwOp
    def __init__(self, PhoIndex, Redshift, norm, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zpowerlw(SpectralModel):
    def __init__(self, PhoIndex=None, Redshift=None, norm=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zpowerlwComponent(PhoIndex, Redshift, norm, name, grad_method, eps)])


class zreddenOp(XspecNumericGradOp):
    modname = 'zredden'
    optype = 'mul'

class zreddenComponent(SpectralComponent):
    _comp_name = 'zredden'
    _config = {'E_BmV': (0.05, 0.0, 10.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False)}
    _op_class = zreddenOp
    def __init__(self, E_BmV, Redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zredden(SpectralModel):
    def __init__(self, E_BmV=None, Redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zreddenComponent(E_BmV, Redshift, name, grad_method, eps)])


class zsmdustOp(XspecNumericGradOp):
    modname = 'zsmdust'
    optype = 'mul'

class zsmdustComponent(SpectralComponent):
    _comp_name = 'zsmdust'
    _config = {'E_BmV': (0.1, 0.0, 100.0, False, False), 'ExtIndex': (1.0, -10.0, 10.0, False, False), 'Rv': (3.1, 0.0, 10.0, True, False), 'redshift': (0.0, 0.0, 20.0, True, False)}
    _op_class = zsmdustOp
    def __init__(self, E_BmV, ExtIndex, Rv, redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zsmdust(SpectralModel):
    def __init__(self, E_BmV=None, ExtIndex=None, Rv=None, redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zsmdustComponent(E_BmV, ExtIndex, Rv, redshift, name, grad_method, eps)])


class zvarabsOp(XspecNumericGradOp):
    modname = 'zvarabs'
    optype = 'mul'

class zvarabsComponent(SpectralComponent):
    _comp_name = 'zvarabs'
    _config = {'H': (1.0, 0.0, 10000.0, True, False), 'He': (1.0, 0.0, 10000.0, True, False), 'C': (1.0, 0.0, 10000.0, True, False), 'N': (1.0, 0.0, 10000.0, True, False), 'O': (1.0, 0.0, 10000.0, True, False), 'Ne': (1.0, 0.0, 10000.0, True, False), 'Na': (1.0, 0.0, 10000.0, True, False), 'Mg': (1.0, 0.0, 10000.0, True, False), 'Al': (1.0, 0.0, 10000.0, True, False), 'Si': (1.0, 0.0, 10000.0, True, False), 'S': (1.0, 0.0, 10000.0, True, False), 'Cl': (1.0, 0.0, 10000.0, True, False), 'Ar': (1.0, 0.0, 10000.0, True, False), 'Ca': (1.0, 0.0, 10000.0, True, False), 'Cr': (1.0, 0.0, 10000.0, True, False), 'Fe': (1.0, 0.0, 10000.0, True, False), 'Co': (1.0, 0.0, 10000.0, True, False), 'Ni': (1.0, 0.0, 10000.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False)}
    _op_class = zvarabsOp
    def __init__(self, H, He, C, N, O, Ne, Na, Mg, Al, Si, S, Cl, Ar, Ca, Cr, Fe, Co, Ni, Redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zvarabs(SpectralModel):
    def __init__(self, H=None, He=None, C=None, N=None, O=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, S=None, Cl=None, Ar=None, Ca=None, Cr=None, Fe=None, Co=None, Ni=None, Redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zvarabsComponent(H, He, C, N, O, Ne, Na, Mg, Al, Si, S, Cl, Ar, Ca, Cr, Fe, Co, Ni, Redshift, name, grad_method, eps)])


class zvfeabsOp(XspecNumericGradOp):
    modname = 'zvfeabs'
    optype = 'mul'

class zvfeabsComponent(SpectralComponent):
    _comp_name = 'zvfeabs'
    _config = {'nH': (1.0, 0.0, 1000000.0, False, False), 'metals': (1.0, 0.0, 100.0, False, False), 'FEabun': (1.0, 0.0, 100.0, False, False), 'FEKedge': (7.11, 7.0, 9.5, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False)}
    _op_class = zvfeabsOp
    def __init__(self, nH, metals, FEabun, FEKedge, Redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zvfeabs(SpectralModel):
    def __init__(self, nH=None, metals=None, FEabun=None, FEKedge=None, Redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zvfeabsComponent(nH, metals, FEabun, FEKedge, Redshift, name, grad_method, eps)])


class zvphabsOp(XspecNumericGradOp):
    modname = 'zvphabs'
    optype = 'mul'

class zvphabsComponent(SpectralComponent):
    _comp_name = 'zvphabs'
    _config = {'nH': (1.0, 0.0, 1000000.0, False, False), 'He': (1.0, 0.0, 1000.0, True, False), 'C': (1.0, 0.0, 1000.0, True, False), 'N': (1.0, 0.0, 1000.0, True, False), 'O': (1.0, 0.0, 1000.0, True, False), 'Ne': (1.0, 0.0, 1000.0, True, False), 'Na': (1.0, 0.0, 1000.0, True, False), 'Mg': (1.0, 0.0, 1000.0, True, False), 'Al': (1.0, 0.0, 1000.0, True, False), 'Si': (1.0, 0.0, 1000.0, True, False), 'S': (1.0, 0.0, 1000.0, True, False), 'Cl': (1.0, 0.0, 1000.0, True, False), 'Ar': (1.0, 0.0, 1000.0, True, False), 'Ca': (1.0, 0.0, 1000.0, True, False), 'Cr': (1.0, 0.0, 1000.0, True, False), 'Fe': (1.0, 0.0, 1000.0, True, False), 'Co': (1.0, 0.0, 1000.0, True, False), 'Ni': (1.0, 0.0, 1000.0, True, False), 'Redshift': (0.0, -0.999, 10.0, True, False)}
    _op_class = zvphabsOp
    def __init__(self, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Cl, Ar, Ca, Cr, Fe, Co, Ni, Redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zvphabs(SpectralModel):
    def __init__(self, nH=None, He=None, C=None, N=None, O=None, Ne=None, Na=None, Mg=None, Al=None, Si=None, S=None, Cl=None, Ar=None, Ca=None, Cr=None, Fe=None, Co=None, Ni=None, Redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zvphabsComponent(nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Cl, Ar, Ca, Cr, Fe, Co, Ni, Redshift, name, grad_method, eps)])


class zwabsOp(XspecNumericGradOp):
    modname = 'zwabs'
    optype = 'mul'

class zwabsComponent(SpectralComponent):
    _comp_name = 'zwabs'
    _config = {'nH': (1.0, 0.0, 1000000.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False)}
    _op_class = zwabsOp
    def __init__(self, nH, Redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zwabs(SpectralModel):
    def __init__(self, nH=None, Redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zwabsComponent(nH, Redshift, name, grad_method, eps)])


class zwndabsOp(XspecNumericGradOp):
    modname = 'zwndabs'
    optype = 'mul'

class zwndabsComponent(SpectralComponent):
    _comp_name = 'zwndabs'
    _config = {'nH': (1.0, 0.0, 20.0, False, False), 'WindowE': (1.0, 0.03, 20.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False)}
    _op_class = zwndabsOp
    def __init__(self, nH, WindowE, Redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zwndabs(SpectralModel):
    def __init__(self, nH=None, WindowE=None, Redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zwndabsComponent(nH, WindowE, Redshift, name, grad_method, eps)])


class zxipabOp(XspecNumericGradOp):
    modname = 'zxipab'
    optype = 'mul'

class zxipabComponent(SpectralComponent):
    _comp_name = 'zxipab'
    _config = {'nHmin': (0.01, 1e-07, 1000000.0, False, False), 'nHmax': (10.0, 1e-07, 1000000.0, False, False), 'beta': (0.0, -10.0, 10.0, False, False), 'log_xi': (3.0, -3.0, 6.0, False, False), 'redshift': (0.0, 0.0, 10.0, True, False)}
    _op_class = zxipabOp
    def __init__(self, nHmin, nHmax, beta, log_xi, redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zxipab(SpectralModel):
    def __init__(self, nHmin=None, nHmax=None, beta=None, log_xi=None, redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zxipabComponent(nHmin, nHmax, beta, log_xi, redshift, name, grad_method, eps)])


class zxipcfOp(XspecNumericGradOp):
    modname = 'zxipcf'
    optype = 'mul'

class zxipcfComponent(SpectralComponent):
    _comp_name = 'zxipcf'
    _config = {'Nh': (10.0, 0.05, 500.0, False, False), 'log_xi': (3.0, -3.0, 6.0, False, False), 'CvrFract': (0.5, 0.0, 1.0, False, False), 'Redshift': (0.0, -0.999, 10.0, True, False)}
    _op_class = zxipcfOp
    def __init__(self, Nh, log_xi, CvrFract, Redshift, name, grad_method, eps):
        kwargs = {k: v for k, v in locals().items() if k not in ('self', '__class__')}
        super().__init__(**kwargs)

class zxipcf(SpectralModel):
    def __init__(self, Nh=None, log_xi=None, CvrFract=None, Redshift=None, name=None, grad_method='f', eps=1e-7):
        super().__init__([zxipcfComponent(Nh, log_xi, CvrFract, Redshift, name, grad_method, eps)])
