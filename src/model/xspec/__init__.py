import pathlib

import xspec_models_cxc as _xsmodels

_xsmodels.chatter(0)

abundance = _xsmodels.abundance
cross_section = _xsmodels.cross_section

__all__ = list(_xsmodels.list_models())
__all__.extend(['abundance', 'cross_section', 'all_models'])

all_models = _xsmodels.list_models()

path = pathlib.Path(__file__).parent / 'model.py'

if not path.exists():
    template = """
class {mod_name}Op(XspecNumericGradOp):
    modname = '{mod_name}'
    optype = '{mod_type}'

class {mod_name}ComponentNode(SpectralComponent):
    _comp_name = '{mod_name}'
    _config = {par_config}
    _op_class = {mod_name}Op
    def __init__(self, {pars_list}, name, grad_method, eps):
        kwargs = {{k: v for k, v in locals().items() if k not in ('self', '__class__')}}
        super().__init__(**kwargs)
    
class {mod_name}(SpectralModel):
    def __init__(self, {pars_expr}, name=None, grad_method='f', eps=1e-7):
        super().__init__([{mod_name}ComponentNode({pars_list}, name, grad_method, eps)])
    """
    code = ''
    code += 'from elisa.model.base import SpectralComponent, SpectralModel\n'
    code += 'from elisa.model.xspec.base import XspecNumericGradOp\n'
    code += f'\n__all__ = {all_models}\n'
    for m in _xsmodels.list_models():
        mod_type = _xsmodels.info(m).modeltype.name.lower()
        pars_list = [p.name for p in _xsmodels.info(m).parameters]
        par_config = {
            p.name: (p.default, p.hardmin, p.hardmax, p.frozen, False)
            for p in _xsmodels.info(m).parameters
        }
        if mod_type == 'add':
            pars_list.append('norm')
            par_config['norm'] = (1, 1e-10, 1e10, False, False)
        pars_expr = [f'{p}=None' for p in pars_list]

        code += '\n' + template.format(mod_name=m,
                                       mod_type=mod_type,
                                       par_config=par_config,
                                       pars_name=str(tuple(pars_list)),
                                       pars_expr=', '.join(pars_expr),
                                       pars_list=', '.join(pars_list))
    with path.open('w') as f:
        f.write(code)
    del code, f, m, mod_type, par_config, pars_expr, pars_list, template

del path

from .model import *
