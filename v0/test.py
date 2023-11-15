import os
import matplotlib.pyplot as plt
import numpy as np
import pickle

from elisa import Data, Infer, BlackBodyRad, Comptonized, CutoffPowerlaw, \
    OTTB, Powerlaw, xs, UniformParameter, EnergyFlux
from pyda.stats.significance import significance_lima


data_path = '/hxmt/work/USERS/xuewc/SGR1935'
result_path = '/hxmt/work/USERS/xuewc/SGR1935/analysis_mcmc'

obsid_list = [
    'P051435700205', 'P051435700211', 'P051435700212', 'P051435700301',
    'P051435700302', 'P051435700303', 'P051435700304', 'P051435700305',
    'P051435700312', 'P051435700401', 'P051435700402', 'P051435700403',
    'P051435700409', 'P051435700502', 'P051435700503', 'P051435700510',
    'P051435700611'
]
n_burst = [1, 4, 3, 1, 1, 7, 15, 2, 1, 5, 2, 1, 1, 1, 1, 1, 1]

src_models = {
    'BB': EnergyFlux(1.5, 250) * BlackBodyRad(norm=1),
    'PL': EnergyFlux(1.5, 250) * Powerlaw(norm=1),
    'OTTB': EnergyFlux(1.5, 250) * OTTB(norm=1),
    'Compt': EnergyFlux(1.5, 250) * Comptonized(norm=1),
    # 'CPL': EnergyFlux(1.5, 250) * CutoffPowerlaw(),
    'PL+BB': EnergyFlux(1.5, 250) * Powerlaw(norm=1) + EnergyFlux(1.5,
                                                                  250) * BlackBodyRad(
        norm=1),
    'BB+BB': EnergyFlux(1.5, 250) * BlackBodyRad(norm=1) + EnergyFlux(1.5,
                                                                      250) * BlackBodyRad(
        norm=1),
}
src_models['BB'].BBrad.norm.log = True
src_models['BB'].BBrad.kT.log = True

src_models['PL'].PL.norm.log = True

src_models['OTTB'].OTTB.kT.log = True
src_models['OTTB'].OTTB.norm.log = True

src_models['Compt'].Compt.norm.log = True
src_models['Compt'].Compt.Ep.log = True

# src_models['CPL'].CPL.norm.log = True
# src_models['CPL'].CPL.Ec.log = True

src_models['PL+BB'].BBrad.kT.log = True
src_models['PL+BB'].BBrad.norm.log = True
src_models['PL+BB'].PL.norm.log = True

src_models['BB+BB'].BBrad.kT.log = True
src_models['BB+BB'].BBrad.norm.log = True

f = UniformParameter('f', 0.5, 0.001, 0.999, log=True)
src_models['BB+BB'].BBrad_2.kT = src_models['BB+BB'].BBrad.kT * f
# src_models['BB+BB'].BBrad_2.kT.default = src_models['BB+BB'].BBrad.kT.default + 30.0
# src_models['BB+BB'].BBrad_2.kT.log = True
# src_models['BB+BB'].BBrad_2.norm.log = True

group_type = 'bmin'
group_scale = 25
wabs = xs.wabs(2.79)

for m in list(src_models.keys()):
    src = src_models[m]
    res_i = []
    print(m)
    # if not os.path.exists(f'/Users/xuewc/BurstData/SGRJ1935/analysis/{src.expression}'):
    #     os.mkdir(f'/Users/xuewc/BurstData/SGRJ1935/analysis/{src.expression}')
    for obsid, n in zip(obsid_list, n_burst):
        for i in range(1, n + 1):
            # if (i == 9 or i == 10) and m == 'PL+BB':
            #     continue
            data_list = []
            model_list = []
            sig_lm = np.array(['X.X'] * 3)
            net_counts = np.array(['X.X'] * 3)
            if os.path.exists(f'{data_path}/{obsid}/LE_pha{i}_g0_0-94.pha'):
                d = Data([1.5, 10.0],
                         f'{data_path}/{obsid}/LE_pha{i}_g0_0-94.pha',
                         f'{data_path}/{obsid}/LE_phabkg{i}_g0_0-94.pha',
                         f'{data_path}/{obsid}/LE_rsp{i}.fits',
                         name='LE',
                         group_type=group_type,
                         group_scale=group_scale)
                data_list.append(d)
                model_list.append(wabs * src)
                snr = significance_lima(d.spec_counts.sum(),
                                        d.back_counts.sum(),
                                        d.spec_exposure / d.back_exposure)
                sig_lm[0] = str(np.round(snr, 1))
                net_counts[0] = str(np.round(d.net_counts.sum(), 2))
            else:
                sig_lm[0] = 'X'
                net_counts[0] = 'X'

            if os.path.exists(f'{data_path}/{obsid}/ME_pha{i}_g0_0-53.pha') \
                    and os.path.exists(f'{data_path}/{obsid}/ME_rsp{i}.fits'):
                d = Data([8.0, 35.0],
                         f'{data_path}/{obsid}/ME_pha{i}_g0_0-53.pha',
                         f'{data_path}/{obsid}/ME_phabkg{i}_g0_0-53.pha',
                         f'{data_path}/{obsid}/ME_rsp{i}.fits',
                         name='ME',
                         group_type=group_type,
                         group_scale=group_scale)
                data_list.append(d)
                model_list.append(src)
                snr = significance_lima(d.spec_counts.sum(),
                                        d.back_counts.sum(),
                                        d.spec_exposure / d.back_exposure)
                sig_lm[1] = str(np.round(snr, 1))
                net_counts[1] = str(np.round(d.net_counts.sum(), 2))
            else:
                sig_lm[1] = 'X'
                net_counts[1] = 'X'

            if os.path.exists(f'{data_path}/{obsid}/HE_pha{i}_g0_0-12.pha'):
                d = Data([28.0, 250.0],
                         f'{data_path}/{obsid}/HE_pha{i}_g0_0-12.pha',
                         f'{data_path}/{obsid}/HE_phabkg{i}_g0_0-12.pha',
                         f'{data_path}/{obsid}/HE_rsp{i}.fits',
                         name='HE',
                         group_type=group_type,
                         group_scale=group_scale)
                data_list.append(d)
                model_list.append(src)
                snr = significance_lima(d.spec_counts.sum(),
                                        d.back_counts.sum(),
                                        d.spec_exposure / d.back_exposure)
                sig_lm[2] = str(np.round(snr, 1))
                net_counts[2] = str(np.round(d.net_counts.sum(), 2))
            else:
                sig_lm[2] = 'X'
                net_counts[2] = 'X'

            if data_list:
                try:
                    data = data_list
                    model = model_list
                    infer = Infer(data, model, 'wstat', True)

                    mle = infer.mle()
                    infer.mcmc_nuts()
                    infer.ppc()

                    ci_mcmc = infer.ci(method='hdi')

                    pars_info = []
                    for p in infer._rv['name']:
                        info = [*mle['pars'][p]]
                        info.extend(ci_mcmc[p][1:])
                        pars_info.extend(
                            [i if not np.isnan(i) else 0.0 for i in info])

                    idx = np.append(0, np.flatnonzero(sig_lm != 'X') + 1)
                    p_value = np.array(['X.XXX'] * 4)
                    p_res = np.append(infer._ppc_result['p_value']['total'],
                                      infer._ppc_result['p_value']['group'])
                    p_value[idx] = np.round(p_res, 3).astype(str)
                    p_value[p_value == 'X.XXX'] = 'X'
                    res_i.append(
                        [f'{obsid}_{i:02d}',
                         *pars_info, np.round(mle['stat'], 2),
                         mle['dof'], np.round(mle['stat'] / mle['dof'], 2),
                         *p_value,
                         *sig_lm,
                         *net_counts,
                         np.round(mle['aic'], 2),
                         np.round(mle['bic'], 2)]
                    )

                    if not os.path.exists(
                            f'{result_path}/{obsid}_{i:02d}'):
                        os.mkdir(
                            f'{result_path}/{obsid}_{i:02d}')
                    infer.plot_data(show_pars=infer._rv['name'],
                                    fig_path=f'{result_path}/{obsid}_{i:02d}/{m}_{group_type}{group_scale}.pdf',
                                    sim='ppc')
                    infer.plot_trace(
                        fig_path=f'{result_path}/{obsid}_{i:02d}/{m}_{group_type}{group_scale}_trace.pdf')
                    try:
                        infer.plot_corner(root=1,
                                          fig_path=f'{result_path}/{obsid}_{i:02d}/{m}_{group_type}{group_scale}_corner.pdf')
                    except:
                        pass
                    plt.close('all')
                    with open(
                            f'{result_path}/{obsid}_{i:02d}/{m}_{group_type}{group_scale}_result.pkl',
                            'wb') as f:
                        mle_res = {k: v for k, v in infer._mle_result.items()
                                   if k != 'minuit'}
                        pickle.dump([mle_res, infer._idata, infer._ppc_result],
                                    f)

                    print(f'{obsid}-{i:02d}', end=', ')

                except Exception as e:
                    print(f'{obsid}-{i:02d}_ERROR: {e}', end=', ')
                    info = ['X'] * (6 * len(infer._rv['name']) + 7)
                    res_i.append(
                        [f'{obsid}_{i:02d}',
                         *info,
                         *sig_lm,
                         *net_counts,
                         'X',
                         'X']
                    )
                    raise e

    print('\n\n\n')
    pars_name = [j for i in infer._rv['name'] for j in
                 [i, 'se', 'lower', 'upper']]
    res_i.insert(0,
                 ['ObsID', *pars_name, 'stat', 'dof', 'stat/dof',
                  'p-value', 'p-value_LE', 'p-value_ME', 'p-value_HE',
                  'SNR_LE', 'SNR_ME', 'SNR_HE',
                  'Net_LE', 'Net_ME', 'Net_HE',
                  'AIC', 'BIC'])
    np.savetxt(
        f'{result_path}/{src.expression}_{group_type}{group_scale}.txt',
        res_i, fmt='%s', delimiter='\t')
