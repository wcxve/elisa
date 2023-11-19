if __name__ == '__main__':
    # NaI_data = Data(
    #     [28, 250],
    #     '/Users/xuewc/BurstData/EXO2030+375/upper_limit/NaI_TOTAL.fits{2}',
    #     '/Users/xuewc/BurstData/EXO2030+375/upper_limit/NaI_BKG.fits{2}',
    #     '/Users/xuewc/BurstData/EXO2030+375/upper_limit/HE_rsp.rsp',
    #     name='NaI'
    # )
    # CsI_data = Data(
    #     [200, 600],
    #     '/Users/xuewc/BurstData/EXO2030+375/upper_limit/CsI_TOTAL.fits{2}',
    #     '/Users/xuewc/BurstData/EXO2030+375/upper_limit/CsI_BKG.fits{2}',
    #     '/Users/xuewc/BurstData/EXO2030+375/upper_limit/DA.rsp',
    #     name='CsI'
    # )
    #
    # from elisa import xs, CutoffPowerlaw, UniformParameter
    # src = CutoffPowerlaw() + xs.posm()
    # # src.CPL.norm.log = True
    # src.CPL.norm = src.posm.norm * UniformParameter('f', 10, 1e-5, 1e5, log=1)
    # src.CPL.Ec.log = True
    # i = Infer([NaI_data, CsI_data], src, ['wstat', 'pgstat'])
    # print(i.mle())
    # i.plot_spec()
    # i.mcmc_nuts(1000, 1000)
    import os
    os.sys.path.append(os.path.abspath('../../'))
    from elisa.v0 import *
    path = '/Users/xuewc/BurstData/FRB221014/HXMT/'
    LE = Data([2, 10], f'{path}/LE_optbmin5.fits',
              f'{path}/LE_phabkg20s_g0_0-94.pha', f'{path}/LE_rsp.rsp',
              group='bmin', scale=25)

    ME = Data([10, 35], f'{path}/ME_optbmin5.fits',
              f'{path}/ME_phabkg20s_g0_0-53.pha', f'{path}/ME_rsp.rsp',
              group='bmin', scale=25)

    HE = Data([28, 250], f'{path}/HE_optbmin5.fits',
              f'{path}/HE_phabkg20s_g0_0-12.pha', f'{path}/HE_rsp.rsp',
              group='bmin', scale=25)


    wabs = xs.wabs(2.79)
    src = [
        xs.powerlaw(),
        Powerlaw(),
        EnergyFlux(1.5, 250) * Powerlaw(norm=1),
        EnergyFlux(1.5, 250) * Powerlaw(norm=1) + EnergyFlux(1.5, 250) * BlackBodyRad(norm=1),
        CutoffPowerlaw()
    ][1]
    # src.CPL.Ec.max = 300
    # src.CPL.Ec.log = True
    # src.CPL.norm.log = True
    # src.PL.norm = src.PL.PhoIndex + src.PL.PhoIndex
    src.PL.norm.log = 1
    # src = BlackBodyRad()
    # src.BBrad.norm.log = 1
    # src.BBrad.kT.log = 1
    # src = OOTB()
    # src.OOTB.kT.log = 1
    # src.OOTB.norm.log = 1
    # src = EnergyFlux(1.5, 250)*BlackBodyRad(norm=1) + EnergyFlux(1.5, 250)*BlackBodyRad(norm=1)
    # src.BBrad_2.kT = src.BBrad.kT * UniformParameter('factor', 0.5, 0.001, 0.999, log=1)
    m = wabs*src
    infer = Infer([LE, ME, HE], m, 'wstat')
    # infer.bootstrap()
    # infer.mcmc_nuts()
    # infer.plot_corner()
    # infer.ppc()
    infer.plot_data('ldata sdev icnt',
                    sim=None,
                    show_pars=infer._rv['name'])

    # test for GRB 230307A
    # path = '/Users/xuewc/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/188ccc99a439bc9cc579a929ee149d49/Message/MessageTemp/f73fb0aee92be46a90a93165d0b6ae4c/OpenData/11668/39e6f9ad32176b0685ae7020e561c41b'
    # data = []
    # for i in [4,5,1,3,12]:
    #     data.extend([
    #         Data([40, 300],
    #              f'{path}/full_spec_bg{i:02d}H_v20230704.FITS{{1}}',
    #              f'{path}/bg_spec_bg{i:02d}H_v20230704.FITS{{1}}',
    #              f'{path}/gbg_{i:02d}H_x_evt_v00.rsp',
    #              name=f'GBG{i:02d}H', is_spec_poisson=True, is_back_poisson=False),
    #         Data([700, 8000] if i != 1 else [700, 7400],
    #              f'{path}/full_spec_bg{i:02d}L_v20230704.FITS{{1}}',
    #              f'{path}/bg_spec_bg{i:02d}L_v20230704.FITS{{1}}',
    #              f'{path}/gbg_{i:02d}L_x_evt_v00.rsp',
    #              name=f'GBG{i:02d}L', is_spec_poisson=True, is_back_poisson=False)
    #     ])
    #
    # data.extend([
    #     Data([[15,35],[40,100]],
    #          f'{path}/GCG01H_195Phases_TOTAL.fits{{56}}',
    #          f'{path}/GCG01H_195Phases_BKG.fits{{56}}',
    #          f'{path}/gcg_01H_x_evt.rsp',
    #          name='GCG01H', is_spec_poisson=True, is_back_poisson=False),
    #     Data([1000,6000],
    #          f'{path}/full_spec_cg01L_v20230704.FITS{{1}}',
    #          f'{path}/bg_spec_cg01L_v20230704.FITS{{1}}',
    #          f'{path}/gcg_01L_x_evt.rsp',
    #          name='GCG01L', is_spec_poisson=True, is_back_poisson=False),
    # ])
    #
    # src = xs.cutoffpl()
    # src.cutoffpl.HighECut.max = 3000
    # # src.CPL.norm.log=1
    # # src.CPL.Ec.log=1
    # from elisa import Constant
    # f1 = Constant()*src
    # f1.constant.name='B05'
    # f2 = Constant()*src
    # f2.constant.name='B01'
    # f3 = Constant()*src
    # f3.constant.name='B03'
    # f4 = Constant()*src
    # f4.constant.name='B12'
    # f5 = Constant()*src
    # f5.constant.name='C01'
    # infer = Infer(data, [src, src,
    #                             f1,f1,f2,f2,f3,f3,f4,f4,f5,f5], 'pgstat')
    # infer.restore('/Users/xuewc/test')
    # plt.close('all')
