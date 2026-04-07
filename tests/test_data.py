import numpy as np
import pytest
from astropy.io import fits
from jax.experimental.sparse import BCSR

import elisa.data.base as data_base
import elisa.data.ogip as ogip_mod
from elisa.data import Data
from elisa.data.base import ObservationData
from elisa.data.grouping import significance_gv, significance_lima
from elisa.data.ogip import Response, ResponseData, Spectrum, SpectrumData
from elisa.models import PowerLaw

CURATED_RESPONSE_CASES = [
    pytest.param(
        'Chandra/ACIS/acisf04487_001N022_r0009_rmf3.fits.gz',
        'Chandra/ACIS/acisf04487_001N022_r0009_arf3.fits.gz',
        id='Chandra/ACIS',
    ),
    # pytest.param(
    #     'Chandra/LETGS/leg_1.rmf.gz',
    #     'Chandra/LETGS/leg_1.arf.gz',
    #     id='Chandra/LETGS',
    # ),
    pytest.param(
        'NuSTAR/FPMA/nu90402339002A01_sr.rmf',
        'NuSTAR/FPMA/nu90402339002A01_sr.arf',
        id='NuSTAR/FPMA',
    ),
    pytest.param(
        'NuSTAR/FPMB/nu90402339002B01_sr.rmf',
        'NuSTAR/FPMB/nu90402339002B01_sr.arf',
        id='NuSTAR/FPMB',
    ),
    pytest.param(
        'XMM-Newton/EPIC-PN/PN.rmf',
        'XMM-Newton/EPIC-PN/PN.arf',
        id='XMM-Newton/EPIC-pn',
    ),
    pytest.param(
        'XMM-Newton/EPIC-MOS1/MOS1.rmf',
        'XMM-Newton/EPIC-MOS1/MOS1.arf',
        id='XMM-Newton/EPIC-MOS1',
    ),
    pytest.param(
        'XMM-Newton/EPIC-MOS2/MOS2.rmf',
        'XMM-Newton/EPIC-MOS2/MOS2.arf',
        id='XMM-Newton/EPIC-MOS2',
    ),
    pytest.param(
        'XMM-Newton/RGS/P0871591801R1S004RSPMAT1003.FIT.gz',
        None,
        id='XMM-Newton/RGS',
    ),
    pytest.param(
        'NICER/XTI/2050300110.rmf',
        'NICER/XTI/2050300110_g2_b_001.arf',
        id='NICER/XTI',
    ),
    pytest.param(
        'XRISM/Resolve/xa_merged_p0px1000_HpS.rmf.gz',
        'XRISM/Resolve/rsl_standard_GVclosed.arf',
        id='XRISM/Resolve',
    ),
    pytest.param(
        'Hitomi/SXS/ah100040040sxs.rmf.gz',
        'Hitomi/SXS/ah100040040sxs.arf.gz',
        id='Hitomi/SXS',
    ),
    pytest.param(
        'Lynx/HDXI/xrs_hdxi.rmf',
        'Lynx/HDXI/xrs_hdxi_3x10.arf',
        id='Lynx/HDXI',
    ),
    pytest.param(
        'IXPE/GPD/ixpe_d1_20170101_alpha075_02.rmf',
        'IXPE/GPD/ixpe_d1_20170101_alpha075_03.arf',
        id='IXPE/GPD',
    ),
    pytest.param('HXMT/LE/CygX-1_LE.rsp', None, id='HXMT/LE'),
    pytest.param('HXMT/ME/CygX-1_ME.rsp', None, id='HXMT/ME'),
    pytest.param('HXMT/HE/CygX-1_HE.rsp', None, id='HXMT/HE'),
]

CURATED_DATA_CASES = [
    pytest.param(
        'NuSTAR/FPMA/nu90402339002A01_sr.pha',
        'NuSTAR/FPMA/nu90402339002A01_bk.pha',
        'NuSTAR/FPMA/nu90402339002A01_sr.rmf',
        'NuSTAR/FPMA/nu90402339002A01_sr.arf',
        id='NuSTAR/FPMA',
    ),
    pytest.param(
        'NuSTAR/FPMB/nu90402339002B01_sr.pha',
        'NuSTAR/FPMB/nu90402339002B01_bk.pha',
        'NuSTAR/FPMB/nu90402339002B01_sr.rmf',
        'NuSTAR/FPMB/nu90402339002B01_sr.arf',
        id='NuSTAR/FPMB',
    ),
    pytest.param(
        'XMM-Newton/EPIC-PN/PN_spectrum_grp20.fits',
        'XMM-Newton/EPIC-PN/PNbackground_spectrum.fits',
        'XMM-Newton/EPIC-PN/PN.rmf',
        'XMM-Newton/EPIC-PN/PN.arf',
        id='XMM-Newton/EPIC-pn',
    ),
    pytest.param(
        'XMM-Newton/EPIC-MOS1/MOS1_spectrum_grp.fits',
        'XMM-Newton/EPIC-MOS1/MOS1background_spectrum.fits',
        'XMM-Newton/EPIC-MOS1/MOS1.rmf',
        'XMM-Newton/EPIC-MOS1/MOS1.arf',
        id='XMM-Newton/EPIC-MOS1',
    ),
    pytest.param(
        'XMM-Newton/EPIC-MOS2/MOS2_spectrum_grp.fits',
        'XMM-Newton/EPIC-MOS2/MOS2background_spectrum.fits',
        'XMM-Newton/EPIC-MOS2/MOS2.rmf',
        'XMM-Newton/EPIC-MOS2/MOS2.arf',
        id='XMM-Newton/EPIC-MOS2',
    ),
    pytest.param(
        'XMM-Newton/RGS/P0871591801R1S004SRSPEC1003.FIT.gz',
        None,
        'XMM-Newton/RGS/P0871591801R1S004RSPMAT1003.FIT.gz',
        None,
        id='XMM-Newton/RGS',
    ),
    pytest.param(
        'NICER/XTI/g2_b_001_raw_opt.pha',
        None,
        'NICER/XTI/2050300110.rmf',
        'NICER/XTI/2050300110_g2_b_001.arf',
        id='NICER/XTI',
    ),
    pytest.param(
        'XRISM/Resolve/xa_merged_p0px1000_Hp.pi.gz',
        None,
        'XRISM/Resolve/xa_merged_p0px1000_HpS.rmf.gz',
        'XRISM/Resolve/rsl_standard_GVclosed.arf',
        id='XRISM/Resolve',
    ),
    pytest.param(
        'Hitomi/SXS/ah100040040sxs_src_grp.pha.gz',
        None,
        'Hitomi/SXS/ah100040040sxs.rmf.gz',
        'Hitomi/SXS/ah100040040sxs.arf.gz',
        id='Hitomi/SXS',
    ),
    pytest.param(
        'Lynx/HDXI/fakeit_lynx.pha',
        None,
        'Lynx/HDXI/xrs_hdxi.rmf',
        'Lynx/HDXI/xrs_hdxi_3x10.arf',
        id='Lynx/HDXI',
    ),
    pytest.param(
        'IXPE/GPD/ixpe_det1_src_I.pha',
        None,
        'IXPE/GPD/ixpe_d1_20170101_alpha075_02.rmf',
        'IXPE/GPD/ixpe_d1_20170101_alpha075_03.arf',
        id='IXPE/GPD-I',
    ),
    pytest.param(
        'IXPE/GPD/ixpe_det1_src_Q.pha',
        None,
        'IXPE/GPD/ixpe_d1_20170101_alpha075_02.rmf',
        'IXPE/GPD/ixpe_d1_20170101_alpha075_03.mrf',
        id='IXPE/GPD-Q',
    ),
    pytest.param(
        'IXPE/GPD/ixpe_det1_src_U.pha',
        None,
        'IXPE/GPD/ixpe_d1_20170101_alpha075_02.rmf',
        'IXPE/GPD/ixpe_d1_20170101_alpha075_03.mrf',
        id='IXPE/GPD-U',
    ),
    pytest.param(
        'HXMT/LE/CygX-1_LE.pha',
        'HXMT/LE/CygX-1_LE_bkg.pha',
        'HXMT/LE/CygX-1_LE.rsp',
        None,
        id='HXMT/LE',
    ),
    pytest.param(
        'HXMT/ME/CygX-1_ME.pha',
        'HXMT/ME/CygX-1_ME_bkg.pha',
        'HXMT/ME/CygX-1_ME.rsp',
        None,
        id='HXMT/ME',
    ),
    pytest.param(
        'HXMT/HE/CygX-1_HE.pha',
        'HXMT/HE/CygX-1_HE_bkg.pha',
        'HXMT/HE/CygX-1_HE.rsp',
        None,
        id='HXMT/HE',
    ),
    # pytest.param(
    #     'Chandra/LETGS/pha2.gz{1}',
    #     'Chandra/LETGS/pha2_bg.gz{1}',
    #     'Chandra/LETGS/leg_1.rmf.gz',
    #     'Chandra/LETGS/leg_1.arf.gz',
    #     id='Chandra/LETGS',
    # ),
]


def _make_observation(
    spec_counts,
    spec_area,
    spec_back,
    *,
    back_counts=None,
    back_area=None,
    back_back=None,
    spec_poisson=False,
    back_poisson=False,
    spec_net=None,
    grouping=None,
    quality=None,
):
    spec_counts = np.asarray(spec_counts, dtype=np.float64)
    nchan = len(spec_counts)
    egrid = np.linspace(1.0, nchan + 1.0, nchan + 1)
    response = ResponseData(
        photon_egrid=egrid,
        channel_emin=egrid[:-1],
        channel_emax=egrid[1:],
        response_matrix=np.eye(nchan),
        channel=np.arange(nchan).astype(str),
    )
    spec_errors = (
        np.sqrt(np.clip(spec_counts, 0.0, None))
        if spec_poisson
        else np.ones(nchan, dtype=np.float64)
    )
    spec = SpectrumData(
        counts=spec_counts,
        errors=spec_errors,
        poisson=spec_poisson,
        exposure=1.0,
        quality=quality,
        grouping=grouping,
        area_scale=spec_area,
        back_scale=spec_back,
        net=spec_net,
    )

    if back_counts is None:
        back = None
    else:
        back_counts = np.asarray(back_counts, dtype=np.float64)
        back_errors = (
            np.sqrt(np.clip(back_counts, 0.0, None))
            if back_poisson
            else np.ones(nchan, dtype=np.float64)
        )
        back = SpectrumData(
            counts=back_counts,
            errors=back_errors,
            poisson=back_poisson,
            exposure=1.0,
            quality=quality,
            grouping=grouping,
            area_scale=back_area,
            back_scale=back_back,
        )

    return ObservationData(
        name='test',
        erange=[(egrid[0], egrid[-1])],
        spec_data=spec,
        resp_data=response,
        back_data=back,
    )


def _write_vector_scale_spectrum(path, *, hduclas2='', backfile=''):
    counts = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    cols = [
        fits.Column(name='COUNTS', format='D', array=counts),
        fits.Column(
            name='QUALITY', format='I', array=np.zeros(3, dtype=np.int16)
        ),
        fits.Column(
            name='GROUPING', format='I', array=np.ones(3, dtype=np.int16)
        ),
        fits.Column(
            name='AREASCAL', format='D', array=np.array([1.0, 2.0, 3.0])
        ),
        fits.Column(
            name='BACKSCAL', format='D', array=np.array([4.0, 5.0, 6.0])
        ),
    ]
    spectrum = fits.BinTableHDU.from_columns(cols, name='SPECTRUM')
    spectrum.header['POISSERR'] = True
    spectrum.header['EXPOSURE'] = 1.0
    spectrum.header['DETCHANS'] = 3
    spectrum.header['DETNAM'] = 'TEST'
    if hduclas2:
        spectrum.header['HDUCLAS2'] = hduclas2
    if backfile:
        spectrum.header['BACKFILE'] = backfile
    fits.HDUList([fits.PrimaryHDU(), spectrum]).writeto(path)


def _make_dummy_response(nchan):
    egrid = np.linspace(1.0, nchan + 1.0, nchan + 1)
    return ResponseData(
        photon_egrid=egrid,
        channel_emin=egrid[:-1],
        channel_emax=egrid[1:],
        response_matrix=np.eye(nchan),
        channel=np.arange(nchan).astype(str),
    )


@pytest.mark.parametrize(
    'erange',
    [pytest.param([(1, 37), (42, 99)]), pytest.param(None)],
)
def test_data_plot(simulation, erange):
    data = simulation

    if erange is not None:
        data.set_erange(erange)

    data.plot_spec(xlog=False, data_ylog=True, sig_ylog=False)
    data.plot_matrix(hatch=True)
    data.plot_effective_area(hatch=False, ylog=True)


@pytest.mark.parametrize(
    'erange',
    [pytest.param([(1, 37), (42, 99)]), pytest.param(None)],
)
def test_data_grouping(erange):
    # Setup simulation configuration
    seed = 42
    emin = 1.0
    emax = 100.0
    nbins = 200
    photon_egrid = np.linspace(emin, emax, nbins + 1)
    channel_emin = photon_egrid[:-1]
    channel_emax = photon_egrid[1:]
    response_matrix = np.eye(nbins)
    spec_exposure = 50.0

    # Setup model and simulate data
    alpha = 0.0
    K = 10.0  # photon flux between emin and emax, when alpha = 0
    model = PowerLaw(K=[K], alpha=alpha)
    compiled_model = model.compile()
    data = compiled_model.simulate(
        photon_egrid=photon_egrid,
        channel_emin=channel_emin,
        channel_emax=channel_emax,
        response_matrix=response_matrix,
        spec_exposure=spec_exposure,
        spec_poisson=True,
        back_counts=np.full(nbins, 10),
        back_exposure=2.0,
        back_poisson=True,
        seed=seed,
    )

    if erange is not None:
        data.set_erange(erange)

    nchan = data.channel.size

    scale = 6
    data.group('const', scale)
    assert data.channel.size == nchan // scale

    scale = 1
    data.group('min', scale)
    assert np.all(data.spec_counts >= scale)

    scale = 2.5
    data.group('sig', scale)
    sig = significance_lima(
        data.spec_counts, data.back_counts, data.back_ratio
    )
    assert np.all(sig >= scale)

    scale = 10
    data.group('bmin', scale)
    assert np.all(data.back_counts >= scale)

    scale = 1
    data.group('bsig', scale)
    sig = data.back_counts / data.back_errors
    assert np.all(sig >= scale)

    data.group('opt')
    assert data.channel.size == nchan

    scale = 1
    data.group('optmin', scale)
    assert np.all(data.spec_counts >= scale)

    scale = 10
    data.group('optbmin', scale)
    assert np.all(data.back_counts >= scale)

    scale = 2.5
    data.group('optsig', scale)
    sig = significance_lima(
        data.spec_counts, data.back_counts, data.back_ratio
    )
    assert np.all(sig >= scale)

    scale = 2.5
    data.group('optbsig', scale)
    sig = data.back_counts / data.back_errors
    assert np.all(sig >= scale)

    # test preserve_data_group=True
    scale = 2
    original_nchan = data.channel.size
    original_grouping = data.grouping
    data.spec_data._grouping = original_grouping
    data.group('const', scale, preserve_data_group=True)
    assert data.channel.size == original_nchan // scale
    assert np.all(data.grouping[original_grouping == -1] == -1)

    with pytest.raises(ValueError):
        data.group('optbsig', scale, preserve_data_group=True)

    data = compiled_model.simulate(
        photon_egrid=photon_egrid,
        channel_emin=channel_emin,
        channel_emax=channel_emax,
        response_matrix=response_matrix,
        spec_exposure=spec_exposure,
        spec_poisson=True,
        back_counts=np.full(nbins, 10),
        back_errors=np.full(nbins, 2),
        back_exposure=2.0,
        back_poisson=False,
        seed=seed,
    )

    if erange is not None:
        data.set_erange(erange)

    scale = 2.5
    data.group('sig', scale)
    sig = significance_gv(
        data.spec_counts, data.back_counts, data.back_errors, data.back_ratio
    )
    assert np.all(sig >= scale)

    scale = 2.5
    data.group('optbsig', scale)
    sig = data.back_counts / data.back_errors
    assert np.all(sig >= scale)

    # test grouping of Poisson spectrum with no background data (Issue #286)
    data = compiled_model.simulate(
        photon_egrid=photon_egrid,
        channel_emin=channel_emin,
        channel_emax=channel_emax,
        response_matrix=response_matrix,
        spec_exposure=spec_exposure,
        spec_poisson=True,
        seed=seed,
    )

    scale = 2.5
    data.group('sig', scale)
    sig = data.net_counts / data.net_errors
    assert np.all(sig >= scale)

    data.group('optsig', scale)
    sig = data.net_counts / data.net_errors
    assert np.all(sig >= scale)


def test_grouping_warning(simulation):
    with pytest.warns(match='reset the first grouping flag from -1 to 1'):
        grouping = simulation.grouping.copy()
        grouping[0] = -1
        simulation.set_grouping(grouping)

    with pytest.warns(match='reset the first grouping flag from -1 to 1'):
        SpectrumData(
            counts=np.ones(2),
            errors=np.ones(2),
            poisson=True,
            exposure=1.0,
            grouping=np.array([-1, 1]),
        )


def test_scale_arrays_and_net_support():
    with pytest.warns(Warning) as record:
        spec = SpectrumData(
            counts=np.array([1.0, 2.0, 3.0]),
            errors=np.ones(3),
            poisson=False,
            exposure=1.0,
            quality=np.array([0, 1, 0]),
            area_scale=np.array([0.0, 0.0, 4.0]),
            back_scale=np.array([2.0, 0.0, 0.0]),
        )
    messages = [str(i.message) for i in record]
    assert any('zero area_scale' in i for i in messages)
    assert any('zero back_scale' in i for i in messages)
    np.testing.assert_allclose(spec.area_scale, np.array([1.0, 0.0, 4.0]))
    np.testing.assert_allclose(spec.back_scale, np.array([2.0, 0.0, 1.0]))

    net_spec = SpectrumData(
        counts=np.array([1.0, -1.0]),
        errors=np.ones(2),
        poisson=False,
        exposure=1.0,
    )
    assert net_spec.net is True

    with pytest.raises(ValueError, match='area_scale must be non-negative'):
        SpectrumData(
            counts=np.ones(2),
            errors=np.ones(2),
            poisson=False,
            exposure=1.0,
            area_scale=np.array([-1.0, 1.0]),
        )


def test_grouped_scales_match_xspec_formulas():
    data = _make_observation(
        spec_counts=[10.0, 20.0],
        spec_area=[2.0, 6.0],
        spec_back=[4.0, 12.0],
        spec_net=False,
    )
    data.set_grouping(np.array([1, -1]))
    np.testing.assert_allclose(data.area_scale, np.array([3.6]))
    np.testing.assert_allclose(data._spec_back_scale, np.array([7.2]))

    net_data = _make_observation(
        spec_counts=[10.0, 20.0],
        spec_area=[2.0, 6.0],
        spec_back=[3.0, 9.0],
        spec_net=True,
    )
    net_data.set_grouping(np.array([1, -1]))
    np.testing.assert_allclose(net_data.area_scale, np.array([4.0]))
    np.testing.assert_allclose(net_data._spec_back_scale, np.array([6.0]))

    fallback = _make_observation(
        spec_counts=[1.0, -3.0],
        spec_area=[1.0, 3.0],
        spec_back=[2.0, 6.0],
        spec_net=False,
    )
    fallback.set_grouping(np.array([1, -1]))
    np.testing.assert_allclose(fallback.area_scale, np.array([1.5]))
    np.testing.assert_allclose(fallback._spec_back_scale, np.array([3.0]))


def test_background_grouped_ratio_uses_non_net_formula():
    data = _make_observation(
        spec_counts=[10.0, 20.0],
        spec_area=[2.0, 6.0],
        spec_back=[3.0, 9.0],
        back_counts=[4.0, 8.0],
        back_area=[2.0, 6.0],
        back_back=[2.0, 8.0],
        spec_net=True,
    )
    data.set_grouping(np.array([1, -1]))
    expected_spec_area = 30.0 / (10.0 / 2.0 + 20.0 / 6.0)
    expected_spec_back = 30.0 / (10.0 / 3.0 + 20.0 / 9.0)
    expected_back_area = 12.0 / (4.0 / 2.0 + 8.0 / 6.0)
    expected_back_back = 12.0 / (4.0 / 2.0 + 8.0 / 8.0)
    expected_ratio = (
        expected_spec_area
        * expected_spec_back
        / (expected_back_area * expected_back_back)
    )
    np.testing.assert_allclose(data.area_scale, np.array([expected_spec_area]))
    np.testing.assert_allclose(
        data._spec_back_scale, np.array([expected_spec_back])
    )
    np.testing.assert_allclose(data.back_ratio, np.array([expected_ratio]))


def test_scalar_scales_group_to_constant_ratio():
    data = _make_observation(
        spec_counts=[10.0, 20.0],
        spec_area=2.0,
        spec_back=3.0,
        back_counts=[4.0, 8.0],
        back_area=5.0,
        back_back=6.0,
        spec_net=True,
    )
    data.set_grouping(np.array([1, -1]))
    np.testing.assert_allclose(data.area_scale, np.array([2.0]))
    np.testing.assert_allclose(data._spec_back_scale, np.array([3.0]))
    np.testing.assert_allclose(data._back_area_scale, np.array([5.0]))
    np.testing.assert_allclose(data._back_back_scale, np.array([6.0]))
    np.testing.assert_allclose(data.back_ratio, np.array([0.2]))


@pytest.mark.parametrize('scale_mode', ['vector', 'scalar'])
@pytest.mark.parametrize(
    ('method', 'back_poisson', 'helper_name'),
    [
        ('sig', True, 'group_sig_lima'),
        ('sig', False, 'group_sig_gv'),
        ('optsig', True, 'group_optsig_lima'),
        ('optsig', False, 'group_optsig_gv'),
    ],
)
def test_significance_grouping_helpers_ignore_net(
    monkeypatch, scale_mode, method, back_poisson, helper_name
):
    if scale_mode == 'vector':
        spec_area = [2.0, 6.0, 10.0]
        spec_back = [3.0, 9.0, 15.0]
        back_area = [2.0, 6.0, 10.0]
        back_back = [2.0, 8.0, 4.0]
    else:
        spec_area = 2.0
        spec_back = 3.0
        back_area = 5.0
        back_back = 6.0

    data = _make_observation(
        spec_counts=[40.0, 50.0, 60.0],
        spec_area=spec_area,
        spec_back=spec_back,
        back_counts=[5.0, 5.0, 5.0],
        back_area=back_area,
        back_back=back_back,
        spec_poisson=True,
        back_poisson=back_poisson,
        spec_net=True,
    )

    original = getattr(data_base, helper_name)
    called = {}

    def wrapper(*args, **kwargs):
        called['has_source_net'] = 'source_net' in kwargs
        return original(*args, **kwargs)

    monkeypatch.setattr(data_base, helper_name, wrapper)
    data.group(method, 0.1)
    assert called['has_source_net'] is False


def test_preserve_grouping_recomputes_ratio():
    grouping = np.array([1, -1, 1, -1])
    data = _make_observation(
        spec_counts=[20.0, 20.0, 20.0, 20.0],
        spec_area=np.ones(4),
        spec_back=np.ones(4),
        back_counts=[10.0, 10.0, 10.0, 10.0],
        back_area=np.ones(4),
        back_back=np.ones(4),
        spec_poisson=True,
        back_poisson=True,
        spec_net=False,
        grouping=grouping,
    )
    data._back_ratio = np.full(data.back_ratio.shape, 1.0e6)
    data.group('sig', 2.5, preserve_data_group=True)
    np.testing.assert_array_equal(data.grouping, grouping)


@pytest.mark.parametrize(
    ('hduclas2', 'expected_net'),
    [('NET', True), ('TOTAL', False), ('', False)],
)
def test_ogip_vector_scales_and_hduclas2(tmp_path, hduclas2, expected_net):
    specfile = tmp_path / 'spec.pha'
    _write_vector_scale_spectrum(specfile, hduclas2=hduclas2)
    spectrum = Spectrum(str(specfile))
    np.testing.assert_allclose(spectrum.area_scale, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(spectrum.back_scale, np.array([4.0, 5.0, 6.0]))
    assert spectrum.net is expected_net


@pytest.mark.parametrize(
    ('hduclas2', 'has_background', 'message'),
    [
        (
            'NET',
            True,
            'marked as NET but background file .* is also provided',
        ),
        (
            'BKG',
            False,
            'spectrum .* is marked as BKG; check whether source and '
            'background files are swapped',
        ),
    ],
)
def test_ogip_spectrum_class_warnings(
    tmp_path, monkeypatch, hduclas2, has_background, message
):
    specfile = tmp_path / 'spec.pha'
    _write_vector_scale_spectrum(specfile, hduclas2=hduclas2)

    kwargs = {
        'erange': [(1.0, 4.0)],
        'specfile': str(specfile),
        'respfile': 'dummy.rsp',
    }
    if has_background:
        backfile = tmp_path / 'back.pha'
        _write_vector_scale_spectrum(backfile, hduclas2='BKG')
        kwargs['backfile'] = str(backfile)

    monkeypatch.setattr(
        ogip_mod,
        'Response',
        lambda respfile, ancrfile, sparse: _make_dummy_response(3),
    )

    with pytest.warns(Warning, match=message):
        Data(**kwargs)


def test_simulate_with_scale_arrays():
    photon_egrid = np.linspace(1.0, 5.0, 5)
    channel_emin = photon_egrid[:-1]
    channel_emax = photon_egrid[1:]
    response_matrix = np.eye(4)
    model = PowerLaw(K=[10.0], alpha=0.0).compile()
    data = model.simulate(
        photon_egrid=photon_egrid,
        channel_emin=channel_emin,
        channel_emax=channel_emax,
        response_matrix=response_matrix,
        spec_exposure=10.0,
        spec_poisson=True,
        back_counts=np.full(4, 5.0),
        back_exposure=5.0,
        back_poisson=True,
        spec_area_scale=np.array([1.0, 2.0, 3.0, 4.0]),
        spec_back_scale=np.array([1.0, 1.0, 2.0, 2.0]),
        back_area_scale=np.array([2.0, 2.0, 4.0, 4.0]),
        back_back_scale=np.array([1.0, 2.0, 1.0, 2.0]),
    )
    np.testing.assert_equal(data.spec_data.area_scale.shape, (4,))
    np.testing.assert_equal(data.back_data.area_scale.shape, (4,))
    data.set_grouping(np.array([1, -1, 1, -1]))
    np.testing.assert_equal(data.area_scale.shape, data.spec_counts.shape)
    np.testing.assert_equal(data.back_ratio.shape, data.back_counts.shape)
    fixed = data.get_fixed_data()
    np.testing.assert_equal(fixed.area_scale.shape, fixed.spec_counts.shape)
    np.testing.assert_equal(fixed.back_ratio.shape, fixed.back_counts.shape)


@pytest.mark.parametrize(
    ('resp_relpath', 'anc_relpath'), CURATED_RESPONSE_CASES
)
def test_load_response_from_curated_data(
    curated_test_data_path, resp_relpath, anc_relpath
):
    # test Response against big-endian files
    rsp = Response(
        str(curated_test_data_path(resp_relpath)),
        None
        if anc_relpath is None
        else str(curated_test_data_path(anc_relpath)),
    )
    # test if the response matrix can be converted to a BCSR matrix in JAX
    assert np.all(rsp.channel_fwhm > 0)
    np.testing.assert_allclose(
        rsp.matrix,
        BCSR.from_scipy_sparse(rsp.sparse_matrix).todense(),
        atol=1.0e-37,
    )


@pytest.mark.parametrize(
    ('spec_relpath', 'back_relpath', 'resp_relpath', 'anc_relpath'),
    CURATED_DATA_CASES,
)
def test_load_data_from_curated_datasets(
    curated_test_data_path,
    spec_relpath,
    back_relpath,
    resp_relpath,
    anc_relpath,
):
    kwargs = {
        'erange': [(1.0, 100.0)],
        'specfile': str(curated_test_data_path(spec_relpath)),
        'respfile': str(curated_test_data_path(resp_relpath)),
    }
    if back_relpath is not None:
        kwargs['backfile'] = str(curated_test_data_path(back_relpath))
    if anc_relpath is not None:
        kwargs['ancrfile'] = str(curated_test_data_path(anc_relpath))
    data = Data(**kwargs)
    assert data.spec_counts.size > 0
    if back_relpath is None:
        assert data.back_counts is None
        assert data.back_ratio is None
    else:
        assert data.back_counts.size > 0
        assert data.back_ratio.shape == data.back_counts.shape
    assert data.area_scale.shape == data.spec_counts.shape


def test_response():
    # test ResponseData against different endianness
    photon_egrid = np.linspace(1.0, 100.0, 101)
    channel = np.arange(100)
    channel_emin = photon_egrid[:-1]
    channel_emax = photon_egrid[1:]
    mat1 = np.eye(100).astype('<f4')
    mat2 = np.eye(100).astype('>f4')
    r1 = ResponseData(photon_egrid, channel_emin, channel_emax, mat1, channel)
    r2 = ResponseData(photon_egrid, channel_emin, channel_emax, mat2, channel)
    # test if the response matrix can be converted to a BCSR matrix in JAX
    for r in [r1, r2]:
        assert np.array_equal(
            r.matrix, BCSR.from_scipy_sparse(r.sparse_matrix).todense()
        )
    assert np.all(r1.channel == r2.channel)
    assert np.all(r1.channel_fwhm == r2.channel_fwhm)
