import numpy as np
import pytest
from jax.experimental.sparse import BCSR

from elisa.data.grouping import significance_gv, significance_lima
from elisa.data.ogip import Response, ResponseData, SpectrumData
from elisa.models import PowerLaw


@pytest.mark.parametrize(
    'erange',
    [pytest.param([(1, 37), (42, 99)]), pytest.param(None)],
)
def test_data_plot(simulation, erange):
    data = simulation

    if erange is not None:
        data.set_erange(erange)

    # Check the plot methods
    for xlog in [True, False]:
        for data_ylog in [True, False]:
            for sig_ylog in [True, False]:
                data.plot_spec(xlog, data_ylog, sig_ylog)

    for hatch in [True, False]:
        data.plot_matrix(hatch)
        for log in [True, False]:
            data.plot_effective_area(hatch, log)


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

    scale = 6
    nchan = data.channel.size
    data.group('const', scale)
    assert data.channel.size == nchan // scale

    scale = 1
    data.group('min', scale)
    assert np.all(data.spec_counts >= scale)

    scale = 1
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
    assert data.channel.size == nbins

    scale = 1
    data.group('optmin', scale)
    assert np.all(data.spec_counts >= scale)

    scale = 10
    data.group('optbmin', scale)
    assert np.all(data.back_counts >= scale)

    scale = 1
    data.group('optsig', scale)
    sig = significance_lima(
        data.spec_counts, data.back_counts, data.back_ratio
    )
    assert np.all(sig >= scale)

    scale = 1
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

    scale = 1
    data.group('sig', scale)
    sig = significance_gv(
        data.spec_counts, data.back_counts, data.back_errors, data.back_ratio
    )
    assert np.all(sig >= scale)

    scale = 1
    data.group('optbsig', scale)
    sig = significance_gv(
        data.spec_counts, data.back_counts, data.back_errors, data.back_ratio
    )
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


@pytest.mark.parametrize(
    'file',
    [
        'docs/notebooks/data/P011160500104_LE.rsp',
        'docs/notebooks/data/P011160500104_ME.rsp',
        'docs/notebooks/data/P011160500104_HE.rsp',
    ],
)
def test_load_response(file):
    # test Response against big-endian files
    rsp = Response(file)
    # test if the response matrix can be converted to a BCSR matrix in JAX
    assert np.all(rsp.channel_fwhm > 0)
    assert np.array_equal(
        rsp.matrix,
        BCSR.from_scipy_sparse(rsp.sparse_matrix).todense(),
    )


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
