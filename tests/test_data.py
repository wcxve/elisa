import numpy as np

from elisa.data.grouping import significance_gv, significance_lima
from elisa.models import PowerLaw


def test_data_plot():
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
        back_counts=np.full(nbins, 5),
        back_exposure=1.0,
        back_poisson=True,
        seed=seed,
    )

    # Check the plot methods
    data.plot_spec()
    data.plot_effective_area()
    data.plot_matrix()


def test_data_grouping():
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

    scale = 6
    data.group('const', scale)
    assert data.channel.size == nbins // scale

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
