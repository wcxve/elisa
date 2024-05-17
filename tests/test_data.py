import numpy as np

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
        seed=seed,
    )

    # Check the plot methods
    data.plot_spec()
    data.plot_effective_area()
    data.plot_matrix()
