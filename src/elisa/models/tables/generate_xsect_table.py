"""Generate photon cross-sections of XSPEC absorption models."""

import numpy as np
from mxspec import callModelFunction
from xspec import Xset


def get_cross_sections(
    energy: np.ndarray,
    model: str,
    abund: str = 'angr',
    xsect: str = 'vern',
) -> np.ndarray:
    """Get photon cross-section from XSPEC.

    Parameters
    ----------
    energy : ndarray
        Photon energy grids to calculate cross-sections.
    model : str
        XSPEC Model name.
    abund : {'angr', 'aspl', 'feld', 'aneb', 'grsa', 'wilm', 'lodd'}
        XSPEC abundance table name.
    abund : {'bcmc', 'obcm', 'vern'}
        XSPEC photoelectric absorption cross-sections name.

    Returns
    -------
    ndarray
        The photon cross-sections.
    """
    Xset.abund = abund
    Xset.xsect = xsect
    egrid = np.column_stack([energy, energy])
    abs_model = []
    callModelFunction(model, egrid.ravel().tolist(), [0.001], abs_model)
    sigma_ = np.array(-1000.0 * np.log(abs_model[::2]))
    inv_sigma = 1.0 / sigma_
    sigma = np.empty(energy.size)
    for i, e, inv_sigma_e in zip(range(energy.size), egrid, inv_sigma):
        callModelFunction(model, e.tolist(), [inv_sigma_e], m := [])
        sigma[i] = -np.log(m[0]) * sigma_[i]
    return sigma


def get_wabs_cross_sections(energy: np.ndarray) -> np.ndarray:
    delta = 1e-6
    egrid = np.column_stack([energy - delta, energy + delta])
    abs_model = []
    callModelFunction('wabs', egrid.ravel().tolist(), [0.001], abs_model)
    sigma_ = np.array(-1000.0 * np.log(abs_model[::2]))
    inv_sigma = 1.0 / sigma_
    sigma = np.empty(energy.size)
    for i, e, inv_sigma_e in zip(range(energy.size), egrid, inv_sigma):
        callModelFunction('wabs', e.tolist(), [inv_sigma_e], m := [])
        sigma[i] = -np.log(m[0]) * sigma_[i]
    return sigma


if __name__ == '__main__':
    # energy grid of the table
    egrid = np.geomspace(0.1, 20.0, 10000)

    # import matplotlib.pyplot as plt

    # increase resolution for tbabs
    extra = []
    for i, j in [[3123, 3155], [3182, 3219], [3202, 3215]]:
        extra.append(np.geomspace(egrid[i], egrid[j], (j - i) * 5 + 1))
    egrid = np.unique(np.hstack([egrid, *extra]))

    # XSPEC abundance and cross-section
    abund_list = [
        'angr',
        'aspl',
        'feld',
        'aneb',
        'grsa',
        'wilm',
        'lodd',
        'lpgp',
        'lpgs',
    ]
    xsect_list = ['bcmc', 'obcm', 'vern']

    # generate phabs xsect table
    phabs_xsect = {
        (xsect, abund): get_cross_sections(egrid, 'phabs', abund, xsect)
        for xsect in xsect_list
        for abund in abund_list
    }

    # generate tbabs xsect table
    tbabs_xsect = {
        abund: get_cross_sections(egrid, 'tbabs', abund, 'vern')
        for abund in abund_list
    }

    # generate wabs xsect table
    wabs_xsect = get_wabs_cross_sections(egrid)

    # write to files
    import h5py

    with h5py.File('xsect.hdf5', 'w') as file:
        file.create_dataset(
            name='energy',
            shape=egrid.shape,
            dtype=np.float32,
            data=egrid,
        )

        for xsect in xsect_list:
            for abund in abund_list:
                file.create_dataset(
                    f'phabs/{xsect}/{abund}',
                    shape=egrid.shape,
                    dtype=np.float32,
                    data=phabs_xsect[(xsect, abund)],
                )

        for abund in abund_list:
            file.create_dataset(
                f'tbabs/vern/{abund}',
                shape=egrid.shape,
                dtype=np.float32,
                data=tbabs_xsect[abund],
            )

        file.create_dataset(
            'wabs/wabs/aneb',
            shape=egrid.shape,
            dtype=np.float32,
            data=wabs_xsect,
        )
