"""Data classes for fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import numpy as np
    from scipy.sparse import coo_matrix as coo_array

    from elisa.data.ogip import Data

    NDArray = np.ndarray


class FitData(NamedTuple):
    """Data to fit."""

    name: str
    """Name of the observation data."""

    spec_counts: NDArray
    """Spectrum counts in each measuring channel."""

    spec_error: NDArray
    """Uncertainty of spectrum counts."""

    spec_poisson: bool
    """Whether spectrum data follows counting statistics."""

    spec_exposure: np.float64
    """Spectrum exposure."""

    area_factor: np.float64 | NDArray
    """Area scaling factor."""

    has_back: bool
    """Whether spectrum data includes background."""

    back_counts: NDArray | None
    """Background counts in each measuring channel."""

    back_error: NDArray | None
    """Uncertainty of background counts."""

    back_poisson: bool | None
    """Whether background data follows counting statistics."""

    back_exposure: np.float64 | None
    """Background exposure."""

    back_ratio: np.float64 | NDArray | None
    """Ratio of spectrum to background effective exposure."""

    net_counts: NDArray
    """Net counts in each measuring channel."""

    net_error: NDArray
    """Uncertainty of net counts in each measuring channel."""

    ce: NDArray
    """Net counts per second per keV."""

    ce_error: NDArray
    """Uncertainty of net counts per second per keV."""

    ph_egrid: NDArray
    """Photon energy grid of response matrix."""

    channel: NDArray
    """Measurement channel information."""

    ch_emin: NDArray
    """Left edge of measurement energy grid."""

    ch_emax: NDArray
    """Right edge of measurement energy grid."""

    ch_emid: NDArray
    """Middle of measurement energy grid."""

    ch_width: NDArray
    """Width of measurement energy grid."""

    ch_mean: NDArray
    """Geometric mean of measurement energy grid."""

    ch_error: NDArray
    """Width between left/right and geometric mean of channel grid."""

    resp_matrix: NDArray
    """Response matrix."""

    sparse_resp_matrix: coo_array
    """Sparse response matrix."""

    resp_sparse: bool
    """Whether the response matrix is sparse."""

    @classmethod
    def from_ogip(cls, data: Data):
        """Convert Data to FitData."""
        return cls(
            name=data.name,
            spec_counts=data.spec_counts.copy(),
            spec_error=data.spec_error.copy(),
            spec_poisson=data.spec_poisson,
            spec_exposure=data.spec_exposure,
            area_factor=data.area_factor.copy(),
            has_back=data.has_back,
            back_counts=data.back_counts.copy(),
            back_error=data.back_error.copy(),
            back_poisson=data.back_poisson,
            back_exposure=data.back_exposure,
            back_ratio=data.back_ratio.copy(),
            net_counts=data.net_counts.copy(),
            net_error=data.net_error.copy(),
            ce=data.ce.copy(),
            ce_error=data.ce_error.copy(),
            ph_egrid=data.ph_egrid.copy(),
            channel=data.channel.copy(),
            ch_emin=data.ch_emin.copy(),
            ch_emax=data.ch_emax.copy(),
            ch_emid=data.ch_emid.copy(),
            ch_width=data.ch_width.copy(),
            ch_mean=data.ch_mean.copy(),
            ch_error=data.ch_error.copy(),
            resp_matrix=None if data.resp_sparse else data.resp_matrix.copy(),
            sparse_resp_matrix=data.sparse_resp_matrix.copy(),
            resp_sparse=data.resp_sparse,
        )
