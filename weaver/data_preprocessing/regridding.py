"""Conservative lat/lon regridding utilities.

Adapted from WeatherBench 2:
https://github.com/google-research/weatherbench2/blob/main/weatherbench2/regridding.py
"""
from __future__ import annotations

import dataclasses
import functools
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
import xarray

# NumPy or JAX array.
Array = Union[np.ndarray, jax.Array]


@dataclasses.dataclass(frozen=True)
class Grid:
  """Rectangular grid with longitude and latitude in radians."""

  lon: np.ndarray
  lat: np.ndarray

  @classmethod
  def from_degrees(cls, lon: np.ndarray, lat: np.ndarray) -> Grid:
    """Create a grid from degree coordinates."""
    return cls(np.deg2rad(lon), np.deg2rad(lat))

  @property
  def shape(self) -> tuple[int, int]:
    """Return ``(longitude, latitude)`` sizes."""
    return (len(self.lon), len(self.lat))

  def _to_tuple(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return a hashable coordinate tuple."""
    return tuple(self.lon.tolist()), tuple(self.lat.tolist())

  def __eq__(self, other):  # needed for hashability
    """Compare coordinate values."""
    return isinstance(other, Grid) and self._to_tuple() == other._to_tuple()

  def __hash__(self):
    """Hash coordinate values."""
    return hash(self._to_tuple())


@dataclasses.dataclass(frozen=True)
class Regridder:
  """Base class for rectangular-grid regridders."""

  source: Grid
  target: Grid

  def regrid_array(self, field: Array) -> jax.Array:
    """Regrid an array whose trailing dimensions are ``(lon, lat)``."""
    raise NotImplementedError

  def regrid_dataset(self, dataset: xarray.Dataset) -> xarray.Dataset:
    """Regrid every field in an xarray dataset."""
    # Ensure latitude is increasing.
    if not (dataset['latitude'].diff('latitude') > 0).all():
      # ensure latitude is increasing
      dataset = dataset.isel(latitude=slice(None, None, -1))  # reverse
    assert (dataset['latitude'].diff('latitude') > 0).all()
    # Apply the array regridder to every data variable.
    dataset = xarray.apply_ufunc(
        self.regrid_array,
        dataset,
        dask='parallelized',
        input_core_dims=[['longitude', 'latitude']],
        output_core_dims=[['longitude', 'latitude']],
        exclude_dims={'longitude', 'latitude'},
        dask_gufunc_kwargs={
          'output_sizes': {
            'longitude': self.target.lon.shape[0],
            'latitude': self.target.lat.shape[0],
          }
        },
        output_dtypes=[np.float32],
    )
    # apply_ufunc creates the replacement dimensions but does not preserve
    # the physical coordinates of the target grid.  Assign them explicitly;
    # otherwise a rebuilt NetCDF file can contain integer index coordinates
    # even though its values are on the correct 1.40625-degree grid.
    return dataset.assign_coords(
        latitude=np.rad2deg(self.target.lat),
        longitude=np.rad2deg(self.target.lon),
    )


def _assert_increasing(x: np.ndarray) -> None:
  """Raise if coordinates are not strictly increasing."""
  if not (np.diff(x) > 0).all():
    raise ValueError(f'array is not increasing: {x}')


def _latitude_cell_bounds(x: Array) -> jax.Array:
  """Return latitude cell edges, including the poles."""
  pi_over_2 = jnp.array([np.pi / 2], dtype=x.dtype)
  return jnp.concatenate([-pi_over_2, (x[:-1] + x[1:]) / 2, pi_over_2])


def _latitude_overlap(
    source_points: Array,
    target_points: Array,
) -> jax.Array:
  """Return normalized latitude-cell overlap areas."""
  # Compute source and target cell edges.
  source_bounds = _latitude_cell_bounds(source_points)
  target_bounds = _latitude_cell_bounds(target_points)
  # Compute overlap bounds.
  upper = jnp.minimum(
      target_bounds[1:, jnp.newaxis], source_bounds[jnp.newaxis, 1:]
  )
  lower = jnp.maximum(
      target_bounds[:-1, jnp.newaxis], source_bounds[jnp.newaxis, :-1]
  )
  # Normalized cell area: integral of cos(latitude).
  return (upper > lower) * (jnp.sin(upper) - jnp.sin(lower))


def _conservative_latitude_weights(
    source_points: Array, target_points: Array
) -> jax.Array:
  """Build normalized conservative latitude weights."""
  # Coordinates must be increasing.
  _assert_increasing(source_points)
  _assert_increasing(target_points)
  # Compute overlap areas.
  weights = _latitude_overlap(source_points, target_points)
  # Normalize each target row.
  weights /= jnp.sum(weights, axis=1, keepdims=True)
  assert weights.shape == (target_points.size, source_points.size)
  return weights


def _align_phase_with(x, target, period):
  """Shift periodic coordinates to the phase nearest ``target``."""
  shift_down = x > target + period / 2
  shift_up = x < target - period / 2
  return x + period * shift_up - period * shift_down


def _periodic_upper_bounds(x, period):
  """Return upper cell bounds for periodic coordinates."""
  x_plus = _align_phase_with(jnp.roll(x, -1), x, period)
  return (x + x_plus) / 2


def _periodic_lower_bounds(x, period):
  """Return lower cell bounds for periodic coordinates."""
  x_minus = _align_phase_with(jnp.roll(x, +1), x, period)
  return (x_minus + x) / 2


def _periodic_overlap(x0, x1, y0, y1, period):
  """Return the overlap length of two periodic intervals."""
  # Align the second interval to the first.
  y0 = _align_phase_with(y0, x0, period)
  y1 = _align_phase_with(y1, x0, period)
  # Compute the overlap.
  upper = jnp.minimum(x1, y1)
  lower = jnp.maximum(x0, y0)
  return jnp.maximum(upper - lower, 0)


def _longitude_overlap(
    first_points: Array,
    second_points: Array,
    period: float = 2 * np.pi,
) -> jax.Array:
  """Return periodic longitude-cell overlap lengths."""
  # Normalize coordinates to one period.
  first_points = first_points % period
  # Compute bounds for the first grid.
  first_upper = _periodic_upper_bounds(first_points, period)
  first_lower = _periodic_lower_bounds(first_points, period)

  # Normalize the second grid.
  second_points = second_points % period
  # Compute bounds for the second grid.
  second_upper = _periodic_upper_bounds(second_points, period)
  second_lower = _periodic_lower_bounds(second_points, period)

  # Vectorize all pairwise overlaps.
  return jnp.vectorize(functools.partial(_periodic_overlap, period=period))(
      first_lower[:, jnp.newaxis],
      first_upper[:, jnp.newaxis],
      second_lower[jnp.newaxis, :],
      second_upper[jnp.newaxis, :],
  )


def _conservative_longitude_weights(
    source_points: np.ndarray, target_points: np.ndarray
) -> jax.Array:
  """Build normalized conservative longitude weights."""
  # Coordinates must be increasing.
  _assert_increasing(source_points)
  _assert_increasing(target_points)
  # Compute overlap areas.
  weights = _longitude_overlap(target_points, source_points)
  # Normalize each target row.
  weights /= jnp.sum(weights, axis=1, keepdims=True)
  assert weights.shape == (target_points.size, source_points.size)
  return weights


class ConservativeRegridder(Regridder):
  """Conservative regridder implemented with JAX."""

  @functools.partial(jax.jit, static_argnums=0)
  def _mean(self, field: Array) -> jax.Array:
    """Compute target-cell means for a field shaped ``(..., lon, lat)``."""
    # Compute longitude and latitude weights.
    lon_weights = _conservative_longitude_weights(
        self.source.lon, self.target.lon
    )
    lat_weights = _conservative_latitude_weights(
        self.source.lat, self.target.lat
    )
    # Contract source dimensions with the conservative weights.
    return jnp.einsum(
        'ab,cd,...bd->...ac',
        lon_weights,
        lat_weights,
        field,
        precision='highest',
    )

  @functools.partial(jax.jit, static_argnums=0)
  def _nanmean(self, field: Array) -> jax.Array:
    """Compute cell means while ignoring NaNs."""
    # Mark missing values.
    nulls = jnp.isnan(field)
    # Compute the valid-value sum.
    total = self._mean(jnp.where(nulls, 0, field))
    # Compute the valid-value count.
    count = self._mean(jnp.logical_not(nulls))
    # Zero counts intentionally produce NaN.
    return total / count  # intentionally NaN if count == 0

  # Use NaN-aware means for array regridding.
  regrid_array = _nanmean
