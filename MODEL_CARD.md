# WEAVER model card

## Model

WEAVER is a deterministic global weather forecasting model with Cross-Coupled
Attention and Regional Prior Mixture-of-Experts. This release contains the
single-stage configuration used with 71 ERA5 fields on a
`128 x 256` latitude-longitude grid.

The model predicts normalized state increments conditioned on a base lead time
of 6, 12, or 24 hours.  Medium-range forecasts are produced autoregressively:
the increment is converted to physical units, added to the current state, and
the updated state is normalized before the next step.  Forecasts from every
base lead time that divides the requested target lead time are averaged in
physical space.

## Released checkpoint

`checkpoints/weaver_weights.ckpt` is the released checkpoint for prediction
and evaluation.

The corresponding architecture and variable ordering are fixed by
`configs/weaver.yaml`. Checkpoint loading is strict so that a configuration
or variable mismatch fails rather than silently producing invalid forecasts.

## Training and evaluation data

The paper uses ERA5 at 1.40625 degree resolution:

- training: 2008--2016;
- validation: 2017;
- testing: 2018;
- temporal spacing: 6 hours;
- 69 dynamic fields and 2 static fields.

Normalization statistics and the climatology distributed in
`assets/statistics/` were computed from the training period.  The fixed
nine-region map in `assets/region_maps/` was also constructed only from the
training-period climatology.

The raw ERA5/WeatherBench2 data are not included and must be obtained under
their own data terms; the repository contains only the small derived arrays
needed by the released configuration.

## Intended use and limitations

This model is released for research and reproducibility.  It is not an
operational warning system and should not be used as the sole basis for safety,
emergency, financial, or infrastructure decisions.  Forecast quality may
degrade outside the training distribution, for extreme events, or when inputs
use a different grid, variable order, units, normalization, or timestamp
convention.

`infer.py` is a low-level normalized-increment diagnostic.  Use `predict.py`
for physical-unit autoregressive forecasts and `evaluate.py` for the paper's
test-set metrics.
