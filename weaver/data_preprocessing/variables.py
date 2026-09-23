"""Canonical data schema used by the published WEAVER model."""

PRESSURE_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

STATIC_VARIABLES = (
    "geopotential_at_surface",
    "land_sea_mask",
)

SINGLE_LEVEL_VARIABLES = (
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
)

PRESSURE_VARIABLES = (
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
)

TIME_VARYING_VARIABLES = SINGLE_LEVEL_VARIABLES + PRESSURE_VARIABLES
SOURCE_VARIABLES = STATIC_VARIABLES + TIME_VARYING_VARIABLES
ALL_CHANNELS = (
    SINGLE_LEVEL_VARIABLES
    + tuple(f"{name}_{level}" for name in PRESSURE_VARIABLES for level in PRESSURE_LEVELS)
    + STATIC_VARIABLES
)

# Common ERA5 short names. Files and output datasets always use canonical names.
VARIABLE_ALIASES = {
    "geopotential_at_surface": ("geopotential_at_surface", "z"),
    "land_sea_mask": ("land_sea_mask", "lsm"),
    "2m_temperature": ("2m_temperature", "t2m"),
    "10m_u_component_of_wind": ("10m_u_component_of_wind", "u10"),
    "10m_v_component_of_wind": ("10m_v_component_of_wind", "v10"),
    "mean_sea_level_pressure": ("mean_sea_level_pressure", "msl"),
    "geopotential": ("geopotential", "z"),
    "u_component_of_wind": ("u_component_of_wind", "u"),
    "v_component_of_wind": ("v_component_of_wind", "v"),
    "temperature": ("temperature", "t"),
    "specific_humidity": ("specific_humidity", "q"),
}

EXPECTED_HEIGHT = 128
EXPECTED_WIDTH = 256
DATA_FREQUENCY_HOURS = 6
TRAIN_YEARS = tuple(range(2008, 2017))
VAL_YEARS = (2017,)
TEST_YEARS = (2018,)
SPLIT_YEARS = {"train": TRAIN_YEARS, "val": VAL_YEARS, "test": TEST_YEARS}


def find_data_variable(dataset, canonical_name):
    """Return a DataArray using either a canonical ERA5 name or its short alias."""
    for candidate in VARIABLE_ALIASES[canonical_name]:
        if candidate in dataset.data_vars:
            return dataset[candidate]
    available = ", ".join(dataset.data_vars)
    raise KeyError(f"{canonical_name!r} not found; available data variables: {available}")


def coordinate_name(obj, candidates):
    """Find a coordinate/dimension name across common ERA5 conventions."""
    for name in candidates:
        if name in obj.coords or name in obj.dims:
            return name
    raise KeyError(f"none of {tuple(candidates)!r} found in coordinates/dimensions")
