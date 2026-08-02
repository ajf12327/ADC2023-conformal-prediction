# Ariel ADC 2023 Dataset Structure

## Overview


For each planet, the information is spread across several files. The `planet_ID` field is the key that links those files together.

The extracted dataset has the following overall structure:

```text
FullDataset/
├── TrainingData/
│   ├── SpectralData.hdf5
│   ├── AuxillaryTable.csv
│   └── Ground Truth Package/
│       ├── FM_Parameter_Table.csv
│       └── TraceData.hdf5
│
└── TestData/
    ├── SpectralData.hdf5
    └── AuxillaryTable.csv
```

The training folder contains spectra, metadata and known target values. The test folder contains spectra and metadata, but it does not contain the target values.

---

## How one planet is represented

A single training planet is represented in several different files.

For example, for a planet with the identifier:

```text
train1
```

its information is distributed as follows:

```text
SpectralData.hdf5
└── Planet_train1
    ├── instrument_spectrum
    ├── instrument_noise
    ├── instrument_wlgrid
    └── instrument_width

AuxillaryTable.csv
└── row where planet_ID == "train1"
    ├── star_radius_m
    ├── star_temperature
    ├── planet_mass_kg
    ├── planet_surface_gravity
    └── other metadata

FM_Parameter_Table.csv
└── row where planet_ID == "train1"
    ├── planet_radius
    ├── planet_temp
    ├── log_H2O
    ├── log_CO2
    ├── log_CO
    ├── log_CH4
    └── log_NH3

TraceData.hdf5
└── Planet_train1
    ├── tracedata
    └── weights
```

This means that there is not one complete “training row” stored in one place. You have to construct one modelling example by joining the relevant pieces using `planet_ID`.

---

## `SpectralData.hdf5`

`SpectralData.hdf5` contains the observed spectral information.

There is one HDF5 group per planet:

```text
Planet_train1
Planet_train2
Planet_train3
...
```

Inside each group are four arrays:

```text
instrument_spectrum
instrument_noise
instrument_wlgrid
instrument_width
```

Each of these arrays currently have 52 values.

### Meaning of the spectral arrays

- `instrument_spectrum` contains the measured transit spectrum. This is the main spectral input to the model.
- `instrument_noise` contains the measurement uncertainty associated with each spectral bin.
- `instrument_wlgrid` contains the wavelength corresponding to each bin.
- `instrument_width` contains the width of each wavelength bin.

Conceptually, one planet has:

```text
wavelength bin       1      2      3       ...    52
spectrum value      s₁     s₂     s₃       ...    s₅₂
uncertainty         σ₁     σ₂     σ₃       ...    σ₅₂
wavelength          λ₁     λ₂     λ₃       ...    λ₅₂
bin width           w₁     w₂     w₃       ...    w₅₂
```

If every planet uses the same wavelength grid and the same bin widths, may not need to provide the wavelength and width arrays explicitly to the model. The position of each value in the 52-element spectrum would already encode the wavelength ordering.

This assumption still needs to be checked. If the wavelength grids differ between planets, will need to revisit how wavelength information is represented.

---

## `AuxillaryTable.csv`

`AuxillaryTable.csv` contains additional information about each star and planet.

The training auxiliary table contains 41,423 rows and the following columns:

```text
planet_ID
star_distance
star_mass_kg
star_radius_m
star_temperature
planet_mass_kg
planet_orbital_period
planet_distance
planet_surface_gravity
```

The external test table contains the same columns for 685 planets.

These values are potential model inputs. We do not necessarily need to use every available metadata field.

The initial modelling input should include selected stellar and planetary metadata that would be available at prediction time. The exact metadata subset is still a choice to be determined.

A future input vector for one planet could therefore contain:

```text
52 spectrum values
+ 52 uncertainty values
+ selected metadata values
```

For example, using four metadata features would result in:

```text
104 spectral and uncertainty features
+ 4 metadata features
= 108 total input features
```

The final metadata subset has not yet been fixed.

---

## `FM_Parameter_Table.csv`

`FM_Parameter_Table.csv` contains the seven simulator values that we will predict.

The relevant target columns are:

```text
planet_radius
planet_temp
log_H2O
log_CO2
log_CO
log_CH4
log_NH3
```

The table also contains `planet_ID`, which allows each target vector to be matched to the correct spectrum and metadata row.

The target vector for planet \(i\) is:

\[
y_i =
\left(
R_i,\,
T_i,\,
\log H_{2}O_i,\,
\log CO_{2,i},\,
\log CO_i,\,
\log CH_{4,i},\,
\log NH_{3,i}

ight).
\]

These are the true simulator parameters used to generate each synthetic planet.

The supervised-learning problem is therefore:

```text
spectrum + uncertainty + metadata
                    ↓
           seven simulator parameters
```

---

## `TraceData.hdf5`

`TraceData.hdf5` is different from `FM_Parameter_Table.csv`.

`FM_Parameter_Table.csv` contains one true seven-dimensional simulator parameter vector per planet.

`TraceData.hdf5` contains weighted retrieval samples for each planet:

```text
tracedata: many samples × 7 parameters
weights:   one weight per sample
```

These samples represent a Bayesian atmospheric-retrieval posterior over possible parameter values.

The distinction is:

```text
FM_Parameter_Table.csv
    one simulator-truth vector per planet

TraceData.hdf5
    many weighted retrieval samples per planet
```

We use the simulator truth as the prediction target. We are not trying to reproduce the complete Bayesian posterior (TO BE CONFIRMED!).

Therefore, for the current project:

```text
FM_Parameter_Table.csv    use for targets
TraceData.hdf5            do not use for model training
```

Retain `TraceData.hdf5` because it belongs to the original challenge and could become relevant to the alternate approach, but it is not needed for the core modelling dataset.

---

## Challenge training data versus our definitions

The word “training” is used in two different ways, which can be confusing.

The challenge calls the complete labelled package:

```text
TrainingData/
```

However, we will divide the 41,423 labelled examples in this package into four internal partitions:

```text
Challenge TrainingData
        41,423 labelled planets
                   │
                   ▼
        Project internal partitions
        ├── training
        ├── validation
        ├── calibration
        └── internal test
```

The roles of these partitions are:

### Training set

Used to fit the neural-network parameters.

### Validation set

Used as expected.

### Calibration set

Used only after the model has been frozen. It is used to calculate conformity scores and conformal thresholds.

### Internal test set

Kept untouched until final evaluation. It is used to measure prediction accuracy, conformal coverage and interval efficiency.

The calibration and internal-test sets still come from the folder named `TrainingData`, because that is where the known target values are available.

We will create these four partitions using unique planet identifiers. Their identifier lists will be saved so that the split is reproducible and mutually exclusive.

---

## What `FullDataset/TestData` represents

`FullDataset/TestData` is the original challenge’s external test set.

It contains:

```text
SpectralData.hdf5
AuxillaryTable.csv
```

but it does not contain:

```text
FM_Parameter_Table.csv
```

Can make predictions for these 685 planets, but cannot directly calculate:

- prediction error;
- marginal conformal coverage;
- simultaneous conformal coverage;
- any other metric that requires the true simulator targets.

For this reason, the external challenge test set cannot be used for the main track A/project coverage evaluation.

To avoid ambiguity, we use the following names consistently:

```text
labelled_pool
train
validation
calibration
internal_test
challenge_test
```

---

## Complete Track A data flow

The challenge training package provides the labelled data used for Track A:

```text
                 CHALLENGE TRAINING PACKAGE
                 41,423 labelled planets

     SpectralData.hdf5
        spectrum
        uncertainty
        wavelength
        bin width
              │
              │ joined using planet_ID
              ▼
     AuxillaryTable.csv
        stellar metadata
        planetary metadata
              │
              │ joined using planet_ID
              ▼
     FM_Parameter_Table.csv
        seven simulator targets
              │
              ▼
       Complete labelled dataset
              │
              ▼
     ┌────────┬────────────┬─────────────┬───────────────┐
     │ train  │ validation │ calibration │ internal test │
     └────────┴────────────┴─────────────┴───────────────┘
              │
              ▼
       Model and conformal evaluation
```

The external challenge test package remains separate:

```text
                 CHALLENGE TEST PACKAGE
                   685 unlabelled planets

     SpectralData.hdf5 + AuxillaryTable.csv
                       │
                       ▼
                challenge_test
                       │
                       ▼
               predictions only
```

The posterior-trace data is also separate from the Track A modelling target:

```text
TraceData.hdf5
    retrieval posterior samples
    retained but excluded from Track A training
```

---

## Mental model

For every labelled planet, the data pipeline needs to create and validate one complete planet-level record:

```text
planet_ID
    ├── X_spectrum:     52 measured values
    ├── X_uncertainty:  52 uncertainty values
    ├── X_metadata:     selected auxiliary values
    └── y_target:       7 simulator parameters
```

The purpose of the current data pipeline is to make sure that all four pieces belong to the same planet before creating splits, fit preprocessing transformations or train a model.
