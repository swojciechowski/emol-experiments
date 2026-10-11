# Evolutionary Multi-Objective Learning - Experiments
---

This repository is a results archive and reproduction suite for doctoral dissertation:

> Multi-objective Classifier Learning from Imbalanced Data: Data-level Methods and Preference-aware Evaluation

prepared by *Szymon Wojciechowski*.

Presented code is also an example of `emol` package usage, which implements methods presented in thesis.

## Files Structure

The main repository directory contains configuration files. All experiment files are places in `scripts/01_training` directory.

The `_results.zip` file contains predictions of models used in experiments. It is stored in `numpy` files format. To analyze results, unpack this file and use scripts in directory `02_analysis` and `03_analysis_fb`.

**NOTE**: Scripts use deprecated names `DummyFitting` and `BootstrapFitting` for MOWS method, which was changed. If You intend to replicate experiments, please change configuration file to use new splitting strategies names.

Additionally, `scripts/misc` directory is included which can be used to recreate visualizations and discussions outside of experiments scope.

The `keel` directory is repository of datasets used, imported from [keel repository](https://sci2s.ugr.es/keel/datasets.php).

## Dependencies

## emol

The `emol` library is crucial component for experiments reproduction. Please follow installation instructions in [this repository](https://github.com/swojciechowski/emol).

## smote-variants

Please use locally copied repository of [smote-variants](https://github.com/analyticalmindsltd/smote_variants) as *PyPI* contains outdated API.

## pymoo

For custom solvers definitions consult [pymoo](https://pymoo.org) documentation. It is also suggested to use local clone of `pymoo` repository.

Rest of dependencies is described in `requirements.txt` file.
