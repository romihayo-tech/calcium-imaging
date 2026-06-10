# Calcium Imaging Analysis

A reproducible calcium-imaging analysis toolkit and Streamlit interface for inspecting ROI-level calcium-response features, GMM-derived response modes, genotype-aware comparisons, event-composition outputs, and spatial visualizations.

This repository documents the computational workflow developed for a master’s thesis project on pancreatic acinar-cell calcium-response heterogeneity in WT, HET, and MUT Ctrb1-Δex6 mice.

## Repository scope

This repository is intended to support code and workflow transparency. It does **not** provide the confidential raw experimental data used in the thesis.

The repository may include:

- Streamlit application code for interactive inspection of processed outputs
- analysis scripts and clean notebook templates
- environment requirements
- workflow documentation
- synthetic example data showing the expected input structure

The repository should **not** include:

- raw microscope files
- full raw event tables
- full ROI-level experimental tables
- unpublished complete analysis tables unless approved by the host laboratory
- internal server paths, usernames, animal/slice identifiers, or confidential metadata
- notebook outputs containing unpublished data

## Workflow overview

The analysis workflow connects biological calcium-imaging data to reproducible computational outputs:

1. pancreatic tissue slices from WT, HET, and MUT Ctrb1-Δex6 mice
2. calcium imaging during glucose and ACh stimulation
3. ROI detection and fluorescence-trace extraction
4. calcium-event detection
5. ROI-level feature construction
6. quality control and exclusions
7. GMM clustering of ROI-level features
8. genotype-aware comparison
9. event-composition analysis
10. spatial visualization and figure generation

See [`docs/pipeline_overview.md`](docs/pipeline_overview.md) for a more detailed workflow description.

## Data availability

The raw experimental data are confidential and belong to an ongoing research line of the host laboratory. They are therefore not publicly distributed in this repository.

Synthetic example files are provided only to illustrate the expected input-table structure. They are not real biological data and must not be interpreted scientifically.

See [`docs/data_availability.md`](docs/data_availability.md) for details.

## Running the Streamlit app

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run thesis_streamlit_app.py
```

The app is designed to inspect processed analysis tables when available. It does not require raw experimental data for demonstration with synthetic or approved processed inputs.

## Reproducibility note

The repository supports reproducibility at the level of workflow organization, code transparency, and environment documentation. Access to the underlying experimental dataset remains restricted because of confidentiality requirements.

See [`docs/reproducibility_statement.md`](docs/reproducibility_statement.md).
