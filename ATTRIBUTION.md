# Attribution and Provenance

This repository contains two components with different origins.

## 1. PEMFC voltage-efficiency pipeline (unchanged base)

The original code and data in `part1_eda.py`, `part2_modeling.py`,
`pemfc_efficiency_clean.xlsx`, `pmfc_fuel_cell.xlsx` and the original figures
come from:

> **Eratin/PMFC_Efficiency_Predictor**
> https://github.com/Eratin/PMFC_Efficiency_Predictor
> MIT License (as stated in that repository's README)

They are preserved here without modification. Git history retains the original
commits.

## 2. Drone weight optimization component (new work)

All files under `src/`, `notebooks/`, `reports/` and `outputs/drone_weight/`,
plus `run_all.py`, `requirements.txt` and the drone section of this README, are
new work added on top of the base repository.

The drone component uses an external, openly licensed dataset:

> **FPV-DB: open dataset of FPV drone component specifications**
> DOI: 10.5281/zenodo.23613812
> https://github.com/fpvdb/fpv-db-data
> License: CC BY 4.0 — attribution: FPV-DB (fpv-db.com)

The dataset is not redistributed in this repository. `data/download_data.py`
fetches it from the canonical source and verifies the SHA-256 of the exact
snapshot used for the reported results.

## 3. Third-party research

Papers cited in `reports/REFERENCES.md` remain the property of their authors and
publishers. They are referenced, not redistributed. DOIs were verified against
Crossref/OpenAlex.

## 4. What is not claimed

* The PEMFC pipeline is not claimed as new work here.
* The FPV-DB data is not claimed as owned by this repository.
* The drone component does not reproduce any cited paper's model exactly; the
  published-vs-adapted distinction is stated per reference in
  `reports/REFERENCES.md`.
