# Stellar Spectral Classification from SDSS Photometry

**Can cheap five-band imaging stand in for expensive spectroscopy?**

A machine learning pipeline that classifies stars along the Harvard spectral 
sequence (O, B, A, F, G, K, M) using only broadband photometric colors from 
SDSS DR17 — no spectroscopy required at inference time.

## Motivation

Spectroscopy is the gold standard for stellar classification but doesn't scale: 
each star needs individual observation time. SDSS photometry exists for hundreds 
of millions of objects in five filters (u, g, r, i, z). If ML can reliably map 
those five brightness measurements to a spectral class, it unlocks classification 
for the billions of stars in current and next-generation surveys (LSST/Rubin, 
Euclid) where spectra will never be available.

## Dataset

- **Source:** SDSS DR17, via SkyServer ADQL query (SpecObj × PhotoObj)
- **Size:** 93,628 spectroscopically-labelled stars after cleaning
- **Labels:** Parsed from raw SDSS subClass strings (e.g. "G2V" → G2); 
  53 distinct subtypes across OBAFGKM
- **Features:** 11 — six color indices (u−g, g−r, r−i, i−z, u−r, g−i) 
  plus five raw magnitudes

## Models

| Model | Description |
|---|---|
| Baseline | Classical color-cut rules on g−r and u−g |
| Random Forest | 300 trees, balanced class weights, 5-fold CV |
| XGBoost | 300 rounds, depth 6, cost-sensitive |
| MLP | [128, 64, 32] with ReLU, early stopping, StandardScaler |

Training is on granular subtypes (G2, K1) with coarse-level evaluation — 
preserving the continuous temperature gradient across class boundaries.

## Key Result

ML lifts K-type star classification from **F1 = 51% to F1 = 90%** (+39pp), 
with recall improving from 46% to 91%. K stars are the primary exoplanet 
host targets in survey astr
