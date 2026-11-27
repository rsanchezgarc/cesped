# Release notes

## v26.11.0 (2026-11-27)

Changes since v25.12.0, the previous release.

### New: RELION-free evaluation

- `evaluateEntry` accepts `--reconstructionBackend {relion,python}` (Python API: `evaluate(..., reconstructionBackend=...)`).
  The default remains `relion`, so results stay reproducible and comparable with the published CESPED numbers.
  `python` reconstructs volumes with `cesped.utils.backprojection`, a PyTorch reimplementation of
  `relion_reconstruct --ctf --pad 2 [--sym X] [--fom_weighting]` that does not require RELION. Use `--device` to pick
  the device (`auto` by default).
- Validation against RELION 4.0.1 on the TEST entry (box 232):
  - Single reconstructions of 1500 particles with symmetry C1, C4, D3, T and I, with figure-of-merit weighting,
    and with padding 1.5: Pearson correlation with the RELION maps ≥ 0.9984, FSC against the RELION maps ≥ 0.994
    at every frequency up to Nyquist, and amplitude within 0.4%.
  - Full `evaluate` run (5000 particles per half, noisy predictions): every resolution metric within 0.03 Å of
    the RELION backend, and correlations within 0.0001 (absolute). With a soft mask around the particle, masked
    resolutions agree within 0.02 Å and masked correlations within 0.0001.
  - Numbers from the two backends are close but not identical. Do not mix them in one comparison. `metrics.json`
    now records the `reconstructionBackend`, and Python-backend maps are cached with a `_python` suffix.
- Python backend limitations: symmetries C*n*, D*n*, T, O, I (I1-I4) only (RELION axis conventions); optical
  aberrations are ignored with a warning; phase-flipped or CTF-premultiplied data are rejected. For odd box sizes the
  maps differ from relion_reconstruct 4.0.1: on synthetic data the Python backend recovers the true volume
  (correlation 0.999) and RELION does not (0.81). All CESPED entries have even boxes.
- The reconstruction can be used on its own: `python -m cesped.utils.backprojection --starFname ... --outFname ...`.
- RELION settings are no longer checked when `cesped` is imported. They are checked only when the `relion`
  backend is used.

### Bug fixes

- **CTF astigmatism orientation (`cesped.datamanager.ctf._compute_ctf`)**: the row (y) frequency axis was used as
  x, so for astigmatic particles (DefocusU ≠ DefocusV) the CTF was mirrored about the diagonal. The astigmatism
  direction was therefore wrong, with errors of up to a full sign flip of the CTF, while non-astigmatic CTFs were
  unaffected. This affected the CTF correction that `ParticlesRelionStarDataset` applies to images
  (`ctf_correction="phase_flip"`, the default in `defaultDataConfig.yaml`) during training and inference. It did
  **not** affect evaluation, since reconstructions are computed by RELION. The fix changes the preprocessed
  images of astigmatic particles, so models trained with earlier versions may behave slightly differently, and
  retraining may change results relative to the published numbers.
- `evaluate` crashed while writing `metrics.json` because numpy `float32` values are not JSON serializable, so
  the final report was never printed. Metrics are now stored as Python floats.
- The evaluation report printed the `cor_diff (masked, unmasked)` values in reversed order.

### Packaging

- cesped now requires Python 3.13 or newer (`python_requires='>=3.13'`), and the dependencies in `requirements.txt`
  were updated accordingly. An `environment.yml` replaces `conda_cesped.env`
  (`conda env create -f environment.yml`).
- Fixed `install_requires` in `setup.py`, which was passed as a nested list.
