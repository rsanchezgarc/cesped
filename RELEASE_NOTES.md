# Release notes

## Unreleased

### New: RELION-free evaluation

- `evaluateEntry` accepts `--reconstructionBackend {relion,python}` (Python API: `evaluate(..., reconstructionBackend=...)`).
  The default remains `relion`, so results stay reproducible and comparable with the published CESPED numbers.
  `python` reconstructs volumes with `cesped.utils.backprojection`, a PyTorch reimplementation of
  `relion_reconstruct --ctf --pad 2 [--sym X] [--fom_weighting]` that does not require RELION. Use `--device` to pick
  the device (`auto` by default).
- Validation against RELION 4.0.1 on the TEST entry (5000 particles per half, box 232):
  - Single reconstructions (C1, C1 with figure-of-merit weighting, D2, and both half-sets): Pearson correlation
    with the RELION maps ≥ 0.9986; FSC against the RELION maps ≥ 0.9965 at every frequency up to Nyquist.
  - Full `evaluate` run with noisy predictions: every resolution metric within 0.03 Å of the RELION backend;
    correlations equal to 4 decimals.
  - Numbers from the two backends are close but not identical. Do not mix them in one comparison. `metrics.json`
    now records the `reconstructionBackend`, and Python-backend maps are cached with a `_python` suffix.
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
