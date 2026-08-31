# Advisor review artifact manifest

- Canonical carrier: `DataSysResearch/CANDOR-Bench`, base branch `BriskSeed`
- Source base: `4cde3b3f079ad9b1e4c98b10c2dfa7edf405f07a`
- Artifact commit: `f9661a4f6bddd08c18b51a279c17ff1ac904a82e`
- Build: `PATH=/home/shuhao/.conda/envs/neuromem/bin:$PATH make pdf`
- Tectonic exit: `0`; pages: `2`; warnings: none
- Visual check: both pages rendered at 110 DPI and inspected; no clipping,
  overflow, broken glyphs, or unintended blank page.

## SHA256

| Artifact | Digest |
|---|---|
| `main.tex` | `ddb2aef8c6b6f04ab44507183b8eb92a46c13aa1044aff1adbd26d53b9669be6` |
| `references.bib` | `5a62eb8ce80a3534178aa7942cfb322ed65de5e90aacbcd15ef85fbe171c3b9e` |
| `main.pdf` | `4377bce0988855392623ec4290edcf2b35e66b793fb2a0c147dff83c11cc8ad7` |
| `main.log` | `ca39b46db1ebe2ded23fc46133b918502a0d6169eeefbe029e35207807605d65` |
| `BUILD_TRANSCRIPT.txt` | `bd7b5bc03ba154fb5754bbc937e2de2ce031e10006fc0279baf198f982669e30` |

## Test and evidence boundary

- `git diff --check`: pass.
- Full host suite, Python 3.11: collection stopped with seven missing-dependency
  errors (`numpy`, `yaml`).
- Full host suite, Python 3.10: collection stopped with four errors because
  `h5py` is absent.
- Dependency-light subset: 7 passed, 2 failed. Existing failures are
  `random-xs` returning no ground truth and a test shell invoking unavailable
  bare `python`; this paper-only branch changes no mechanism or test code.
- Existing formal student manuscript was preserved unchanged. Its Tectonic
  build fails because the `algorithm` environment is undefined and citation
  `Gong20251593` is absent from its bibliography.
- Existing plots/CSV files remain historical branch evidence. No raw-to-claim
  manifest binds them, so no numerical result is promoted here.
- Manual semantic audit: `paper/advisor_review/main.tex` is an independent
  formal advisor entry, not a generated fragment. Its Q1--Q7 list explicitly
  names problem, importance, cited related-work gap, mechanism, feasibility,
  matched evaluation, and takeaway; all four bibliography entries are cited.
- Implementation, environment, experiments, and results remain student-owned.
