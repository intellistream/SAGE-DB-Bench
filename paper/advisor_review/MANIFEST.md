# Advisor review artifact manifest

- Canonical carrier: `DataSysResearch/CANDOR-Bench`, base branch `BriskSeed`
- Source base: `4cde3b3f079ad9b1e4c98b10c2dfa7edf405f07a`
- Artifact commit: `9d2d6a65a035e8615934ebd00245c32d7f26e702`
- Build: `PATH=/home/shuhao/.conda/envs/neuromem/bin:$PATH make pdf`
- Tectonic exit: `0`; pages: `2`; warnings: none
- Visual check: both pages rendered at 110 DPI and inspected; no clipping,
  overflow, broken glyphs, or unintended blank page.

## SHA256

| Artifact | Digest |
|---|---|
| `main.tex` | `0289b0ef89dd9a26934652d2ca6ac6c3e9d97ad4eb5902fcefd6be4031425077` |
| `references.bib` | `5a62eb8ce80a3534178aa7942cfb322ed65de5e90aacbcd15ef85fbe171c3b9e` |
| `main.pdf` | `5406ef30f7f9e449964451c572975f42a0dff345361dab2d6b08be722b2ac173` |
| `main.log` | `935c8f81feda1a2d2637a56b0ce76d3024643d00951b46f6e2a1a50d0dd9d93a` |
| `BUILD_TRANSCRIPT.txt` | `8608fa9a7e8410d6aac0481e07b7d2f76039db7f7ade2064e10426cc8d7ede5e` |

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
- Implementation, environment, experiments, and results remain student-owned.
