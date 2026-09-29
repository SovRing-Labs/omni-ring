# omni-ring — Provenance

**Licence:** PolyForm Small Business 1.0.0 **or** PolyForm Noncommercial 1.0.0, at the user's choice (`LICENSING.md`). **Disclosure:** published deliberately as prior art; no patent is claimed or filed by the authors (decision 2026-09-29).

## How this code was written
Clean-room: written from published ideas and our own measurements, not from anyone's source code. No third-party source file was copied or adapted. The only runtime dependency is NumPy (BSD-3-Clause); the C kernel (`c_src/`) uses only the C standard library and x86 AVX2 intrinsics.

## Ideas this engine builds on (cited, not copied)
| Idea | Source |
|---|---|
| Resonator networks (alternating unbind + codebook cleanup) | Frady, Kent, Olshausen, Sommer, *Resonator networks*, Neural Computation 2020 (arXiv:2007.03748); Kent et al., arXiv:1906.11684 |
| Residue number systems in hyperdimensional vectors (coprime rings + CRT) | Kymn et al., residue hyperdimensional computing (2023–24) |
| Fractional power encoding (continuous "dial" positions) | Plate, holographic reduced representations; Komer et al. / Frady et al. on FPE |
| Redundant residue check ring (single-ring slip detection) | classical redundant residue number systems |
| Grid-cell modular codes (motivation) | Fiete, Burak, Brookings 2008 |

## Patent design-around
Built to avoid the claims reviewed in the fleet's clean-room plan (IBM US 12,561,553 in-memory crossbar resonator; US 12,306,870 N resonators with permuted-and-bundled inputs; US 12,518,150 share-based bundling; US 12,579,411 reviewed separately). A freedom-to-operate opinion is still required before any commercial licence fee is charged.

## Checks run (2026-09-29)
| Check | Result |
|---|---|
| ScanCode 32.5.0 (`scancode -clpi`), 47 files | no third-party licence text or copyright notices detected. ScanCode detects licences and notices, not copied snippets (snippet matching needs ScanCode.io / matchcode) |
| gitleaks 8.30.1 (`--no-git`) | no secrets |
| Absolute home paths | stripped from README |
| Tests | 85 passed (`python3 -m pytest -q`) |

## Honest claims
Addresses and content are integrity-checked by a provable 17-ring CRT gate (100% single-ring slip detection, 54,600 exhaustive cases). Retrieval accuracy on superposition is measured, not assumed; decoding requires a cue (unguided decoding is seed-dependent — see `codebook.verify_unguided`). Abstention is available at every gate.
