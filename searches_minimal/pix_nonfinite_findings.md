# Where does the pixelized likelihood go non-finite?

**Issue:** autolens_workspace_developer#104 (phase 1 — localise only, no library edits).
**Predecessor:** #100/#101 (`lr_free_findings.md`, `pix_gradient_findings.md`), which
proved *by elimination* that the pixelized objective has hard non-finite walls but
could not say **which** intermediate broke.

**Probe:** `searches_minimal/probe_nonfinite_pix.py`, run on the RAL A100
(jobs 330609 / 330611, commit `cde7e60`).

---

## Answer: `log_det_regularization_matrix_term`, driven by the regularization coefficient

The NaN enters at **exactly one site** — `AbstractInversion.log_det_regularization_matrix_term`
(`autoarray/inversion/inversion/abstract.py:734-764`), the `log(diag(cholesky(H)))`
of the **reduced regularization matrix**. Everything upstream of it is finite.

The forward walk at a rejected seed-0 draw (`r_E = 1.2824` — **in-basin**, right by
the converged Nautilus mode at 1.31, so this is *not* garbage parameter space):

```
   mapping_matrix                       finite  n=14132120 range=[0.0000e+00, 9.9999e-01]
   operated_mapping_matrix              finite  n=14132120 range=[-1.0140e-16, 9.9980e-01]
   data_vector                          finite  n=920      range=[-7.5441e-01, 1.9186e+06]
   curvature_matrix                     finite  n=846400   range=[-1.3976e-12, 1.3575e+07]
   regularization_matrix                finite  n=846400   range=[-4.7988e+07, 1.9195e+08]
   curvature_reg_matrix                 finite  n=846400   range=[-4.7988e+07, 1.9201e+08]
   _nnls_precond_d                      finite  n=920      range=[3.1623e-02, 1.3857e+04]
   _nnls_precond_D                      finite  n=920      range=[7.2168e-05, 3.1623e+01]
   reconstruction                       finite  n=920      range=[0.0000e+00, 8.2063e+01]
   mapped_reconstructed_data            finite  n=15361    range=[8.1646e-03, 1.0944e+02]
   regularization_term                  finite  n=1        range=[5.2038e+01, 5.2038e+01]
   log_det_curvature_reg_matrix_term    finite  n=1        range=[1.6913e+04, 1.6913e+04]
>> log_det_regularization_matrix_term   NON-FINITE  n=1        nan=1 inf=0
   ... (chi_squared, noise_normalization all finite) ...
>> log_evidence                         NON-FINITE  n=1        nan=1 inf=0
>> figure_of_merit                      NON-FINITE  n=1        nan=1 inf=0
```

The backward walk (job 330611, after the probe's tracer fix) lands on the **same
site** and names the responsible parameter — every other stage grads finite,
including `mapped_reconstructed_data`, `model_data`, `chi_squared` and the
neighbouring `log_det_curvature_reg_matrix_term`:

```
   reconstruction                       grad finite   |g|max=4.0377e-01
   mapped_reconstructed_data            grad finite   |g|max=1.2521e-01
   regularization_term                  grad finite   |g|max=5.5472e+01
   log_det_curvature_reg_matrix_term    grad finite   |g|max=2.5947e-01
>> log_det_regularization_matrix_term   grad NON-FINITE  nan=1
      params: ('galaxies', 'source', 'pixelization', 'regularization', 'coefficient')
   chi_squared                          grad finite   |g|max=1.1100e+02
>> log_evidence                         grad NON-FINITE  nan=1   params: (... 'coefficient')
>> figure_of_merit                      grad NON-FINITE  nan=1   params: (... 'coefficient')

--> first non-finite VALUE:    log_det_regularization_matrix_term
--> first non-finite GRADIENT: log_det_regularization_matrix_term
```

**Both death classes die at the same site.** Point 0 is in-basin (`r_E = 1.2824`)
and point 1 is out-of-basin (`r_E = 5.9252`, the "arcs miss the mesh" regime), and
the summary is identical for both:

```
log_det_regularization_matrix_term | log_det_regularization_matrix_term
    <- point 0 (non-finite loss), point 1 (non-finite loss)
```

So this is **one bug, not two** — the in-basin/out-of-basin triage that motivated
#104 turns out not to split the fix. One site accounts for both.

### What this exonerates

Three of the four sites suspected when #104 was planned are **clean at these points**:

| Suspect | Verdict |
|---|---|
| `abstract.py:719` `log_det_curvature_reg_matrix_term` (curvature-reg cholesky) | **exonerated** — finite (1.6913e+04). This was the *prime* suspect (PyAutoArray#607-adjacent) and it is not the site. |
| `inversion_util.py:333-335` NNLS Jacobi preconditioner `1/sqrt(diag)` | **exonerated** — `d` ∈ [3.16e-2, 1.39e4], `D` finite. No zero diagonal, so the structurally-unmapped-mesh-pixel theory (via `knn-barycentric`) does not fire here. |
| `rectangular_kernel.py:123` `w / sum(w)` weight-map normalise | **exonerated** — `mapping_matrix` finite, so the mesh transform survives the trace. |
| `abstract.py:754` `log_det_regularization_matrix_term` | **CONFIRMED — the site.** |

That the curvature-reg cholesky is finite while the *regularization-only* cholesky
NaNs is the sharp part of this result: `curvature_reg_matrix = F + H` is
better-conditioned than `H` alone, because `F` (the curvature) lifts the null space
that `H` has by construction. Only the bare-`H` log-det breaks.

---

## Mechanism (partially confirmed — one open question)

`constant_regularization_matrix_from` (`autoarray/inversion/regularization/constant.py:43-58`)
builds

```python
regularization_coefficient = coefficient * coefficient          # λ²
diag_vals = 1e-8 + regularization_coefficient * neighbors_sizes
mat = diag(diag_vals) − regularization_coefficient * adjacency  # = λ²·L + 1e-8·I
```

`L` is a **graph Laplacian** — degree on the diagonal, −1 on adjacency. A Laplacian
is positive **semi**-definite: the constant vector is an exact null mode. So the
diagonal `1e-8` is what makes `H` invertible at all, and because that lift is
**absolute rather than relative to λ**, the spectrum is

* `eig_min ≈ 1e-8` (pinned by the lift, independent of λ)
* `eig_max ≈ λ² · max_degree` (grows without bound)
* `cond(H) ≈ λ² · max_degree / 1e-8`

**Confirmed** against the real library function on a synthetic 4-connected 30×30
mesh (`eigvalsh` + `cholesky`, both backends):

| coefficient | λ² | cond(H) | eig_min | numpy | JAX |
|---|---|---|---|---|---|
| 1.0 | 1.0 | 7.98e+08 | 1.000e-08 | ok | ok |
| 1e2 | 1e4 | 7.98e+12 | 9.998e-09 | ok | ok |
| 1e3 | 1e6 | 8.15e+14 | 9.794e-09 | ok | ok |
| 6.928e3 | 4.8e7 | 3.85e+16 | — | ok | ok |
| 1e4 | 1e8 | 2.96e+16 | 1.226e-07 | ok | ok |
| **3e4** | 9e8 | 6.72e+16 | — | **RAISES** | **NaN** |
| **1e5** | 1e10 | 6.91e+16 | **−9.931e-06** | **RAISES** | **NaN** |

`eig_min` pinned at exactly `1e-8` across four decades of λ confirms the
Laplacian-null-mode + absolute-lift structure. At large λ the smallest eigenvalue
goes **numerically negative** and the matrix is indefinite: numpy raises
`LinAlgError`, JAX returns NaN silently (`abstract.py:762` already documents this
asymmetry: *"numpy-only (JAX cholesky returns NaN)"*).

### Open question — do NOT skip this in phase 2

The synthetic mesh only fails at **λ ≳ 3e4**, but the real fit NaN'd at **λ ≈ 6.9e3**
(read off the logged `regularization_matrix` min = `−4.7988e7` = `−λ²`, and max
`1.9195e8` = `λ²·4`, consistent with a degree-4 rectangular mesh). At that λ the
synthetic matrix is fine on *both* backends — so **a numpy-vs-JAX backend
divergence is ruled out** (tested, they agree exactly), and the real reduced
regularization matrix must be **worse-conditioned than a clean regular grid**.

Candidate explanations, untested:

* The inversion is **920×920**, not 900×900 — 900 mesh pixels **plus 20 unregularized
  MGE linear amplitudes**, which is why the term uses `regularization_matrix_reduced`
  (`abstract.py:331`) and `zeroed_ids_to_keep`. The reduction may not leave a clean
  Laplacian.
* The kernel-CDF adaptive mesh may produce **isolated or disconnected mesh pixels**
  (degree 0 → diagonal `1e-8` alone; multiple connected components → multiple null
  modes → more near-zero eigenvalues than the single constant mode).
* `constant.py:55-58` scatters with `unique_indices=True`; if the mesh's `neighbors`
  array ever contains duplicate `(i, j)` pairs, that flag is a false promise to XLA
  and the scatter-add result is undefined.

**Phase 2 must dump the real `regularization_matrix_reduced` spectrum at the death
point before choosing a fix.** The localisation above is solid; this mechanism
detail is not yet settled, and the fix depends on which explanation holds.

---

## Per-site verdicts

### 1. `abstract.py:734-764` `log_det_regularization_matrix_term` — **FIX**

Not a guard, not invalid space. The evidence for calling this a genuine bug:

* the death point is **in-basin** (`r_E = 1.2824` vs the Nautilus mode 1.31), i.e.
  the sampler happily works in this region — a gradient consumer must too. The
  out-of-basin point (`r_E = 5.9252`) dies at the *same* site, so a single fix
  covers both classes and no "penalty for invalid space" verdict is needed;
* every other term of the evidence is finite at the same point, including the
  closely-related `log_det_curvature_reg_matrix_term`;
* the conditioning collapse is a **formulation artifact** (an absolute `1e-8` lift on
  a scale-free λ²), not a property of the data or the model.

Directions for phase 2, cheapest first:

* **Scale the lift with the coefficient** — e.g. `1e-8 * (1 + λ²)`, or a relative
  floor `eps · trace(H)/S`. Makes `cond(H)` bounded in λ. Changes `log_det(H)` values,
  so it is a **science-visible change** and needs an evidence-parity check against
  current results before it can ship.
* **Use `slogdet`** (or an eigenvalue sum) instead of `log(diag(cholesky))` — returns
  a sign and survives indefiniteness rather than NaN-ing. Still wrong-ish if the
  matrix is genuinely singular, but it fails *loudly*.
* **Treat the null space explicitly.** `log det(λ²L + εI)` for a singular `L` is
  ε-dependent by construction: the term is arguably ill-posed for `Constant`
  regularization on a connected mesh, and the `1e-8` is doing load-bearing science
  work while presenting as a numerical hack. Worth a Warren & Dye / Nightingale & Dye
  cross-check on how the evidence normalisation is *supposed* to treat it.

Whichever is chosen, the JAX/numpy asymmetry at `abstract.py:754` should stop being
silent: a non-PD `H` currently yields NaN on JAX and `LinAlgError` on numpy.

### 2. `autofit/non_linear/fitness.py:239-240` — **FIX (separate repo, separate issue)**

```python
log_likelihood = xp.where(xp.isnan(log_likelihood), self.resample_figure_of_merit, log_likelihood)
log_likelihood = xp.where(xp.isinf(log_likelihood), self.resample_figure_of_merit, log_likelihood)
```

This is the standard JAX `where`-guard gradient trap. It repairs the **value**
(the probe's rejected draws report `loss = inf`, not `nan` — the guard fired) but
reverse-mode AD still differentiates the masked branch and computes `0 * NaN = NaN`,
so **the resample guard does not protect gradient consumers at all**. Any
`jax.grad` user gets a NaN gradient from a point the guard "handled".

This retroactively explains a #101 observation: starts died with a *non-finite*
objective rather than a NaN one, and `optax.apply_if_finite` latched at the cliff.
It is PyAutoFit, not PyAutoArray — file separately; the standard remedy is the
double-`where` ("safe-x") pattern.

### 3. Everything else — **NO ACTION**

`log_det_curvature_reg_matrix_term`, the NNLS Jacobi preconditioner, and the
kernel-CDF weight-map normalise are all finite at the reproduced death points. Do
not pre-emptively guard them; there is no evidence they fire.

---

## Reproduction notes (for whoever picks this up)

* **The recorded death points do not reproduce anything.** `pix_lr_free.py:206-208`
  stores `last_finite_params` — the params at the last step that was *still finite*.
  They evaluate finite by construction. The NaN-producing params (one optimizer step
  later) were never persisted. Use the seed-0 draw sequence instead (below).
* **The cheap reproduction is the rejected draws.** `pix_lr_free.py:124-130` loops
  until it has 16 draws with finite loss *and* finite gradient, silently discarding
  the rest. `probe_nonfinite_pix.py --mode draws` keeps them. Seed 0, band
  U(0.15, 0.85): draws **12 and 35** are rejects (2 in 90).
* **This will not run on a laptop.** A single point's `jax.value_and_grad` needs
  **10.90 GiB**; a 15 GB / 6 GB-VRAM machine OOMs both on CPU (killed at 9.3 GB RSS)
  and on GPU. It ran in **306 s** on an A100 80GB (`probe_nonfinite.sbatch`,
  `--partition=gpu --mem=64gb`).
* **The forward walk alone is not enough.** #101 reported broad draws with *finite
  loss but NaN gradient*; for those, every forward value is finite and only the
  backward walk localises anything. (In 90 seed-0 draws this run found only
  non-finite-loss rejects, not that class — so the finite-loss/NaN-grad case from
  #101 is **not yet reproduced** and remains open.)
* RAL clone is single-branch: `git fetch origin <branch>:refs/remotes/origin/<branch>`
  explicitly; a bare `git fetch origin` fatals on deleted refspecs.
