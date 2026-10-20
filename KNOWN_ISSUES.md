# Known issues

**Article.** Zhou, M., Yan, J., Peng, J., Zhang, Y., Pei, S., He, Z. *The Data Silo Trap: A
Coordination Game Analysis of AI-Driven Breeding Adoption in Fragmented Seed Markets.*
Farming System (2026). doi:10.1016/j.farsys.2026.100282

**Scope of this file.** A maintained record of the unintentional errors in the published
article and in the code that produced it, with the corrected values and a script that
recomputes them. Last updated 20 October 2026.

**Status of the article.** No erratum has been issued. The errors below leave every headline
result intact (Section 5), and in that sense the paper stands as published. What they do change is
one secondary judgement — the ranking of the three transfers (§2.5) — plus one clause of §4.11 and
two supplementary tables. They are recorded here so that anyone reusing the code or the numbers can
see exactly what is affected, and so that the record is citable.

**Can this file be ignored?** Yes, for most uses, and it is worth being exact about when.

- **If you cite the paper's qualitative claims only** — that a data-pool composition trap exists,
  that transfers which subsidise adoption do not solve it, that germplasm misalignment dominates
  while transfer decay does not, that forced adoption lowers welfare — nothing here changes them,
  and you do not need this file.
- **One secondary judgement does change.** The paper implies the quality bonus is the least costly
  of the three transfers; corrected, it is the most costly (§2.5). A sentence that rests on that
  ordering needs revising. Nothing else in the paper's argument does.
- **If you quote a number**, take it from §2.4 and §3 rather than from the published tables, and
  state which set you used (§7). Otherwise the number is ambiguous: for the bonus at b = 0.30 the
  two sets differ in sign, not merely in the last digit.

**Status of the repository.** Nothing in `data/` and nothing in `run_all.py`'s expected values
has been changed. `python run_all.py --verify-only` still reproduces the published numbers
exactly (42/42), which is what makes the checks below meaningful. Corrected values are written
to a new file and produced by a new script, so published and corrected numbers can never be
confused with one another.

**How exact the numbers below are.** `analysis/welfare_corrected.py` first re-derives, from the
same seeds and the same code path, every published number it then corrects. All five checks
reproduce to better than 5×10⁻⁷ (§6). The corrections are therefore not a different experiment
run on a different sample; they are the same arithmetic with the defect removed.

---

## 1. Summary

| ID | Issue | Type | Affected outputs | Does it change a conclusion? |
|----|-------|------|------------------|------------------------------|
| **KI-1** | Quality-bonus welfare: the pool is normalised over all firms rather than contributors, and `part_c` values the *level* of the induced pool instead of its change | coding error | §4.11 (one sentence), Tables S4 and S24 | Yes, one: the bonus is the **most** costly of the three transfers, not the least, and it never breaks even over the range in which it buys any contribution. The claim that it does not pay for itself survives, and gets stronger. |
| **KI-2** | The accuracy index has no lower bound | modelling convention | §4.11 forced-adoption gap: 21.1% → 12.5% | No. Sign and 100%-of-draws robustness unchanged; magnitude overstated by 1.69× |
| **KI-3** | θ one-at-a-time range [1, 3] is narrow | reporting choice | Table S9 | No. γ remains 8× larger |
| **KI-4** | The verification harness does not lock `W_bonus_net` (or `W_platform_net`) | test coverage | `run_all.py` | n/a — part of why KI-1 survived |
| **KI-5** | README said "29 verification values"; `run_all.py` checks 42 | documentation | README | No — corrected in the README alongside this record |

---

## 2. KI-1 — Quality-bonus welfare (the substantive one)

### 2.1 What the code does

Three places are involved.

`model/scenarios.py::_pool_quality` (which fills Table S4) normalises over **all** firms:

```python
den = omega[i].sum()                      # all 1000 firms
S_firm[i] = (q * omega[i]).sum() / den
```

`analysis/welfare.py::part_a` and `part_c` do the same at the type level:

```python
num = float((W * q_bar * omega_t[phi]).sum())
den = float((W * omega_t[phi]).sum())     # all three types, contributing or not
S_q[phi] = num / den
```

The two paths then diverge. `part_a` (Table S24) takes the *change* against the baseline pool
and nets out the bonus bill:

```python
delta_pool = np.array([S_q[t] for t in types]) - np.array([S0[t] for t in types])
gain = DELTA * BETA * float(delta_pool[adopt0].sum())
gain += DELTA * LAMBDA * BETA * float(delta_pool[~adopt0].sum())
w_b_net = gain - 0.30 * n_contrib
```

`part_c` (§4.11, and the bonus rows of `data/welfare_scan.csv`) values the *level*, with the
full β for every firm, and never references the baseline:

```python
gain = DELTA * BETA * float(sum(S_q[t] * (types == t).sum() for t in range(3)))
W_net = (gain - b * n_contrib) / n
```

### 2.2 Why it is wrong

1. **Wrong denominator (both paths).** `accuracy.shared_pool_and_centroid`, used for every
   other pool computation in the paper, normalises over pool members (Σ wρ). The P2 routines
   normalise over the whole population. Since `q = 0` for non-contributors, this records them
   as *contributing data of quality zero*, which dilutes the pool from I\* = 0.500 to 0.074 —
   the contributor share (12.0%) scaled up by the fact that contributors sit closer to the
   centroid and so carry more weight in `ω`. Table S4's 0.071–0.085 is this diluted object.
2. **Level instead of change (`part_c` only).** The gain from a bonus is the change in welfare
   relative to the pool that already existed (S₀ = 0.484), not the value of the pool as such.
   `part_c` omits the baseline, so its "gain" is the whole value of the pool, 0.037 rather than
   0.003. `part_a` gets this right.
3. **No free-riding discount (`part_c` only).** Non-adopters receive λβS, not βS; crediting βS
   to all N firms overstates the gain by more than a factor of two at the baseline adoption
   rate. `part_a` also gets this right.

**Why nobody noticed.** At b = 0.30 the level term (δβS̄ = 0.5 × 0.0744 = 0.0372) happens to be
almost exactly the bonus bill (0.30 × 0.120 = 0.0360). Net welfare therefore comes out at
+0.001, i.e. "at best breaks even", which reads like a substantive finding and is in fact two
large errors cancelling to three decimal places. The cancellation is arithmetic, not economic:
it holds at one bonus level only, which is why the published curve turns negative as soon as b
rises. Under the corrected accounting there is no near-zero region at all.

### 2.3 The two published numbers are not the same object

§4.11 reports +0.006 per firm at b = 0.26; Table S24 reports −0.1219 at b = 0.30. Both are net
of the bonus bill, and both are wrong for different reasons:

| Path | Formula | Result |
|---|---|---|
| `part_c` → §4.11 | level, full β, all firms in the denominator | +0.0058 at b = 0.26; +0.0010 at b = 0.30 |
| `part_a` → Table S24 | change against S₀, λ applied, all firms in the denominator | −0.1219 at b = 0.30 |
| corrected | change against S₀, λ applied, contributors only | +0.0029 gross; −0.028 at b = 0.26; −0.033 at b = 0.30 |

The sign flip between the first two rows is not caused by the different bonus levels. It is
caused by the level-versus-change error: `part_c` credits the whole pool (0.037), `part_a`
credits the change (−0.086, negative only because the denominator defect makes the change
negative).

### 2.4 Published versus corrected values

| Quantity | Published | Corrected |
|---|---|---|
| Induced pool quality S(q) | 0.071–0.085 (Table S4: H 0.085, M 0.080, L 0.071) | **0.500**, exactly, for every type |
| ΔS against the baseline pool | −0.409 in `part_a` (denominator defect); not taken at all in `part_c` | **+0.016** (S₀ = 0.484) |
| Gain before the bonus bill | +0.0372 (level, `part_c`) / −0.0859 (change, `part_a`) | **+0.0029** |
| Net at b = 0.26, payment counted as a cost | +0.0058 (§4.11) | **−0.028** |
| Net at b = 0.30, payment counted as a cost | −0.1219 (Table S24); +0.0010 (`part_c`) | **−0.033** |
| Net at b = 0.30, payment treated as a transfer | not reported | **+0.003** |
| Break-even bonus | "beyond b ≈ 0.35" (§4.11; its own scan crosses at ≈ 0.31) | **b ≈ 0.024** |

**An analytic observation worth keeping.** Under a threshold contract every contributor supplies
exactly `q_i = I*`, so the contributors' pool quality is exactly `I* = 0.500` regardless of
genetic distance, of firm type and of how many firms contribute — verified as the maximum
deviation from I\* over 300 draws, which is 0.0. A threshold instrument does not lift the pool
above the standard; it pins the pool at the standard. That standard is 3.4% above the baseline
pool (0.484), which is why the corrected gain is small and positive rather than large.

### 2.5 Consequence for the paper's claims

- **Survives, and is strengthened:** transfers that subsidise adoption do not pay for
  themselves; the bonus is not a way out of the trap. The phase transition at
  b\* = c₀I\*² = 0.25 is untouched — it is a property of the contribution decision, not of the
  welfare accounting.
- **Corrected:** on the paper's own convention of netting out policy expenditure, the bonus
  costs 0.033 per firm at b = 0.30, against 0.0085 for the subsidy and 0.0090 for the platform.
  It is therefore the **most** costly of the three interventions, not the least, and the
  ordering implied by §4.11 is wrong.
- **Corrected more sharply than the published text admits:** §4.11 says the bonus "at best
  breaks even" and "turns negative beyond b ≈ 0.35". Corrected, its
  break-even payment is b ≈ 0.024, which is below the b = 0.25 at which the bonus begins to buy
  any contribution at all. Over the entire range in which the instrument does anything, it
  destroys welfare — in 300 of 300 draws at both b = 0.26 and b = 0.30, not merely on average.
  Even the published curve does not support "≈ 0.35": its own scan crosses zero at b ≈ 0.31,
  between +0.0010 at b = 0.30 and −0.0110 at b = 0.40. The published near-zero is the
  cancellation described in §2.2 and is positive in only 212 of 300 draws.
- **Unaffected:** everything in Section 5.

---

## 3. KI-2 — Accuracy index without a lower bound

`a_i = αI_i + βS − γd` is not floored, so `a_i` can be negative. In the baseline equilibrium
36.2% of firms have `a_i < 0`, and **none of them are adopters** (a firm with a negative index
has a negative premium and does not adopt), so the floor changes no baseline number. Under the
forced-adoption counterfactual 79.2% of firms have `a_i < 0`.

If `a` is read as an accuracy — as the phrase "accuracy index" invites — a negative value is not
admissible, and flooring it at zero changes the forced-adoption comparison:

| | unfloored | floored at 0 |
|---|---|---|
| W₀ | 1.0845 | 1.0845 (floor does not bind) |
| W_full | 0.8558 | 0.9491 |
| Welfare gap | −21.1% | **−12.5%** |

The published gap of 21% therefore overstates the loss by a factor of 1.69. The direction of the
result is untouched: W₀ > W_full in 100% of draws under either convention, which is the claim the
paper actually makes.

The paper's §2 states that `a` is "a latent technical-performance index rather than a normalized
correlation", with an arbitrary zero. That reading is defensible and needs no floor — but then
the −21% should not be described as an accuracy loss. Under either reading the number needs a
qualifier; the qualitative conclusion does not.

---

## 4. KI-3 to KI-5 — smaller items

**KI-3 (reporting choice).** The θ one-at-a-time range is [1, 3]. Over that range the published
sensitivity table gives a T₁₀₀ span of 0.0141 (Table S9, 300 draws). Re-solving the same seeds on
a wider grid, θ ∈ {0.2, 0.5, 1, 2, 3, 5, 8}, gives 0.1597 at θ = 0.2 and 0.2139 at θ = 8, a span
of 0.0542 — nearly four times larger. γ's span is 0.433 either way, so "γ dominates θ" is
unaffected and remains true by a factor of eight; what is understated is the reported leverage of
θ. If the sensitivity table is ever reused, quote the wide range.

**KI-4 (test coverage).** `run_all.py` locks the baseline welfare (1.0845), the forced-adoption
welfare (0.8558) and the subsidy net welfare (−0.0085), but not `W_bonus_net` — the value
produced by the defective path — nor `W_platform_net`. A green verification run therefore says
nothing about KI-1. Any future revision should lock both.

**KI-5 (documentation).** README stated "29-value verification"; `EXPECTED` in `run_all.py`
holds 42 entries. Cosmetic, and corrected in the README alongside this record: it now says 42 in
both places, so the two descriptions agree.

---

## 5. What is not affected

These results are unchanged by everything above, and were re-verified while auditing:

- Baseline adoption T₁₀₀ = 0.170 (H 0.835, M 0.631, L 0.061) and the equilibrium premiums
  V_H = 0.385, V_M = 0.179, V_L = −0.053.
- The channel decomposition: the germplasm misalignment penalty γ is the dominant driver, θ a
  weak one (KI-3 changes only the reported leverage, not the ranking).
- The monotone, modest effect of cost subsidies and the linear platform response.
- The forced-adoption result: baseline welfare exceeds full adoption in 100% of draws, under
  either accuracy convention (KI-2 changes the magnitude from 21% to 12.5%).
- The ABM result that the low-adoption equilibrium welfare-dominates the high-adoption one in
  every realisation (1.112 vs 0.987 per firm).
- The analytic phase transition b\* = c₀I\*² = 0.25.
- Figures 1–7. Figure 7a plots the subsidy and platform welfare curves only; no bonus curve is
  plotted, so no published figure displays the KI-1 error.

---

## 6. Reproducing the corrected values

```
python analysis/welfare_corrected.py             # MC = 300, ~30 s (part_a's own sample)
python analysis/welfare_corrected.py --mc 40     # quick look, ~5 s
```

The script re-runs the model exactly as the published pipeline does, and for each draw computes
three things: the published bonus arithmetic (both paths), the same object with the denominator
restricted to contributors and the gain taken against S₀, and the forced-adoption gap with and
without the accuracy floor. It writes `data/welfare_bonus_corrected.csv` and modifies nothing
else.

Output at MC = 300:

```
=== 1. Reproduction of the published bonus numbers, MC = 300 ===
  Table S24  W_bonus_net (= gain_a - 0.30*share)   -0.12189   published -0.12189
  Sec. 4.11  net at b = 0.26 (= level - 0.26*share)   +0.00594   published +0.00585
  scan b=0.30 net (= level - 0.30*share)   +0.00114   published +0.00103
  ... first 200 draws, part_c's own sample: +0.00585 / +0.00103

=== 2. Where the numbers come from ===
  baseline pool quality S0                 0.48375
  contributors / N                         0.12012
  induced pool, paper denominator          0.07435   (dS = -0.40940)
  induced pool, contributors only          0.50000   (max deviation from I*: 0.00e+00)
  dS against S0, corrected                 +0.01625
  gain, level, full beta to everyone       +0.03718   <- what part_c did
  gain, dS, paper denominator              -0.08585   <- what part_a did
  gain, dS, contributors only              +0.00288   <- corrected

=== 3. Corrected net welfare of the bonus (per firm) ===
  b = 0.26   gain +0.00288  - bill 0.03123  = -0.02835
  b = 0.3    gain +0.00288  - bill 0.03604  = -0.03316
  break-even bonus b* = gain / share       0.0239   (below the 0.25 at which the bonus starts to buy anything)

=== 4. Forced adoption, with and without the accuracy floor ===
  W0, unfloored / floored                  1.0845 / 1.0845
  W_full, a unfloored                      0.8558   (gap -21.1%)
  W_full, a floored at 0                   0.9491   (gap -12.5%)
  overstatement factor                     1.69x
  a < 0: baseline 36.2%, adopters at baseline 0.0%, forced adoption 79.2%

=== 5. Reproduction check against the stored values ===
  [OK ] Table S24 W0                 +1.08448 vs +1.08448   |dev| 4.0e-07
  [OK ] Table S24 W_full             +0.85578 vs +0.85578   |dev| 8.8e-09
  [OK ] Table S24 W_bonus_net        -0.12189 vs -0.12189   |dev| 1.6e-07
  [OK ] welfare_scan bonus b=0.26    +0.00585 vs +0.00585   |dev| 1.1e-07
  [OK ] welfare_scan bonus b=0.30    +0.00103 vs +0.00103   |dev| 4.9e-07
  => all reproduced
```

---

## 7. If you are citing these numbers

Use the corrected values in §2.4 and §3. Cite the article for the published results and this
record for the corrections:

> Zhou, M., Yan, J., Peng, J., Zhang, Y., Pei, S., He, Z. *The Data Silo Trap: A Coordination
> Game Analysis of AI-Driven Breeding Adoption in Fragmented Seed Markets.* Farming System
> (2026). doi:10.1016/j.farsys.2026.100282 — with corrections recorded in
> `KNOWN_ISSUES.md`, v1 (20 October 2026), §2.4 and §3.

State which set a quoted figure comes from. The two are not within rounding of one another: at
b = 0.30 the bonus is +0.001 published and −0.033 corrected, a difference of sign as well as
magnitude, and the induced pool quality is 0.074 published and 0.500 corrected.

---

## 8. Version

**v1** (20 October 2026) — first record, and the current one. KI-1 through KI-5 documented;
`analysis/welfare_corrected.py` added; no published artefact modified, and
`run_all.py --verify-only` still reproduces the published numbers exactly (42/42). Every
published number quoted in §2 and §3 is re-derived by the script itself, so the record and the
code cannot drift apart (checks in §6, all deviations below 5×10⁻⁷).
