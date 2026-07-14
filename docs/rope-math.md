# The RoPE scaling math, written out

Every number ropecalc prints comes from the formulas below. Symbols:
`D` = rotary dimension (`head_dim × partial_rotary_factor`, always even),
`i` = dimension-pair index (`0 … D/2−1`), `b` = RoPE base (`rope_theta`),
`L` = trained context, `T` = target context, `s = T/L` = scaling factor.

## 1. Unscaled RoPE

RoPE rotates pair `i` by `θᵢ` radians per token:

```
θᵢ = b^(−2i/D)
λᵢ = 2π/θᵢ                    wavelength: tokens per full rotation
rᵢ = L·θᵢ/2π                  rotations completed over the trained context
```

Pair 0 always turns 1 rad/token (λ = 2π ≈ 6.28 tokens); the slowest pair of
a 128-dim, 10000-base head has λ = 2π·10^3.9375 ≈ 54 411 tokens — trained at
4k, it never completed even a tenth of a rotation. That asymmetry is why the
methods below disagree about what to do.

Positions beyond `L` put the slow pairs into rotation angles the model has
never attended over — that, not any hard limit, is what breaks extrapolation.

## 2. Linear / Position Interpolation (`rope_type: "linear"`)

kaiokendev (2023-06); Chen et al., arXiv:2306.15595.

```
θ′ᵢ = θᵢ / s        for every i
```

Positions land `s×` closer together; nothing leaves the trained angle range.
The cost is uniform: the fastest pairs — the ones encoding *local* order —
are compressed just as hard, which is measurable degradation past ~2× and
serious past ~4× unless the model is fine-tuned at the new spacing.

## 3. NTK-aware, static (no HF rope_type — a `rope_theta` override)

bloc97, r/LocalLLaMA (2023-06).

```
b′ = b · s^(D/(D−2))
```

Substituting `b′` into `θᵢ = b′^(−2i/D)` gives the per-pair divisor

```
θᵢ/θ′ᵢ = s^(2i/(D−2))
```

which is exactly 1 at `i = 0` (local detail untouched) and exactly `s` at
`i = D/2−1` (the slowest pair interpolated like linear would). In between the
compression is geometric. One honest caveat ropecalc repeats: with the middle
pairs compressed by less than `s`, some of them still leave their trained
angle range slightly before `T`, so the effective reach shrinks near the
target — plan with headroom.

Dynamic NTK (`rope_type: "dynamic"`, emozilla 2023-07) re-derives `b′` per
sequence length:

```
b′(len) = b · (s·len/L − (s−1))^(D/(D−2))      once len > L
```

Drop-in, but keys cached early were rotated under a different base than keys
cached late — long-lived KV caches drift. `ropecalc check` warns about it.

## 4. YaRN (`rope_type: "yarn"`)

Peng et al., arXiv:2309.00071. Classify each pair by its rotations `rᵢ` over
`L`, with cutoffs `β_fast = 32` and `β_slow = 1`:

```
corr(β) = D·ln(L/(2πβ)) / (2·ln b)          pair index rotating exactly β times
low  = ⌊corr(β_fast)⌋      high = ⌈corr(β_slow)⌉      (clamped to [0, D−1])
```

A linear ramp `γᵢ` runs 0 → 1 across `[low, high]`; the extrapolation weight
is `mᵢ = 1 − γᵢ`, and

```
θ′ᵢ = θᵢ·mᵢ + (θᵢ/s)·(1 − mᵢ)
```

so pairs that rotated ≥ 32 times (fully "seen") keep their trained frequency,
pairs that never completed one rotation get pure interpolation, and the band
between blends. Finally the attention logits are scaled by the fitted
temperature

```
√(1/t) = 0.1·ln(s) + 1
```

exposed by runtimes as `attention_factor` / derived internally by llama.cpp.
Worked example (the classic 7B, `D=128, b=10000, L=4096, s=4`):
`corr(32) = 20.94 → low 20`, `corr(1) = 45.03 → high 46`, mscale `1.1386` —
21 pairs kept, 25 blended, 18 interpolated.

## 5. Llama-3.1 wavelength bands (`rope_type: "llama3"`)

Meta, 2024-07. Bands are cut by wavelength against `L`:

```
λᵢ < L/high_freq_factor   →  θ′ᵢ = θᵢ                       (kept)
λᵢ > L/low_freq_factor    →  θ′ᵢ = θᵢ/s                     (interpolated)
otherwise                 →  blend by  (L/λᵢ − low)/(high − low)
```

Note the trap ropecalc encodes: here `factor` is a *frequency divisor for the
low band*, not a reach multiplier — the shipped configs pair `factor 8` with
a declared max 16× the original, because the reach comes from long-context
fine-tuning, not from the formula. `check` therefore takes
`max_position_embeddings` as the reach for llama3 blocks and says so.

## 6. What ropecalc does not model

- Attention-quality predictions: the zones and divisors are exact, the
  perplexity consequences are empirical. The recommendation policy encodes
  the published findings (linear ≤ 4× with fine-tuning, NTK ≤ ~2× drop-in,
  YaRN beyond), not a simulation.
- Fine-tuning dynamics, data mixes, or how many steps an extension needs.
- Fork-specific extras (DeepSeek's `mscale`/`mscale_all_dim`) — recognized
  and flagged by `check`, not computed.
