/**
 * Core RoPE arithmetic — the unscaled facts every method builds on.
 *
 * RoPE rotates dimension pair i (i = 0 … D/2−1, D = rotary dim) at angular
 * frequency θᵢ = base^(−2i/D) radians per token. Everything else in this
 * project is derived from that one line:
 *
 *   wavelength λᵢ = 2π / θᵢ          tokens per full rotation
 *   rotations rᵢ  = ctx · θᵢ / 2π    full turns completed over a context
 *
 * NTK-aware scaling ("scaling by base") replaces the base so that the
 * highest-frequency pair (i = 0) is untouched and the lowest-frequency pair
 * (i = D/2−1) is interpolated by exactly the factor s:
 *
 *   base′ = base · s^(D/(D−2))
 *
 * because θ′ᵢ/θᵢ = s^(−(D/(D−2))·2i/D) = s^(−2i/(D−2)), which is 1 at i = 0
 * and 1/s at i = D/2−1. Derivations with worked examples: docs/rope-math.md.
 */

export const TWO_PI = 2 * Math.PI;

/** Per-pair angular frequencies θᵢ = base^(−2i/D), i = 0 … D/2−1. */
export function invFreqs(base: number, rotaryDim: number): number[] {
  const pairs = rotaryDim / 2;
  const out: number[] = [];
  for (let i = 0; i < pairs; i++) {
    out.push(Math.pow(base, (-2 * i) / rotaryDim));
  }
  return out;
}

/** Tokens per full rotation for one pair: λ = 2π / θ. */
export function wavelength(invFreq: number): number {
  return TWO_PI / invFreq;
}

/** Full rotations one pair completes over `ctx` tokens: r = ctx · θ / 2π. */
export function rotations(invFreq: number, ctx: number): number {
  return (ctx * invFreq) / TWO_PI;
}

/** Static NTK-aware scaled base: base′ = base · factor^(D/(D−2)). */
export function ntkBase(base: number, factor: number, rotaryDim: number): number {
  return base * Math.pow(factor, rotaryDim / (rotaryDim - 2));
}

/**
 * Per-pair frequency divisor under static NTK: θᵢ/θ′ᵢ = factor^(2i/(D−2)).
 * Exactly 1 for the fastest pair and exactly `factor` for the slowest.
 */
export function ntkDivisor(factor: number, pairIndex: number, rotaryDim: number): number {
  return Math.pow(factor, (2 * pairIndex) / (rotaryDim - 2));
}

/**
 * Dynamic NTK base at a given sequence length (transformers' "dynamic" type):
 * base′ = base · (factor·seqLen/trainedCtx − (factor−1))^(D/(D−2)), applied
 * only once seqLen exceeds the trained context.
 */
export function dynamicNtkBase(
  base: number,
  factor: number,
  seqLen: number,
  trainedCtx: number,
  rotaryDim: number,
): number {
  const inner = (factor * seqLen) / trainedCtx - (factor - 1);
  return base * Math.pow(inner, rotaryDim / (rotaryDim - 2));
}
