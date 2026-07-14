/**
 * YaRN — "NTK-by-parts" interpolation plus an attention temperature
 * (Peng et al. 2023, arXiv:2309.00071), matching the reference
 * implementation that Hugging Face transformers ships.
 *
 * Instead of scaling every pair the same way, YaRN asks how many full
 * rotations a pair completes over the *trained* context L:
 *
 *   r(i) = L / λᵢ
 *
 * Pairs rotating at least β_fast times (default 32) have seen every phase
 * during training — they are kept as-is ("extrapolation"). Pairs completing
 * less than β_slow rotations (default 1) never wrapped — they are divided by
 * the full factor s ("interpolation", same as linear). Pairs in between get
 * a linear blend. Solving r(i) = β for the pair index gives the ramp
 * boundaries:
 *
 *   corr(β) = D · ln(L / (2πβ)) / (2 · ln base)
 *   low  = ⌊corr(β_fast)⌋   high = ⌈corr(β_slow)⌉      (HF floor/ceil + clamp)
 *
 * Finally the attention logits are scaled by √(1/t) = 0.1·ln(s) + 1 — the
 * paper's fitted temperature, exposed by runtimes as `attention_factor`.
 */

import { invFreqs, TWO_PI } from "./rope.js";
import type { ZoneCounts } from "./types.js";

/** YaRN defaults, as fixed in the paper and in every runtime. */
export const DEFAULT_BETA_FAST = 32;
export const DEFAULT_BETA_SLOW = 1;

/** Pair index at which a pair completes exactly `numRotations` turns over `trainedCtx`. */
export function correctionDim(
  numRotations: number,
  rotaryDim: number,
  base: number,
  trainedCtx: number,
): number {
  return (rotaryDim * Math.log(trainedCtx / (numRotations * TWO_PI))) / (2 * Math.log(base));
}

/**
 * Ramp boundaries in pair-index space, with HF's exact floor/ceil and
 * clamping semantics (low ≥ 0, high ≤ rotaryDim − 1).
 */
export function correctionRange(
  betaFast: number,
  betaSlow: number,
  rotaryDim: number,
  base: number,
  trainedCtx: number,
): { low: number; high: number } {
  const low = Math.floor(correctionDim(betaFast, rotaryDim, base, trainedCtx));
  const high = Math.ceil(correctionDim(betaSlow, rotaryDim, base, trainedCtx));
  return { low: Math.max(low, 0), high: Math.min(high, rotaryDim - 1) };
}

/**
 * Extrapolation weight per pair: 1 = keep the trained frequency,
 * 0 = interpolate by the full factor. Mirrors HF's `linear_ramp_factor`
 * (including the +0.001 guard when low == high).
 */
export function extrapolationMask(low: number, high: number, pairs: number): number[] {
  const hi = high === low ? high + 0.001 : high;
  const out: number[] = [];
  for (let i = 0; i < pairs; i++) {
    const ramp = Math.min(1, Math.max(0, (i - low) / (hi - low)));
    out.push(1 - ramp);
  }
  return out;
}

/** Scaled per-pair frequencies: θ′ᵢ = θᵢ·mask + (θᵢ/s)·(1 − mask). */
export function yarnInvFreqs(
  base: number,
  rotaryDim: number,
  factor: number,
  trainedCtx: number,
  betaFast: number,
  betaSlow: number,
): number[] {
  const freqs = invFreqs(base, rotaryDim);
  const { low, high } = correctionRange(betaFast, betaSlow, rotaryDim, base, trainedCtx);
  const mask = extrapolationMask(low, high, freqs.length);
  return freqs.map((f, i) => (f / factor) * (1 - mask[i]!) + f * mask[i]!);
}

/** Attention temperature √(1/t) = 0.1·ln(s) + 1; 1 for s ≤ 1 (no shrinking). */
export function yarnMscale(factor: number): number {
  if (factor <= 1) return 1;
  return 0.1 * Math.log(factor) + 1;
}

/** Count pairs per zone from the mask: kept (mask=1), blended, interpolated (mask=0). */
export function yarnZones(low: number, high: number, pairs: number): ZoneCounts {
  const mask = extrapolationMask(low, high, pairs);
  let keep = 0;
  let blend = 0;
  let interpolate = 0;
  for (const m of mask) {
    if (m === 1) keep++;
    else if (m === 0) interpolate++;
    else blend++;
  }
  return { keep, blend, interpolate };
}
