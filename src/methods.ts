/**
 * The method registry — every rope_scaling type ropecalc understands, with
 * its formula and provenance — plus the Llama-3-style wavelength-banded
 * scaling math (validated by `check` and shown by `dims`, but not part of
 * `plan`: it shipped with fine-tuned weights and is not a drop-in recipe).
 */

import { invFreqs, TWO_PI } from "./rope.js";
import type { ZoneCounts } from "./types.js";

export interface MethodInfo {
  /** ropecalc identifier (also the `--method` value where plannable). */
  id: string;
  /** The `rope_type` value HF transformers reads, or "—" when none exists. */
  hfType: string;
  name: string;
  /** Parameters the method takes, human-readable. */
  params: string;
  /** The one-line formula. */
  formula: string;
  /** Who introduced it, where, when. */
  provenance: string;
  /** Whether `ropecalc plan` computes it as a context-extension recipe. */
  plannable: boolean;
}

export const METHODS: MethodInfo[] = [
  {
    id: "linear",
    hfType: "linear",
    name: "Linear / Position Interpolation",
    params: "factor s",
    formula: "θ′ᵢ = θᵢ / s (every pair compressed alike)",
    provenance: "kaiokendev (2023-06); formalized by Chen et al., arXiv:2306.15595",
    plannable: true,
  },
  {
    id: "ntk",
    hfType: "—",
    name: "NTK-aware (static, by base)",
    params: "factor s (as base override)",
    formula: "base′ = base · s^(D/(D−2))",
    provenance: "bloc97, r/LocalLLaMA (2023-06); no HF rope_type — set rope_theta",
    plannable: true,
  },
  {
    id: "dynamic",
    hfType: "dynamic",
    name: "Dynamic NTK",
    params: "factor s (re-derived per length)",
    formula: "base′ = base · (s·len/L − (s−1))^(D/(D−2)) once len > L",
    provenance: "emozilla, r/LocalLLaMA (2023-07)",
    plannable: false,
  },
  {
    id: "yarn",
    hfType: "yarn",
    name: "YaRN (NTK-by-parts + temperature)",
    params: "factor s, β_fast 32, β_slow 1",
    formula: "blend θᵢ vs θᵢ/s by rotations over L; attn × (0.1·ln s + 1)",
    provenance: "Peng et al., arXiv:2309.00071 (2023-09)",
    plannable: true,
  },
  {
    id: "llama3",
    hfType: "llama3",
    name: "Llama-3.1 wavelength bands",
    params: "factor s, low/high_freq_factor, original ctx L",
    formula: "λ < L/high: keep · λ > L/low: θᵢ/s · else smooth blend",
    provenance: "Meta, shipped with Llama 3.1 (2024-07); needs the fine-tuned weights",
    plannable: false,
  },
];

/** Look up a method by id; undefined for unknown ids. */
export function methodById(id: string): MethodInfo | undefined {
  return METHODS.find((m) => m.id === id);
}

/** The rope_scaling `type` values `check` accepts (HF's set, "default" included). */
export const KNOWN_SCALING_TYPES = ["default", "linear", "dynamic", "yarn", "llama3"] as const;

/**
 * Llama-3-style scaled frequencies. Bands are defined by wavelength against
 * the trained context L: pairs with λ < L/high_freq_factor are kept, pairs
 * with λ > L/low_freq_factor are divided by the factor, and the band in
 * between is blended by
 *   smooth = (L/λ − low_freq_factor) / (high_freq_factor − low_freq_factor)
 * exactly as in the reference implementation.
 */
export function llama3InvFreqs(
  base: number,
  rotaryDim: number,
  factor: number,
  lowFreqFactor: number,
  highFreqFactor: number,
  trainedCtx: number,
): number[] {
  const lowFreqWavelen = trainedCtx / lowFreqFactor;
  const highFreqWavelen = trainedCtx / highFreqFactor;
  return invFreqs(base, rotaryDim).map((f) => {
    const wavelen = TWO_PI / f;
    if (wavelen < highFreqWavelen) return f;
    if (wavelen > lowFreqWavelen) return f / factor;
    const smooth = (trainedCtx / wavelen - lowFreqFactor) / (highFreqFactor - lowFreqFactor);
    return (1 - smooth) * (f / factor) + smooth * f;
  });
}

/** Pair counts per Llama-3 band: kept / blended / interpolated by the full factor. */
export function llama3Zones(
  base: number,
  rotaryDim: number,
  lowFreqFactor: number,
  highFreqFactor: number,
  trainedCtx: number,
): ZoneCounts {
  const lowFreqWavelen = trainedCtx / lowFreqFactor;
  const highFreqWavelen = trainedCtx / highFreqFactor;
  let keep = 0;
  let blend = 0;
  let interpolate = 0;
  for (const f of invFreqs(base, rotaryDim)) {
    const wavelen = TWO_PI / f;
    if (wavelen < highFreqWavelen) keep++;
    else if (wavelen > lowFreqWavelen) interpolate++;
    else blend++;
  }
  return { keep, blend, interpolate };
}
