/**
 * The `dims` engine: the per-dimension-pair view that makes the methods'
 * differences visible — for every pair, its wavelength, how many times it
 * rotated during training, which YaRN zone it falls in, and the frequency
 * divisor each method would apply at the target factor.
 *
 * For wide heads the interesting structure is at the edges and around the
 * YaRN ramp, so `selectRows` elides the boring middle unless --all is given.
 */

import { invFreqs, ntkDivisor, rotations, wavelength } from "./rope.js";
import { correctionRange, extrapolationMask, yarnMscale } from "./yarn.js";
import { roundTo } from "./units.js";
import type { DimRow, DimsResult, RopeGeometry, Zone } from "./types.js";

export interface DimsOptions {
  targetCtx: number;
  betaFast: number;
  betaSlow: number;
}

function zoneOf(mask: number): Zone {
  if (mask === 1) return "keep";
  if (mask === 0) return "interp";
  return "blend";
}

/** Compute every pair's row. Pure. */
export function computeDims(geom: RopeGeometry, opts: DimsOptions): DimsResult {
  const factor = opts.targetCtx / geom.trainedCtx;
  const freqs = invFreqs(geom.base, geom.rotaryDim);
  const { low, high } = correctionRange(
    opts.betaFast,
    opts.betaSlow,
    geom.rotaryDim,
    geom.base,
    geom.trainedCtx,
  );
  const mask = extrapolationMask(low, high, geom.pairs);

  const rows: DimRow[] = freqs.map((f, i) => {
    const m = mask[i]!;
    // Divisor = original freq / scaled freq; yarn's scaled freq is the blend.
    const yarnScaled = (f / factor) * (1 - m) + f * m;
    return {
      pair: i,
      invFreq: f,
      wavelength: wavelength(f),
      rotationsAtTrained: rotations(f, geom.trainedCtx),
      zone: zoneOf(m),
      linearDiv: factor,
      ntkDiv: ntkDivisor(factor, i, geom.rotaryDim),
      yarnDiv: f / yarnScaled,
    };
  });

  return {
    tool: "ropecalc",
    source: geom.source,
    rotaryDim: geom.rotaryDim,
    pairs: geom.pairs,
    base: geom.base,
    trainedCtx: geom.trainedCtx,
    targetCtx: opts.targetCtx,
    factor: roundTo(factor, 4),
    betaFast: opts.betaFast,
    betaSlow: opts.betaSlow,
    low,
    high,
    mscale: roundTo(yarnMscale(factor), 4),
    rows,
  };
}

/** A row index to print, or a gap marker where rows were elided. */
export type RowPick = number | "gap";

/** Threshold below which every pair is printed even without --all. */
export const ELIDE_THRESHOLD = 24;

/** How many leading/trailing pairs always survive elision. */
const EDGE = 4;

/**
 * Pick which pair indices to print: everything when small (or --all), else
 * the first/last EDGE pairs plus the YaRN ramp neighborhood, with "gap"
 * markers where runs were skipped. Deterministic, always sorted.
 */
export function selectRows(pairs: number, low: number, high: number, all: boolean): RowPick[] {
  if (all || pairs <= ELIDE_THRESHOLD) {
    return Array.from({ length: pairs }, (_, i) => i);
  }
  const keep = new Set<number>();
  for (let i = 0; i < EDGE; i++) keep.add(i);
  for (let i = pairs - EDGE; i < pairs; i++) keep.add(i);
  for (let i = Math.max(0, low - 1); i <= Math.min(pairs - 1, high + 1); i++) keep.add(i);

  const sorted = [...keep].filter((i) => i >= 0 && i < pairs).sort((a, b) => a - b);
  const out: RowPick[] = [];
  let prev = -1;
  for (const i of sorted) {
    if (prev !== -1 && i > prev + 1) out.push("gap");
    out.push(i);
    prev = i;
  }
  return out;
}
