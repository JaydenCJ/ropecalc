/**
 * YaRN math against hand-derived values, following the HF reference
 * semantics (floor/ceil correction range, linear ramp, mask blending).
 *
 * The D=8, base=10000, L=4096 worked example used below:
 *   corr(β) = D·ln(L/(2πβ)) / (2·ln base) = 8·ln(651.8986/β) / 18.420681
 *   corr(32) = 8·ln(20.3718)/18.420681 = 1.3090 → low = ⌊·⌋ = 1
 *   corr(1)  = 8·ln(651.8986)/18.420681 = 2.8142 → high = ⌈·⌉ = 3
 */

import test from "node:test";
import assert from "node:assert/strict";
import {
  correctionDim,
  correctionRange,
  extrapolationMask,
  yarnInvFreqs,
  yarnMscale,
  yarnZones,
} from "../dist/yarn.js";
import { assertClose } from "./helpers.mjs";

test("correctionDim and correctionRange match the worked D=8 example", () => {
  assertClose(correctionDim(32, 8, 10000, 4096), 1.30903, 1e-4);
  assertClose(correctionDim(1, 8, 10000, 4096), 2.81417, 1e-4);
  assert.deepEqual(correctionRange(32, 1, 8, 10000, 4096), { low: 1, high: 3 });
});

test("correctionRange for real geometries: 7B/10000/4k → 20…46, 128-dim/1M/32k → 23…40", () => {
  // corr(32)=20.944→20, corr(1)=45.029→46 for the classic 7B head.
  assert.deepEqual(correctionRange(32, 1, 128, 10000, 4096), { low: 20, high: 46 });
  // corr(32)=23.596→23, corr(1)=39.651→40 for a 1M-base 32k model.
  assert.deepEqual(correctionRange(32, 1, 128, 1000000, 32768), { low: 23, high: 40 });
});

test("correctionRange clamps into [0, rotaryDim−1]", () => {
  // A tiny trained context pushes both boundaries below zero.
  const r = correctionRange(32, 1, 8, 10000, 8);
  assert.ok(r.low >= 0, `low ${r.low}`);
  assert.ok(r.high <= 7, `high ${r.high}`);
});

test("extrapolationMask: 1 up to low, linear ramp, 0 from high; low==high gets the HF epsilon", () => {
  // low=1, high=3 over 4 pairs: ramp(i) = (i−1)/2 clamped → mask = 1−ramp.
  assert.deepEqual(extrapolationMask(1, 3, 4), [1, 1, 0.5, 0]);
  // The +0.001 guard: no NaN, a hard step at the boundary.
  assert.deepEqual(extrapolationMask(2, 2, 4), [1, 1, 1, 0]);
});

test("yarnInvFreqs blends θᵢ and θᵢ/s by the mask (D=8, s=2)", () => {
  // masks [1, 1, 0.5, 0] over θ = [1, 0.1, 0.01, 0.001]:
  //   pair 2: 0.5·0.01 + 0.5·0.005 = 0.0075; pair 3: 0.001/2 = 0.0005.
  const f = yarnInvFreqs(10000, 8, 2, 4096, 32, 1);
  assertClose(f[0], 1, 1e-15);
  assertClose(f[1], 0.1, 1e-15);
  assertClose(f[2], 0.0075, 1e-15);
  assertClose(f[3], 0.0005, 1e-15);
});

test("yarnMscale: √(1/t) = 0.1·ln(s) + 1, floored at 1 for s ≤ 1", () => {
  assertClose(yarnMscale(4), 1.1386294361, 1e-9);
  assertClose(yarnMscale(2), 1.0693147181, 1e-9);
  assert.equal(yarnMscale(1), 1);
  assert.equal(yarnMscale(0.5), 1); // never shrinks attention
});

test("yarnZones counts kept/blended/interpolated pairs from the mask", () => {
  assert.deepEqual(yarnZones(1, 3, 4), { keep: 2, blend: 1, interpolate: 1 });
  // The classic 7B split at s=4: mask=1 for i≤20, mask=0 for i≥46.
  assert.deepEqual(yarnZones(20, 46, 64), { keep: 21, blend: 25, interpolate: 18 });
});
