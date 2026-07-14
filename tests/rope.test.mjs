/**
 * Core RoPE arithmetic against paper-derived values. Rotary dim 8 at base
 * 10000 is used throughout because its frequencies are exact powers of ten:
 * θᵢ = 10000^(−2i/8) = [1, 0.1, 0.01, 0.001].
 */

import test from "node:test";
import assert from "node:assert/strict";
import {
  invFreqs,
  wavelength,
  rotations,
  ntkBase,
  ntkDivisor,
  dynamicNtkBase,
  TWO_PI,
} from "../dist/rope.js";
import { assertClose } from "./helpers.mjs";

test("invFreqs: θᵢ = base^(−2i/D) — exact powers of ten at D=8, strictly decreasing at D=128", () => {
  const f = invFreqs(10000, 8);
  assert.equal(f.length, 4);
  assertClose(f[0], 1, 1e-15);
  assertClose(f[1], 0.1, 1e-15);
  assertClose(f[2], 0.01, 1e-15);
  assertClose(f[3], 0.001, 1e-15);
  const wide = invFreqs(500000, 128);
  assert.equal(wide.length, 64);
  for (let i = 1; i < wide.length; i++) assert.ok(wide[i] < wide[i - 1], `pair ${i}`);
  assert.equal(wide[0], 1); // the fastest pair always rotates 1 rad/token
});

test("wavelength and rotations: λ = 2π/θ, r = ctx·θ/2π", () => {
  // λ₆₃ = 2π · 10000^(126/128) = 2π · 10^3.9375 ≈ 54410.6 tokens: this is
  // why a 4k-trained model has dims that never completed a single rotation.
  const f = invFreqs(10000, 128);
  assertClose(wavelength(f[63]), TWO_PI * Math.pow(10, 3.9375), 1e-6);
  assertClose(wavelength(f[0]), TWO_PI, 1e-15);
  // Pair 0 over 4096 tokens turns 4096/2π = 651.8986… times.
  assertClose(rotations(1, 4096), 4096 / TWO_PI, 1e-12);
  assertClose(rotations(1, 4096), 651.8986469, 1e-6);
});

test("ntkBase: base′ = base·s^(D/(D−2)) — small case and the flagship 7B number", () => {
  // s=2, D=8: 10000·2^(4/3) = 25198.42.
  assertClose(ntkBase(10000, 2, 8), 10000 * Math.pow(2, 4 / 3), 1e-9);
  assertClose(ntkBase(10000, 2, 8), 25198.420998, 1e-5);
  // s=4, D=128: 10000·4^(128/126) = 40889.94 — the number people paste into
  // --rope-freq-base for a 4×-extended 10000-base model.
  assertClose(ntkBase(10000, 4, 128), 40889.94, 0.01);
});

test("ntkDivisor: exactly 1 at the fastest pair, exactly s at the slowest", () => {
  assert.equal(ntkDivisor(4, 0, 128), 1);
  // Slowest pair index is D/2−1 = 63: exponent 2·63/126 = 1 → divisor = s.
  assertClose(ntkDivisor(4, 63, 128), 4, 1e-12);
  assertClose(ntkDivisor(4, 32, 128), Math.pow(4, 64 / 126), 1e-12);
});

test("ntkDivisor is consistent with ntkBase: scaled freqs equal base′ freqs", () => {
  // θ′ᵢ from base′ must equal θᵢ / divisor(i) — the two formulations are one.
  const D = 128;
  const s = 4;
  const orig = invFreqs(10000, D);
  const scaled = invFreqs(ntkBase(10000, s, D), D);
  for (const i of [0, 1, 17, 40, 63]) {
    assertClose(scaled[i], orig[i] / ntkDivisor(s, i, D), 1e-15, `pair ${i}`);
  }
});

test("dynamicNtkBase: no change at the trained length, base·3^(4/3) at double", () => {
  assertClose(dynamicNtkBase(10000, 2, 4096, 4096, 8), 10000, 1e-9);
  // (2·8192/4096 − 1) = 3; 3^(4/3) = 4.326749 → 43267.49.
  assertClose(dynamicNtkBase(10000, 2, 8192, 4096, 8), 10000 * Math.pow(3, 4 / 3), 1e-9);
  assertClose(dynamicNtkBase(10000, 2, 8192, 4096, 8), 43267.49, 0.01);
});
