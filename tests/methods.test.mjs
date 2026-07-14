/**
 * Method registry invariants and the Llama-3-style banded scaling math.
 * Band boundaries below are derived by solving 2π·base^(2i/D) = L/factor
 * for the pair index i (arithmetic in the comments).
 */

import test from "node:test";
import assert from "node:assert/strict";
import {
  METHODS,
  methodById,
  KNOWN_SCALING_TYPES,
  llama3InvFreqs,
  llama3Zones,
} from "../dist/methods.js";
import { invFreqs } from "../dist/rope.js";
import { assertClose } from "./helpers.mjs";

test("registry: exactly linear/ntk/yarn are plannable, all entries carry receipts", () => {
  assert.deepEqual(
    METHODS.filter((m) => m.plannable).map((m) => m.id),
    ["linear", "ntk", "yarn"],
  );
  for (const m of METHODS) {
    assert.ok(m.formula.length > 10, m.id);
    assert.ok(m.provenance.length > 10, m.id);
  }
});

test("registry: methodById round-trips, rejects unknowns, and hfTypes are all known", () => {
  assert.equal(methodById("yarn")?.hfType, "yarn");
  assert.equal(methodById("ntk")?.hfType, "—"); // static NTK has no HF rope_type
  assert.equal(methodById("frobnicate"), undefined);
  for (const m of METHODS) {
    if (m.hfType !== "—") assert.ok(KNOWN_SCALING_TYPES.includes(m.hfType), m.hfType);
  }
});

test("llama3InvFreqs: fast pairs kept, slow pairs divided by the factor", () => {
  // D=8, base=10000, L=4096, low=1, high=4 → thresholds λ<1024 keep,
  // λ>4096 interpolate. λ = [6.28, 62.8, 628.3, 6283.2]: pairs 0–2 keep,
  // pair 3 ÷8.
  const f = llama3InvFreqs(10000, 8, 8, 1, 4, 4096);
  assertClose(f[0], 1, 1e-15);
  assertClose(f[1], 0.1, 1e-15);
  assertClose(f[2], 0.01, 1e-15);
  assertClose(f[3], 0.001 / 8, 1e-18);
});

test("llama3InvFreqs: the middle band blends by the smooth factor", () => {
  // Shrink L to 1024 so pair 2 (λ=628.32) lands between λ_high=256 and
  // λ_low=1024: smooth = (1024/628.32 − 1)/(4 − 1) = 0.209904;
  // θ′ = (1−smooth)·0.01/8 + smooth·0.01 = 0.00308672.
  const f = llama3InvFreqs(10000, 8, 8, 1, 4, 1024);
  const smooth = (1024 / (2 * Math.PI * 100) - 1) / 3;
  assertClose(f[2], (1 - smooth) * (0.01 / 8) + smooth * 0.01, 1e-15);
  assertClose(f[2], 0.0030867, 1e-7);
});

test("llama3Zones: the shipped 500k-base/8k-orig split is 29/6/29, summing to the pair count", () => {
  // Solving 2π·500000^(2i/128) = 2048 gives i = 28.22 (keep i ≤ 28) and
  // = 8192 gives i = 34.98 (interpolate i ≥ 35): 29 kept, 6 blended, 29 ÷8.
  assert.deepEqual(llama3Zones(500000, 128, 1, 4, 8192), {
    keep: 29,
    blend: 6,
    interpolate: 29,
  });
  const z = llama3Zones(10000, 128, 1, 4, 4096);
  assert.equal(z.keep + z.blend + z.interpolate, 64);
});

test("llama3InvFreqs never increases a frequency", () => {
  const orig = invFreqs(500000, 128);
  const scaled = llama3InvFreqs(500000, 128, 8, 1, 4, 8192);
  for (let i = 0; i < orig.length; i++) {
    assert.ok(scaled[i] <= orig[i] + 1e-18, `pair ${i}`);
  }
});
