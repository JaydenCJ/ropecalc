/**
 * The plan engine: factor arithmetic, per-method parameters, and the
 * recommendation policy (which is a documented contract, so it is pinned
 * branch by branch).
 */

import test from "node:test";
import assert from "node:assert/strict";
import { computePlan } from "../dist/plan.js";
import { geometryFromFlags, normalizeConfig } from "../dist/config.js";
import { tinyConfig } from "./helpers.mjs";
import { assertClose } from "./helpers.mjs";

const g7b = geometryFromFlags({ dim: 128, base: 10000, ctx: 4096 });

function plan(geom, opts) {
  return computePlan(geom, { targetCtx: 16384, ...opts });
}

test("factor is target/trained; the flagship 4× numbers all line up", () => {
  const p = plan(g7b);
  assert.equal(p.factor, 4);
  assert.equal(p.linear.factor, 4);
  assertClose(p.ntk.scaledBase, 40889.9, 0.05); // 10000·4^(128/126)
  assert.equal(p.yarn.low, 20);
  assert.equal(p.yarn.high, 46);
  assert.deepEqual(p.yarn.zones, { keep: 21, blend: 25, interpolate: 18 });
  assertClose(p.yarn.mscale, 1.1386, 1e-4);
});

test("target within trained context: no methods, recommendation none", () => {
  const p = plan(g7b, { targetCtx: 2048 });
  assert.equal(p.factor, 0.5);
  assert.equal(p.linear, null);
  assert.equal(p.ntk, null);
  assert.equal(p.yarn, null);
  assert.equal(p.recommendation.method, "none");
});

test("recommendation policy without fine-tuning: ntk to 2×, yarn beyond — always with a reason", () => {
  assert.equal(plan(g7b, { targetCtx: 8192 }).recommendation.method, "ntk"); // 2×
  assert.equal(plan(g7b, { targetCtx: 16384 }).recommendation.method, "yarn"); // 4×
  for (const target of [2048, 8192, 16384, 65536]) {
    assert.ok(plan(g7b, { targetCtx: target }).recommendation.reason.length > 20, String(target));
  }
});

test("recommendation policy with fine-tuning: linear to 4×, yarn beyond", () => {
  assert.equal(plan(g7b, { targetCtx: 16384, finetune: true }).recommendation.method, "linear");
  assert.equal(plan(g7b, { targetCtx: 32768, finetune: true }).recommendation.method, "yarn"); // 8×
});

test("beta overrides flow into the yarn correction range", () => {
  // Tighter betas (16/2) shrink the ramp against the 32/1 default 20…46.
  const p = plan(g7b, { betaFast: 16, betaSlow: 2 });
  assert.deepEqual([p.yarn.betaFast, p.yarn.betaSlow], [16, 2]);
  assert.ok(p.yarn.low > 20, `low ${p.yarn.low}`);
  assert.ok(p.yarn.high < 46, `high ${p.yarn.high}`);
});

test("a config that already declares scaling plans from its original context", () => {
  const geom = normalizeConfig(
    tinyConfig({
      head_dim: 128,
      max_position_embeddings: 131072,
      rope_scaling: { rope_type: "yarn", factor: 4, original_max_position_embeddings: 32768 },
    }),
    "t",
  );
  const p = computePlan(geom, { targetCtx: 262144 });
  assert.equal(p.trainedCtx, 32768);
  assert.equal(p.factor, 8);
  assert.match(p.warnings.join("\n"), /already declares yarn scaling/);
});

test("factors beyond 32× carry an uncharted-territory warning", () => {
  const p = plan(g7b, { targetCtx: 4096 * 64 }); // 64×
  assert.match(p.warnings.join("\n"), /beyond published results/);
  assert.equal(plan(g7b, { targetCtx: 16384 }).warnings.length, 0);
});

test("non-integer factors survive rounding honestly", () => {
  const p = plan(g7b, { targetCtx: 6144 }); // 1.5×
  assert.equal(p.factor, 1.5);
  assert.equal(p.recommendation.method, "ntk");
});
