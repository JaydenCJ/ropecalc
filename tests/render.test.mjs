/**
 * Renderers: deterministic text with the load-bearing numbers present.
 * These tests pin content, not full snapshots — wording may evolve, the
 * numbers and structure may not.
 */

import test from "node:test";
import assert from "node:assert/strict";
import { renderPlan, renderCheck, renderDims, renderMethods } from "../dist/render.js";
import { computePlan } from "../dist/plan.js";
import { checkConfig } from "../dist/validate.js";
import { computeDims } from "../dist/dims.js";
import { geometryFromFlags, normalizeConfig } from "../dist/config.js";
import { tinyConfig } from "./helpers.mjs";

const g7b = geometryFromFlags({ dim: 128, base: 10000, ctx: 4096 });

test("renderPlan shows all three methods and the recommendation, deterministically", () => {
  const text = renderPlan(computePlan(g7b, { targetCtx: 16384 }), g7b, null);
  for (const want of [
    "factor 4.00×",
    "rope_theta 10000 → 40889.9",
    "ramp pairs 20…46 of 64",
    "mscale 1.14",
    "recommend yarn",
  ]) {
    assert.ok(text.includes(want), want);
  }
  // Identical inputs must render byte-identical output — people pipe this to grep.
  assert.equal(text, renderPlan(computePlan(g7b, { targetCtx: 16384 }), g7b, null));
});

test("renderPlan --method narrows to one block but keeps the recommendation", () => {
  const plan = computePlan(g7b, { targetCtx: 16384 });
  const text = renderPlan(plan, g7b, "ntk");
  assert.ok(text.includes("rope_theta 10000 → 40889.9"));
  assert.ok(!text.includes("mscale"), "yarn block must be hidden");
  assert.ok(text.includes("recommend yarn"));
});

test("renderPlan for a no-op target says none and shows no method blocks", () => {
  const text = renderPlan(computePlan(g7b, { targetCtx: 2048 }), g7b, null);
  assert.ok(text.includes("recommend none"));
  assert.ok(!text.includes("rope_theta 10000 →"));
});

test("renderCheck prints one line per finding plus the verdict", () => {
  const geom = normalizeConfig(
    tinyConfig({ rope_scaling: { rope_type: "yarn", factor: 4, beta_fast: 1, beta_slow: 32 } }),
    "t",
  );
  const text = renderCheck(checkConfig(geom), geom);
  assert.match(text, /error\s+beta_fast \(1\) must be greater/);
  assert.match(text, /verdict\s+INVALID — 1 error/);
});

test("renderCheck verdict counts pluralize correctly", () => {
  const geom = normalizeConfig(tinyConfig(), "t");
  const text = renderCheck(checkConfig(geom), geom);
  assert.match(text, /VALID — 0 errors · 0 warnings/);
});

test("renderDims prints an aligned header and divisor cells", () => {
  const dims = computeDims(g7b, { targetCtx: 16384, betaFast: 32, betaSlow: 1 });
  const text = renderDims(dims, false);
  assert.match(text, /pair\s+wavelength\s+rot@trained\s+zone\s+linear\s+ntk\s+yarn/);
  assert.match(text, /÷4\.00/);
  assert.match(text, /keep/);
  assert.match(text, /interp/);
  assert.ok(text.includes("--all prints every pair"));
});

test("renderDims --all drops the elision notice and prints every pair", () => {
  const dims = computeDims(g7b, { targetCtx: 16384, betaFast: 32, betaSlow: 1 });
  const text = renderDims(dims, true);
  assert.ok(!text.includes("--all prints every pair"));
  assert.match(text, /\n +63 /); // the last pair is present
});

test("renderMethods lists every registry entry with provenance", () => {
  const text = renderMethods();
  for (const id of ["linear", "ntk", "dynamic", "yarn", "llama3"]) {
    assert.ok(text.includes(`\n${id.padEnd(10)}`) || text.startsWith(id), id);
  }
  assert.ok(text.includes("arXiv:2309.00071")); // YaRN's receipt
});
