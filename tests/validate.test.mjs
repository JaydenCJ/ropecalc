/**
 * The check rule engine. Each test pins one rule family: the config that
 * trips it, the severity it must get, and (for the invariants) the config
 * that passes. Findings are asserted by message substring so reworded
 * output stays honest.
 */

import test from "node:test";
import assert from "node:assert/strict";
import { normalizeConfig } from "../dist/config.js";
import { checkConfig } from "../dist/validate.js";
import { tinyConfig } from "./helpers.mjs";

function check(configOverrides, opts = {}) {
  return checkConfig(normalizeConfig(tinyConfig(configOverrides), "t"), opts);
}

function levelOf(result, pattern) {
  const f = result.findings.find((x) => pattern.test(x.message));
  return f ? f.level : null;
}

test("no rope_scaling block: valid as-is, but a target beyond it points at plan", () => {
  const r = check({});
  assert.equal(r.valid, true);
  assert.equal(r.scalingType, null);
  assert.equal(r.effectiveMax, 4096);
  assert.equal(levelOf(r, /no rope_scaling block/), "ok");
  const beyond = check({}, { targetCtx: 8192 });
  assert.equal(levelOf(beyond, /no scaling is declared — run `ropecalc plan`/), "error");
  assert.equal(beyond.valid, false);
});

test("unknown scaling type and missing type keys are errors", () => {
  const unknown = check({ rope_scaling: { rope_type: "quadratic", factor: 2 } });
  assert.equal(unknown.valid, false);
  assert.equal(levelOf(unknown, /unknown rope_scaling type "quadratic"/), "error");
  const missing = check({ rope_scaling: { factor: 2 } });
  assert.equal(levelOf(missing, /neither "rope_type" nor "type"/), "error");
});

test("conflicting type and rope_type is an error; legacy type alone is info", () => {
  const a = check({ rope_scaling: { rope_type: "yarn", type: "linear", factor: 2 } });
  assert.equal(levelOf(a, /both present and disagree/), "error");
  const b = check({ rope_scaling: { type: "linear", factor: 2 } });
  assert.equal(levelOf(b, /legacy "type" key/), "info");
  assert.equal(b.valid, true);
});

test("factor rules: missing, non-numeric, < 1, == 1, in range", () => {
  assert.equal(levelOf(check({ rope_scaling: { rope_type: "linear" } }), /requires a numeric "factor"/), "error");
  assert.equal(levelOf(check({ rope_scaling: { rope_type: "linear", factor: "4" } }), /factor is not a finite number/), "error");
  assert.equal(levelOf(check({ rope_scaling: { rope_type: "linear", factor: 0.5 } }), /must extend, not shrink/), "error");
  assert.equal(levelOf(check({ rope_scaling: { rope_type: "linear", factor: 1 } }), /no-op/), "warn");
  assert.equal(levelOf(check({ rope_scaling: { rope_type: "linear", factor: 2 } }), /factor 2 is in range/), "ok");
});

test("factor × original vs declared max: consistency passes and mismatch warns", () => {
  const okCase = check({
    max_position_embeddings: 16384,
    rope_scaling: { rope_type: "yarn", factor: 4, original_max_position_embeddings: 4096 },
  });
  assert.equal(levelOf(okCase, /= declared max/), "ok");
  const mismatch = check({
    max_position_embeddings: 4096,
    rope_scaling: { rope_type: "yarn", factor: 4, original_max_position_embeddings: 4096 },
  });
  assert.equal(levelOf(mismatch, /runtimes that trust the ratio will disagree/), "warn");
});

test("yarn without original_max_position_embeddings warns about the HF fallback", () => {
  const r = check({ rope_scaling: { rope_type: "yarn", factor: 4 } });
  assert.equal(levelOf(r, /original_max_position_embeddings missing/), "warn");
});

test("yarn betas: inverted is an error with tallied counts, explicit sane betas get an ok", () => {
  const bad = check({
    rope_scaling: { rope_type: "yarn", factor: 16, beta_fast: 1, beta_slow: 32, attention_factor: -1 },
  });
  assert.equal(levelOf(bad, /ramp is inverted/), "error");
  assert.equal(bad.errors, 2); // inverted betas + attention_factor
  assert.ok(bad.warnings >= 1); // missing original_max
  assert.equal(bad.valid, false);
  const good = check({
    rope_scaling: { rope_type: "yarn", factor: 4, beta_fast: 64, beta_slow: 2 },
  });
  assert.equal(levelOf(good, /beta_fast 64 > beta_slow 2/), "ok");
});

test("yarn defaults are info findings, with the derived attention factor and the ramp", () => {
  const r = check({
    max_position_embeddings: 16384,
    rope_scaling: { rope_type: "yarn", factor: 4, original_max_position_embeddings: 4096 },
  });
  assert.equal(levelOf(r, /defaults 32\/1 apply/), "info");
  assert.equal(levelOf(r, /0\.1·ln\(4\) \+ 1 = 1\.14/), "info");
  assert.equal(levelOf(r, /ramp spans pairs/), "ok");
});

test("yarn attention_factor: ≤0 error, absurd warn, sane ok", () => {
  const mk = (attention_factor) =>
    check({ rope_scaling: { rope_type: "yarn", factor: 4, original_max_position_embeddings: 4096, attention_factor } });
  assert.equal(levelOf(mk(-1), /must be > 0/), "error");
  assert.equal(levelOf(mk(25), /far outside the fitted range/), "warn");
  assert.equal(levelOf(mk(1.14), /explicitly set/), "ok");
});

test("linear beyond 4× warns; --strict promotes that to a failing verdict", () => {
  const cfg = { rope_scaling: { rope_type: "linear", factor: 8 } };
  const r = check(cfg);
  assert.equal(levelOf(r, /beyond ~4× expect degraded retrieval/), "warn");
  assert.equal(r.valid, true); // warn, not error
  assert.equal(check(cfg, { strict: true }).valid, false);
});

test("dynamic NTK always warns about KV-cache drift", () => {
  const r = check({ rope_scaling: { rope_type: "dynamic", factor: 2 } });
  assert.equal(levelOf(r, /KV caches drift/), "warn");
});

test("llama3: missing band factors and inverted bands are errors", () => {
  const missing = check({
    rope_scaling: { rope_type: "llama3", factor: 8, original_max_position_embeddings: 4096 },
  });
  assert.equal(levelOf(missing, /requires a numeric "low_freq_factor"/), "error");
  const inverted = check({
    rope_scaling: {
      rope_type: "llama3",
      factor: 8,
      low_freq_factor: 4,
      high_freq_factor: 1,
      original_max_position_embeddings: 4096,
    },
  });
  assert.equal(levelOf(inverted, /wavelength bands are inverted/), "error");
});

test("llama3: factor is not treated as reach — declared max wins effectiveMax", () => {
  // The shipped Llama-3.1 pattern: factor 8 but max = 16 × original.
  const r = check({
    max_position_embeddings: 65536,
    rope_scaling: {
      rope_type: "llama3",
      factor: 8,
      low_freq_factor: 1,
      high_freq_factor: 4,
      original_max_position_embeddings: 4096,
    },
  });
  assert.equal(r.effectiveMax, 65536);
  assert.equal(levelOf(r, /frequency divisor, not a reach multiplier/), "info");
  assert.equal(r.valid, true);
});

test("fork keys and unknown keys warn but do not invalidate", () => {
  const r = check({
    max_position_embeddings: 16384,
    rope_scaling: {
      rope_type: "yarn",
      factor: 4,
      original_max_position_embeddings: 4096,
      mscale: 1.0,
      my_custom_knob: 7,
    },
  });
  assert.equal(levelOf(r, /fork-specific keys present \(mscale\)/), "warn");
  assert.equal(levelOf(r, /unrecognized rope_scaling keys \(my_custom_knob\)/), "warn");
  assert.equal(r.valid, true);
});

test("--target within/beyond the declared reach flips ok to error and the verdict", () => {
  const cfg = {
    max_position_embeddings: 16384,
    rope_scaling: { rope_type: "yarn", factor: 4, original_max_position_embeddings: 4096 },
  };
  const within = check(cfg, { targetCtx: 16384 });
  assert.equal(levelOf(within, /within the effective max/), "ok");
  assert.equal(within.valid, true);
  const beyond = check(cfg, { targetCtx: 32768 });
  assert.equal(levelOf(beyond, /exceeds what the declared scaling reaches/), "error");
  assert.equal(beyond.valid, false);
});
