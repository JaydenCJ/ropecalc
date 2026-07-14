/**
 * config.json normalization: key-driven derivation, nesting, hard errors
 * for structural impossibilities, and soft capture of questionable
 * rope_scaling values (which are check's job to judge, not the loader's).
 */

import test from "node:test";
import assert from "node:assert/strict";
import { normalizeConfig, parseScalingBlock, geometryFromFlags, ConfigError } from "../dist/config.js";
import { tinyConfig } from "./helpers.mjs";

test("explicit head_dim wins silently; hidden_size/heads derives with a warning", () => {
  const explicit = normalizeConfig(tinyConfig(), "t");
  assert.equal(explicit.headDim, 8);
  assert.equal(explicit.pairs, 4);
  assert.deepEqual(explicit.warnings, []);
  const cfg = tinyConfig({ hidden_size: 4096, num_attention_heads: 32 });
  delete cfg.head_dim;
  const derived = normalizeConfig(cfg, "t");
  assert.equal(derived.headDim, 128);
  assert.match(derived.warnings.join("\n"), /derived.*128/);
});

test("underivable or non-dividing head_dim is a hard ConfigError", () => {
  assert.throws(
    () => normalizeConfig({ max_position_embeddings: 4096 }, "t"),
    /cannot determine head_dim/,
  );
  const cfg = tinyConfig({ hidden_size: 100, num_attention_heads: 3 });
  delete cfg.head_dim;
  assert.throws(() => normalizeConfig(cfg, "t"), /does not divide/);
});

test("partial_rotary_factor shrinks the rotary dim; an odd result is rejected", () => {
  // head_dim 8 × 0.5 = rotary 4 → 2 pairs (GPT-NeoX style).
  const g = normalizeConfig(tinyConfig({ partial_rotary_factor: 0.5 }), "t");
  assert.equal(g.rotaryDim, 4);
  assert.equal(g.pairs, 2);
  // head_dim 10 × 0.7 = 7: RoPE rotates pairs, an odd subspace cannot exist.
  assert.throws(
    () => normalizeConfig(tinyConfig({ head_dim: 10, partial_rotary_factor: 0.7 }), "t"),
    /must be even/,
  );
});

test("rope_theta defaults to 10000 with a warning; base ≤ 1 is rejected", () => {
  const cfg = tinyConfig();
  delete cfg.rope_theta;
  const g = normalizeConfig(cfg, "t");
  assert.equal(g.base, 10000);
  assert.match(g.warnings.join("\n"), /assuming the RoPE default 10000/);
  assert.throws(() => normalizeConfig(tinyConfig({ rope_theta: 1 }), "t"), /must be > 1/);
});

test("text_config nesting: inner keys win over outer (multimodal configs)", () => {
  const g = normalizeConfig(
    {
      hidden_size: 999, // decoy at the top level
      text_config: tinyConfig({ rope_theta: 500000 }),
    },
    "t",
  );
  assert.equal(g.headDim, 8);
  assert.equal(g.base, 500000);
});

test("trained ctx prefers the block's original_max_position_embeddings; none at all is fatal", () => {
  const g = normalizeConfig(
    tinyConfig({
      max_position_embeddings: 131072,
      rope_scaling: { rope_type: "yarn", factor: 4.0, original_max_position_embeddings: 32768 },
    }),
    "t",
  );
  assert.equal(g.trainedCtx, 32768);
  assert.equal(g.declaredMax, 131072);
  const bare = tinyConfig();
  delete bare.max_position_embeddings;
  assert.throws(() => normalizeConfig(bare, "t"), /cannot determine trained context/);
});

test("parseScalingBlock: rope_type wins over legacy type; conflicts are flagged", () => {
  const a = parseScalingBlock({ rope_type: "yarn", type: "linear", factor: 4 });
  assert.equal(a.type, "yarn");
  assert.equal(a.conflictingTypeKeys, true);
  const b = parseScalingBlock({ type: "linear", factor: 4 });
  assert.equal(b.type, "linear");
  assert.equal(b.usedLegacyKey, true);
});

test("parseScalingBlock sorts non-numeric, fork and unknown keys into their bins", () => {
  const s = parseScalingBlock({
    rope_type: "yarn",
    factor: "4", // string, not a number
    mscale: 1.0,
    surprise_key: true,
  });
  assert.deepEqual(s.invalidKeys, ["factor"]);
  assert.deepEqual(s.forkKeys, ["mscale"]);
  assert.deepEqual(s.unknownKeys, ["surprise_key"]);
  assert.equal(s.factor, undefined);
});

test("geometryFromFlags: scaling-free geometry, parity check, defaulted base", () => {
  const g = geometryFromFlags({ dim: 128, base: 500000, ctx: 8192 });
  assert.equal(g.source, "flags");
  assert.equal(g.pairs, 64);
  assert.equal(g.scaling, null);
  assert.throws(() => geometryFromFlags({ dim: 127, ctx: 8192 }), ConfigError);
  const defaulted = geometryFromFlags({ dim: 8, ctx: 4096 });
  assert.equal(defaulted.base, 10000);
  assert.match(defaulted.warnings.join("\n"), /assuming the RoPE default/);
});
