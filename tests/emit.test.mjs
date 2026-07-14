/**
 * Emitters: the pasted output must be exactly what each runtime accepts —
 * pure JSON for HF, flag strings for llama.cpp and vLLM (with the vLLM
 * rope-scaling payload itself being valid JSON).
 */

import test from "node:test";
import assert from "node:assert/strict";
import { emit, emitHf, emitLlamaCpp, emitVllm, emitNotes, EmitError } from "../dist/emit.js";
import { computePlan } from "../dist/plan.js";
import { geometryFromFlags } from "../dist/config.js";

const g7b = geometryFromFlags({ dim: 128, base: 10000, ctx: 4096 });
const plan4x = computePlan(g7b, { targetCtx: 16384 });

test("hf yarn: a config.json patch with the full rope_scaling block", () => {
  const patch = JSON.parse(emitHf(plan4x, "yarn"));
  assert.deepEqual(patch, {
    max_position_embeddings: 16384,
    rope_scaling: {
      rope_type: "yarn",
      factor: 4,
      original_max_position_embeddings: 4096,
      beta_fast: 32,
      beta_slow: 1,
    },
  });
});

test("hf ntk: no rope_scaling block — a rope_theta override, as HF wants it", () => {
  const patch = JSON.parse(emitHf(plan4x, "ntk"));
  assert.equal(patch.rope_theta, 40889.9);
  assert.equal(patch.max_position_embeddings, 16384);
  assert.ok(!("rope_scaling" in patch));
});

test("hf linear: rope_type linear with the bare factor", () => {
  const patch = JSON.parse(emitHf(plan4x, "linear"));
  assert.deepEqual(patch.rope_scaling, { rope_type: "linear", factor: 4 });
});

test("llama.cpp flags per method", () => {
  assert.equal(
    emitLlamaCpp(plan4x, "yarn"),
    "--ctx-size 16384 --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 4096",
  );
  assert.equal(emitLlamaCpp(plan4x, "ntk"), "--ctx-size 16384 --rope-freq-base 40889.9");
  assert.equal(
    emitLlamaCpp(plan4x, "linear"),
    "--ctx-size 16384 --rope-scaling linear --rope-scale 4",
  );
});

test("vllm: the inline rope-scaling payload parses as JSON", () => {
  const out = emitVllm(plan4x, "yarn");
  assert.match(out, /^--max-model-len 16384 --rope-scaling '/);
  const payload = JSON.parse(/--rope-scaling '(.+)'$/.exec(out)[1]);
  assert.equal(payload.rope_type, "yarn");
  assert.equal(payload.factor, 4);
  assert.equal(payload.original_max_position_embeddings, 4096);
  assert.equal(emitVllm(plan4x, "ntk"), "--max-model-len 16384 --rope-theta 40889.9");
});

test("emit refuses a plan that computed nothing (factor ≤ 1)", () => {
  const noop = computePlan(g7b, { targetCtx: 2048 });
  assert.throws(() => emit(noop, "yarn", "hf"), EmitError);
});

test("notes: double-scaling and headroom traps are called out, on the side", () => {
  assert.match(emitNotes(plan4x, "yarn", "llamacpp").join("\n"), /do not add it twice/);
  assert.match(emitNotes(plan4x, "ntk", "hf").join("\n"), /rope_theta override/);
  assert.match(emitNotes(plan4x, "linear", "hf").join("\n"), /degrades retrieval/);
});
