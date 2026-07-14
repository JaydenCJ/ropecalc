/**
 * Shared test helpers: a raw-config factory with tiny hand-checkable
 * defaults, a temp-config writer, and a CLI runner that spawns the real
 * compiled binary. Everything is offline and deterministic — configs are
 * literal objects, the CLI reads only local files.
 */

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import path from "node:path";

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const CLI = path.join(ROOT, "dist", "cli.js");
export const EXAMPLES = path.join(ROOT, "examples");

/**
 * A tiny config whose RoPE math is paper-checkable: rotary dim 8
 * gives inv_freq = [1, 0.1, 0.01, 0.001] exactly at base 10000, and the
 * YaRN correction range over 4096 tokens is low=1, high=3 (derived in
 * yarn.test.mjs).
 */
export function tinyConfig(overrides = {}) {
  return {
    model_type: "test",
    hidden_size: 32,
    num_attention_heads: 4,
    head_dim: 8,
    rope_theta: 10000.0,
    max_position_embeddings: 4096,
    ...overrides,
  };
}

/** Write a config object to a fresh temp dir; returns the file path. */
export function writeConfig(obj, name = "config.json") {
  const dir = mkdtempSync(path.join(tmpdir(), "ropecalc-test-"));
  const file = path.join(dir, name);
  writeFileSync(file, JSON.stringify(obj, null, 2));
  return file;
}

/** Run the compiled CLI with the given argv; returns { status, stdout, stderr }. */
export function runCli(args, opts = {}) {
  const res = spawnSync(process.execPath, [CLI, ...args], {
    cwd: ROOT,
    encoding: "utf8",
    ...opts,
  });
  if (res.error) throw res.error;
  return { status: res.status, stdout: res.stdout, stderr: res.stderr };
}

/** assert.ok(|a - b| <= tol) with a readable failure message. */
export function assertClose(actual, expected, tol = 1e-9, label = "") {
  assert.ok(
    Math.abs(actual - expected) <= tol,
    `${label ? label + ": " : ""}expected ${expected} ± ${tol}, got ${actual}`,
  );
}
