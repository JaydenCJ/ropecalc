/**
 * CLI integration: the real compiled binary, spawned per test, against the
 * committed examples and temp-dir configs. Pins the exit-code contract,
 * stdout/stderr separation, and --json shapes.
 */

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { runCli, writeConfig, tinyConfig, ROOT, EXAMPLES } from "./helpers.mjs";

/** Overwrite a written config with syntactically invalid JSON. */
function corruptFile(file) {
  writeFileSync(file, "{not json");
  return file;
}

const BASE7B = path.join(EXAMPLES, "base-10k-7b.json");
const YARN = path.join(EXAMPLES, "yarn-128k.json");
const BROKEN = path.join(EXAMPLES, "broken-rope.json");

test("--version matches package.json; --help documents the whole surface", () => {
  const pkg = JSON.parse(readFileSync(path.join(ROOT, "package.json"), "utf8"));
  const v = runCli(["--version"]);
  assert.equal(v.status, 0);
  assert.equal(v.stdout.trim(), pkg.version);
  const h = runCli(["--help"]);
  assert.equal(h.status, 0);
  for (const word of ["plan", "check", "dims", "methods", "--target", "--emit", "Exit codes"]) {
    assert.ok(h.stdout.includes(word), word);
  }
  assert.equal(runCli([]).status, 2); // bare invocation prints help but fails
});

test("usage and config errors exit 2: bad flags, bad commands, bad files", () => {
  assert.equal(runCli(["plan", BASE7B, "--frobnicate"]).status, 2);
  assert.equal(runCli(["explode", BASE7B]).status, 2);
  assert.equal(runCli(["plan", "--target", "16k"]).status, 2); // no config, no --dim/--ctx
  assert.equal(runCli(["plan", "does-not-exist.json", "--target", "16k"]).status, 2);
  const notJson = corruptFile(writeConfig({}));
  assert.equal(runCli(["plan", notJson, "--target", "16k"]).status, 2); // invalid JSON
  // Cross-command flags are rejected, not ignored.
  assert.equal(runCli(["plan", BASE7B, "--target", "16k", "--strict"]).status, 2);
  assert.equal(runCli(["check", YARN, "--emit", "hf"]).status, 2);
});

test("plan: the flagship 7B 4× run prints all three methods and exits 0", () => {
  const r = runCli(["plan", BASE7B, "--target", "16k"]);
  assert.equal(r.status, 0);
  for (const want of ["rope_theta 10000 → 40889.9", "ramp pairs 20…46 of 64", "recommend yarn"]) {
    assert.ok(r.stdout.includes(want), want);
  }
});

test("plan --json: machine-readable, same numbers, byte-identical across runs", () => {
  const a = runCli(["plan", BASE7B, "--target", "16k", "--json"]);
  assert.equal(a.status, 0);
  const p = JSON.parse(a.stdout);
  assert.equal(p.tool, "ropecalc");
  assert.equal(p.factor, 4);
  assert.equal(p.ntk.scaledBase, 40889.9);
  assert.deepEqual([p.yarn.low, p.yarn.high], [20, 46]);
  const b = runCli(["plan", BASE7B, "--target", "16k", "--json"]);
  assert.equal(a.stdout, b.stdout);
});

test("plan in flag mode matches the file-based run; mixing both is rejected", () => {
  const file = runCli(["plan", BASE7B, "--target", "16k", "--json"]);
  const flags = runCli([
    "plan", "--dim", "128", "--base", "10000", "--ctx", "4096", "--target", "16k", "--json",
  ]);
  const a = JSON.parse(file.stdout);
  const b = JSON.parse(flags.stdout);
  assert.deepEqual(a.yarn, b.yarn);
  assert.deepEqual(a.ntk, b.ntk);
  const mixed = runCli(["plan", BASE7B, "--dim", "128", "--target", "16k"]);
  assert.equal(mixed.status, 2);
  assert.match(mixed.stderr, /not both/);
});

test("plan --emit: pure payload on stdout, notes on stderr, recommendation as default", () => {
  const hf = runCli(["plan", BASE7B, "--target", "16k", "--emit", "hf", "--method", "ntk"]);
  assert.equal(hf.status, 0);
  const patch = JSON.parse(hf.stdout); // throws if anything but JSON leaked
  assert.equal(patch.rope_theta, 40889.9);
  assert.match(hf.stderr, /rope_theta override/);
  // Without --method the recommendation (yarn at 4×) is emitted.
  const llama = runCli(["plan", BASE7B, "--target", "16k", "--emit", "llamacpp"]);
  assert.equal(
    llama.stdout.trim(),
    "--ctx-size 16384 --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 4096",
  );
});

test("plan --emit validates its arguments and refuses no-op targets", () => {
  assert.equal(runCli(["plan", BASE7B, "--target", "16k", "--emit", "ollama"]).status, 2);
  assert.equal(runCli(["plan", BASE7B, "--target", "16k", "--method", "llama3"]).status, 2);
  assert.equal(runCli(["plan", BASE7B, "--target", "2k", "--emit", "hf"]).status, 2);
});

test("check: committed yarn example VALID exit 0, broken example INVALID exit 1", () => {
  const good = runCli(["check", YARN]);
  assert.equal(good.status, 0);
  assert.match(good.stdout, /ramp spans pairs 23…40 of 64/);
  assert.match(good.stdout, /VALID/);
  const bad = runCli(["check", BROKEN]);
  assert.equal(bad.status, 1);
  assert.match(bad.stdout, /ramp is inverted/);
  assert.match(bad.stdout, /INVALID — 2 errors/);
});

test("check gates: --strict flips warnings to exit 1, --target checks reach", () => {
  const cfg = writeConfig(tinyConfig({ rope_scaling: { rope_type: "linear", factor: 8 } }));
  assert.equal(runCli(["check", cfg]).status, 0);
  assert.equal(runCli(["check", cfg, "--strict"]).status, 1);
  assert.equal(runCli(["check", YARN, "--target", "128k"]).status, 0);
  assert.equal(runCli(["check", YARN, "--target", "256k"]).status, 1);
});

test("check --json carries findings and counts; geometry flags are refused", () => {
  const r = runCli(["check", BROKEN, "--json"]);
  assert.equal(r.status, 1);
  const c = JSON.parse(r.stdout);
  assert.equal(c.valid, false);
  assert.equal(c.errors, 2);
  assert.ok(c.findings.some((f) => f.level === "error"));
  assert.equal(runCli(["check", "--dim", "128", "--ctx", "4096"]).status, 2);
});

test("dims: elided by default, complete with --all, target defaults to declared reach", () => {
  const r = runCli(["dims", BASE7B, "--target", "16k"]);
  assert.equal(r.status, 0);
  assert.match(r.stdout, /ramp 20…46/);
  for (const zone of ["keep", "blend", "interp"]) assert.ok(r.stdout.includes(zone), zone);
  assert.ok(r.stdout.includes("--all prints every pair"));
  const all = runCli(["dims", BASE7B, "--target", "16k", "--all"]);
  const pairLines = all.stdout.split("\n").filter((l) => /^ *\d+ /.test(l));
  assert.equal(pairLines.length, 64);
  // A config that declares scaling brings its own reach…
  const scaled = runCli(["dims", YARN]);
  assert.equal(scaled.status, 0);
  assert.match(scaled.stderr, /using the config's declared reach 131072/);
  assert.match(scaled.stdout, /factor 4\.00×/);
  // …a llama3 config's reach is its declared max, not trainedCtx × factor
  // (the factor is a frequency divisor — the trap ropecalc exists to catch)…
  const llama3 = runCli(["dims", path.join(EXAMPLES, "llama3-extended.json")]);
  assert.match(llama3.stderr, /using the config's declared reach 131072/);
  // …and an unscaled one requires --target.
  assert.equal(runCli(["dims", BASE7B]).status, 2);
});

test("methods reference prints receipts; normalization notes go to stderr only", () => {
  const text = runCli(["methods"]);
  assert.equal(text.status, 0);
  assert.ok(text.stdout.includes("arXiv:2306.15595"));
  assert.equal(JSON.parse(runCli(["methods", "--json"]).stdout).methods.length, 5);
  // Flags that don't apply are rejected, not silently ignored.
  assert.equal(runCli(["methods", "--target", "16k"]).status, 2);
  const cfg = tinyConfig();
  delete cfg.rope_theta;
  const r = runCli(["plan", writeConfig(cfg), "--target", "8k"]);
  assert.equal(r.status, 0);
  assert.match(r.stderr, /assuming the RoPE default 10000/);
  assert.ok(!r.stdout.includes("assuming"));
});
