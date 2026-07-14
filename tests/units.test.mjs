/**
 * Context-length parsing and deterministic formatting. These are the values
 * users type on the command line — a silent misparse here corrupts every
 * downstream number.
 */

import test from "node:test";
import assert from "node:assert/strict";
import { parseCtx, fmtCtx, fmtCtxShort, fmt2, fmtAdaptive, roundTo } from "../dist/units.js";

test("parseCtx accepts integers, k (×1024) and m (×1024²) forms", () => {
  assert.equal(parseCtx("4096"), 4096);
  assert.equal(parseCtx("1"), 1);
  assert.equal(parseCtx("16k"), 16384);
  assert.equal(parseCtx("128K"), 131072); // the convention 128k = 131072
  assert.equal(parseCtx("0.5k"), 512); // fractional k is fine if tokens come out whole
  assert.equal(parseCtx("1m"), 1048576);
});

test("parseCtx rejects garbage, negatives and non-integer token counts", () => {
  for (const bad of ["", "abc", "-4096", "4096.5", "12kb", "k", "1e4"]) {
    assert.throws(() => parseCtx(bad), /cannot parse|not a positive whole number/, bad);
  }
});

test("fmtCtx/fmtCtxShort annotate clean multiples of 1024 and leave the rest alone", () => {
  assert.equal(fmtCtx(131072), "131072 (128k)");
  assert.equal(fmtCtx(4096), "4096 (4k)");
  assert.equal(fmtCtx(5000), "5000");
  assert.equal(fmtCtx(100), "100");
  assert.equal(fmtCtxShort(131072), "128k");
  assert.equal(fmtCtxShort(5000), "5000");
});

test("numeric formatters: fixed fmt2, magnitude-adaptive, and float-noise-free roundTo", () => {
  assert.equal(fmt2(4), "4.00");
  assert.equal(fmt2(1.138629), "1.14");
  assert.equal(fmtAdaptive(6.283185), "6.28"); // 2π, the shortest wavelength
  assert.equal(fmtAdaptive(651.8986), "651.9"); // rotations of pair 0 over 4096
  assert.equal(fmtAdaptive(54410), "54410");
  assert.equal(roundTo(3.9999999996, 4), 4); // numbers, not strings
  assert.equal(roundTo(1.1386294361, 4), 1.1386);
});
