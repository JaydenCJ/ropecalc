/**
 * The dims engine: per-pair rows, zone assignment, and the row-elision
 * logic that keeps wide heads readable without hiding the ramp.
 */

import test from "node:test";
import assert from "node:assert/strict";
import { computeDims, selectRows, ELIDE_THRESHOLD } from "../dist/dims.js";
import { geometryFromFlags } from "../dist/config.js";
import { assertClose } from "./helpers.mjs";

const gTiny = geometryFromFlags({ dim: 8, base: 10000, ctx: 4096 });
const g7b = geometryFromFlags({ dim: 128, base: 10000, ctx: 4096 });

test("tiny geometry rows carry the hand-derived divisors (s=2)", () => {
  const d = computeDims(gTiny, { targetCtx: 8192, betaFast: 32, betaSlow: 1 });
  assert.equal(d.rows.length, 4);
  // yarn scaled freqs are [1, 0.1, 0.0075, 0.0005] → divisors 1, 1, 4/3, 2.
  assertClose(d.rows[0].yarnDiv, 1, 1e-12);
  assertClose(d.rows[2].yarnDiv, 0.01 / 0.0075, 1e-12);
  assertClose(d.rows[3].yarnDiv, 2, 1e-12);
  // linear divides everything alike; ntk hits exactly s on the last pair.
  assert.ok(d.rows.every((r) => r.linearDiv === 2));
  assertClose(d.rows[3].ntkDiv, 2, 1e-12);
});

test("zones follow the mask: keep / blend / interp", () => {
  const d = computeDims(gTiny, { targetCtx: 8192, betaFast: 32, betaSlow: 1 });
  assert.deepEqual(
    d.rows.map((r) => r.zone),
    ["keep", "keep", "blend", "interp"],
  );
});

test("rows expose wavelength and trained-context rotations", () => {
  const d = computeDims(gTiny, { targetCtx: 8192, betaFast: 32, betaSlow: 1 });
  assertClose(d.rows[0].wavelength, 2 * Math.PI, 1e-12);
  assertClose(d.rows[0].rotationsAtTrained, 4096 / (2 * Math.PI), 1e-9);
  // Pair 3 (λ = 2000π ≈ 6283) never completes a rotation over 4096 tokens.
  assert.ok(d.rows[3].rotationsAtTrained < 1);
});

test("dims result carries the ramp and mscale for the header line", () => {
  const d = computeDims(g7b, { targetCtx: 16384, betaFast: 32, betaSlow: 1 });
  assert.equal(d.low, 20);
  assert.equal(d.high, 46);
  assertClose(d.mscale, 1.1386, 1e-4);
  assert.equal(d.factor, 4);
});

test("selectRows: small tables and --all are complete and gap-free", () => {
  assert.deepEqual(selectRows(4, 1, 3, false), [0, 1, 2, 3]);
  const all = selectRows(64, 20, 46, true);
  assert.equal(all.length, 64);
  assert.ok(!all.includes("gap"));
  assert.ok(64 > ELIDE_THRESHOLD); // the case below genuinely elides
});

test("selectRows: elision keeps both edges and the full ramp neighborhood", () => {
  const picks = selectRows(64, 20, 46, false);
  const rows = picks.filter((p) => p !== "gap");
  for (const must of [0, 3, 19, 20, 46, 47, 60, 63]) {
    assert.ok(rows.includes(must), `pair ${must} must survive elision`);
  }
  assert.ok(picks.includes("gap"));
  assert.ok(rows.length < 64);
  // Sorted and duplicate-free — a shuffled table would be worse than a long one.
  const sorted = [...rows].sort((a, b) => a - b);
  assert.deepEqual(rows, sorted);
  assert.equal(new Set(rows).size, rows.length);
});

test("selectRows: gap markers sit exactly where indices jump", () => {
  const picks = selectRows(64, 20, 46, false);
  for (let i = 1; i < picks.length; i++) {
    const prev = picks[i - 1];
    const cur = picks[i];
    if (typeof prev === "number" && typeof cur === "number") {
      assert.equal(cur, prev + 1, `contiguous run broken without a gap at ${prev}→${cur}`);
    }
  }
});
