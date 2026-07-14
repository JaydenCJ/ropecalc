/**
 * Text renderers: plain-data results in, aligned monospace text out.
 * No I/O, no colors, no locale-dependent formatting — identical inputs must
 * render byte-identical output (tested), because people pipe this to grep.
 */

import { METHODS } from "./methods.js";
import { fmt2, fmtAdaptive, fmtCtx } from "./units.js";
import { selectRows } from "./dims.js";
import { VERSION } from "./version.js";
import type { CheckResult, DimsResult, MethodId, Plan, RopeGeometry } from "./types.js";

const LABEL = 10;

function line(label: string, text: string): string {
  return label.padEnd(LABEL) + text;
}

function noteLines(notes: string[]): string[] {
  return notes.map((n) => " ".repeat(LABEL) + "· " + n);
}

function ropeLine(geom: RopeGeometry): string {
  const rotary =
    geom.rotaryDim === geom.headDim ? `${geom.rotaryDim}` : `${geom.rotaryDim} (partial)`;
  return line(
    "rope",
    `head_dim ${geom.headDim} · rotary ${rotary} · base ${geom.base} · trained ctx ${fmtCtx(geom.trainedCtx)}`,
  );
}

/** Render `ropecalc plan` output. `only` narrows to a single method block. */
export function renderPlan(plan: Plan, geom: RopeGeometry, only: MethodId | null): string {
  const out: string[] = [`ropecalc ${VERSION} — scaling plan`, ""];
  out.push(line("model", plan.source));
  out.push(ropeLine(geom));
  out.push(line("target", `${fmtCtx(plan.targetCtx)} · factor ${fmt2(plan.factor)}×`));
  out.push("");

  if (plan.linear === null) {
    out.push(line("recommend", `none — ${plan.recommendation.reason}`));
  } else {
    const show = (m: MethodId) => only === null || only === m;
    if (show("linear")) {
      out.push(line("linear", `factor ${fmt2(plan.linear.factor)}`));
      out.push(...noteLines(plan.linear.notes), "");
    }
    if (show("ntk")) {
      const n = plan.ntk!;
      out.push(
        line(
          "ntk",
          `rope_theta ${plan.base} → ${n.scaledBase} (= base × ${fmt2(plan.factor)}^(${plan.rotaryDim}/${plan.rotaryDim - 2}))`,
        ),
      );
      out.push(...noteLines(n.notes), "");
    }
    if (show("yarn")) {
      const y = plan.yarn!;
      out.push(
        line(
          "yarn",
          `factor ${fmt2(y.factor)} · beta ${y.betaFast}/${y.betaSlow} · ramp pairs ${y.low}…${y.high} of ${plan.rotaryDim / 2} · mscale ${fmt2(y.mscale)}`,
        ),
      );
      out.push(
        " ".repeat(LABEL) +
          `zones: ${y.zones.keep} kept · ${y.zones.blend} blended · ${y.zones.interpolate} interpolated`,
      );
      out.push(...noteLines(y.notes), "");
    }
    out.push(line("recommend", `${plan.recommendation.method} — ${plan.recommendation.reason}`));
  }

  for (const w of plan.warnings) out.push(line("note", w));
  return out.join("\n");
}

/** Render `ropecalc check` output. */
export function renderCheck(result: CheckResult, geom: RopeGeometry): string {
  const out: string[] = [`ropecalc ${VERSION} — rope_scaling check`, ""];
  out.push(line("model", result.source));
  out.push(ropeLine(geom));
  const declared = result.declaredMax === null ? "unset" : fmtCtx(result.declaredMax);
  out.push(
    line(
      "scaling",
      `${result.scalingType ?? "none"} · declared max ${declared} · effective max ${fmtCtx(result.effectiveMax)}`,
    ),
  );
  out.push("");
  for (const f of result.findings) {
    out.push("  " + f.level.padEnd(LABEL - 2) + f.message);
  }
  out.push("");
  out.push(
    line(
      "verdict",
      `${result.valid ? "VALID" : "INVALID"} — ${result.errors} error${result.errors === 1 ? "" : "s"} · ${result.warnings} warning${result.warnings === 1 ? "" : "s"}`,
    ),
  );
  return out.join("\n");
}

const GAP_LINE = "   …";

/** Render `ropecalc dims` output. */
export function renderDims(dims: DimsResult, all: boolean): string {
  const out: string[] = [`ropecalc ${VERSION} — dimension table`, ""];
  out.push(line("model", dims.source));
  out.push(
    line("rope", `rotary ${dims.rotaryDim} (${dims.pairs} pairs) · base ${dims.base} · trained ctx ${fmtCtx(dims.trainedCtx)}`),
  );
  out.push(
    line(
      "target",
      `${fmtCtx(dims.targetCtx)} · factor ${fmt2(dims.factor)}× · yarn beta ${dims.betaFast}/${dims.betaSlow} → ramp ${dims.low}…${dims.high} · mscale ${fmt2(dims.mscale)}`,
    ),
  );
  out.push("");

  const header =
    "pair".padStart(4) +
    "wavelength".padStart(12) +
    "rot@trained".padStart(13) +
    "   " +
    "zone".padEnd(8) +
    "linear".padStart(7) +
    "ntk".padStart(8) +
    "yarn".padStart(8);
  out.push(header);

  for (const pick of selectRows(dims.pairs, dims.low, dims.high, all)) {
    if (pick === "gap") {
      out.push(GAP_LINE);
      continue;
    }
    const r = dims.rows[pick]!;
    out.push(
      String(r.pair).padStart(4) +
        fmtAdaptive(r.wavelength).padStart(12) +
        fmtAdaptive(r.rotationsAtTrained).padStart(13) +
        "   " +
        r.zone.padEnd(8) +
        ("÷" + fmt2(r.linearDiv)).padStart(7) +
        ("÷" + fmt2(r.ntkDiv)).padStart(8) +
        ("÷" + fmt2(r.yarnDiv)).padStart(8),
    );
  }

  out.push("");
  out.push("zone/yarn columns use the beta ramp above; ÷n = frequency divided by n at the target");
  if (!all && out.some((l) => l === GAP_LINE)) {
    out.push("rows elided around the uneventful middle — --all prints every pair");
  }
  return out.join("\n");
}

/** Render `ropecalc methods` output. */
export function renderMethods(): string {
  const out: string[] = [`ropecalc ${VERSION} — method reference`, ""];
  for (const m of METHODS) {
    const hf = m.hfType === "—" ? "no rope_type (set rope_theta)" : `rope_type "${m.hfType}"`;
    out.push(line(m.id, `${m.name} · ${hf}${m.plannable ? " · plannable" : ""}`));
    out.push(line("", `formula    ${m.formula}`));
    out.push(line("", `params     ${m.params}`));
    out.push(line("", `origin     ${m.provenance}`));
    out.push("");
  }
  out.push("plannable = `ropecalc plan` computes it; the rest are validated by `check` and shown by `dims`");
  return out.join("\n");
}
