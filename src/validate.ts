/**
 * The `check` rule engine: audits a config's rope_scaling block against the
 * ranges and invariants each method actually requires, plus the known
 * cross-runtime pitfalls. Produces findings, never throws on bad values —
 * `ok` findings record what passed so a clean bill of health is legible too.
 *
 * Severity contract:
 *   error — a runtime will reject this config, compute garbage, or the
 *           declared numbers contradict each other
 *   warn  — loads fine but is a known quality or portability trap
 *   info  — a default is being relied on; stated so it is a choice
 *   ok    — an invariant was checked and holds
 */

import { KNOWN_SCALING_TYPES, llama3Zones } from "./methods.js";
import {
  correctionRange,
  DEFAULT_BETA_FAST,
  DEFAULT_BETA_SLOW,
  yarnMscale,
  yarnZones,
} from "./yarn.js";
import { fmtCtx, fmt2 } from "./units.js";
import type { CheckResult, Finding, RopeGeometry } from "./types.js";

export interface CheckOptions {
  /** Also verify the config reaches this context. */
  targetCtx?: number;
  /** Treat warnings as failures. */
  strict?: boolean;
}

function ok(findings: Finding[], message: string): void {
  findings.push({ level: "ok", message });
}
function info(findings: Finding[], message: string): void {
  findings.push({ level: "info", message });
}
function warn(findings: Finding[], message: string): void {
  findings.push({ level: "warn", message });
}
function error(findings: Finding[], message: string): void {
  findings.push({ level: "error", message });
}

const FACTOR_TYPES = new Set(["linear", "dynamic", "yarn", "llama3"]);

/** Audit one geometry. Pure: findings in, findings out, no I/O. */
export function checkConfig(geom: RopeGeometry, opts: CheckOptions = {}): CheckResult {
  const findings: Finding[] = [];
  const s = geom.scaling;

  let effectiveMax = geom.trainedCtx;

  if (s === null) {
    ok(findings, `no rope_scaling block — positions are used exactly as trained`);
  } else {
    // --- type key hygiene -------------------------------------------------
    if (s.type === "") {
      error(findings, `rope_scaling has neither "rope_type" nor "type" — runtimes will reject it`);
    } else if (!(KNOWN_SCALING_TYPES as readonly string[]).includes(s.type)) {
      error(
        findings,
        `unknown rope_scaling type "${s.type}" (known: ${KNOWN_SCALING_TYPES.join(", ")})`,
      );
    } else {
      ok(findings, `rope_scaling type "${s.type}" is a recognized scaling scheme`);
    }
    if (s.conflictingTypeKeys) {
      error(
        findings,
        `"type" and "rope_type" are both present and disagree — behavior depends on the runtime`,
      );
    } else if (s.usedLegacyKey) {
      info(findings, `block uses the legacy "type" key — modern runtimes also accept "rope_type"`);
    }
    for (const key of s.invalidKeys) {
      error(findings, `rope_scaling.${key} is not a finite number`);
    }
    if (s.forkKeys.length > 0) {
      warn(
        findings,
        `fork-specific keys present (${s.forkKeys.join(", ")}) — vanilla transformers ignores them silently`,
      );
    }
    if (s.unknownKeys.length > 0) {
      warn(
        findings,
        `unrecognized rope_scaling keys (${s.unknownKeys.join(", ")}) — runtimes drop what they don't know`,
      );
    }

    // --- factor -----------------------------------------------------------
    const knownType = (KNOWN_SCALING_TYPES as readonly string[]).includes(s.type);
    if (FACTOR_TYPES.has(s.type)) {
      if (s.factor === undefined) {
        if (!s.invalidKeys.includes("factor")) {
          error(findings, `type "${s.type}" requires a numeric "factor" and none is present`);
        }
      } else if (s.factor < 1) {
        error(findings, `factor ${s.factor} is < 1 — scaling must extend, not shrink, the context`);
      } else if (s.factor === 1) {
        warn(findings, `factor 1 is a no-op — delete the block or set the real factor`);
      } else {
        ok(findings, `factor ${s.factor} is in range`);
      }
    }

    // --- factor × original vs declared max ---------------------------------
    // For llama3-type blocks the factor is a per-band frequency divisor, not
    // a reach multiplier: the shipped configs pair factor 8 with a 16× larger
    // declared max, because the reach comes from long-context fine-tuning.
    const factorIsReach = s.type !== "llama3";
    const orig = s.originalMaxPositionEmbeddings;
    if (s.factor !== undefined && s.factor >= 1) {
      effectiveMax = factorIsReach
        ? Math.round(geom.trainedCtx * s.factor)
        : (geom.declaredMax ?? geom.trainedCtx);
    }
    if (orig !== undefined && s.factor !== undefined && geom.declaredMax !== null && factorIsReach) {
      const product = Math.round(orig * s.factor);
      if (product === geom.declaredMax) {
        ok(
          findings,
          `factor ${s.factor} × original ${fmtCtx(orig)} = declared max ${fmtCtx(geom.declaredMax)}`,
        );
      } else {
        warn(
          findings,
          `factor ${s.factor} × original ${fmtCtx(orig)} = ${fmtCtx(product)}, but max_position_embeddings is ${fmtCtx(geom.declaredMax)} — transformers trusts the factor, runtimes that trust the ratio will disagree`,
        );
      }
    }
    if (orig === undefined && knownType && s.type !== "default") {
      if (s.type === "llama3") {
        error(
          findings,
          `llama3 scaling requires original_max_position_embeddings and it is missing`,
        );
      } else if (s.type === "yarn") {
        warn(
          findings,
          `original_max_position_embeddings missing — transformers falls back to max_position_embeddings (${geom.declaredMax ?? "unset"}), which is wrong once that key holds the extended length`,
        );
      }
    }

    // --- per-type rules -----------------------------------------------------
    if (s.type === "linear" && s.factor !== undefined && s.factor > 4) {
      warn(
        findings,
        `linear factor ${s.factor} compresses every dimension ${s.factor}× — beyond ~4× expect degraded retrieval unless the model was fine-tuned at this factor`,
      );
    }
    if (s.type === "dynamic") {
      warn(
        findings,
        `dynamic NTK re-derives the base as the sequence grows — keys cached early were rotated under a different base, so long-lived KV caches drift`,
      );
    }
    if (s.type === "yarn") {
      checkYarn(geom, findings);
    }
    if (s.type === "llama3") {
      checkLlama3(geom, findings);
    }
  }

  // --- target reachability ----------------------------------------------
  if (opts.targetCtx !== undefined) {
    if (opts.targetCtx <= effectiveMax) {
      ok(findings, `target ${fmtCtx(opts.targetCtx)} is within the effective max ${fmtCtx(effectiveMax)}`);
    } else if (s === null) {
      error(
        findings,
        `target ${fmtCtx(opts.targetCtx)} exceeds the trained context ${fmtCtx(geom.trainedCtx)} and no scaling is declared — run \`ropecalc plan\``,
      );
    } else {
      error(
        findings,
        `target ${fmtCtx(opts.targetCtx)} exceeds what the declared scaling reaches (${fmtCtx(effectiveMax)})`,
      );
    }
  }

  const errors = findings.filter((f) => f.level === "error").length;
  const warnings = findings.filter((f) => f.level === "warn").length;
  const valid = errors === 0 && !(opts.strict === true && warnings > 0);

  return {
    tool: "ropecalc",
    source: geom.source,
    scalingType: s === null ? null : s.type || null,
    trainedCtx: geom.trainedCtx,
    declaredMax: geom.declaredMax,
    effectiveMax,
    findings,
    errors,
    warnings,
    valid,
  };
}

function checkYarn(geom: RopeGeometry, findings: Finding[]): void {
  const s = geom.scaling!;
  const betaFast = s.betaFast;
  const betaSlow = s.betaSlow;
  if (betaFast === undefined && betaSlow === undefined) {
    info(findings, `beta_fast/beta_slow unset — the defaults ${DEFAULT_BETA_FAST}/${DEFAULT_BETA_SLOW} apply`);
  }
  const bf = betaFast ?? DEFAULT_BETA_FAST;
  const bs = betaSlow ?? DEFAULT_BETA_SLOW;
  let betasUsable = true;
  if (bf <= 0 || bs <= 0) {
    error(findings, `beta_fast/beta_slow must be > 0 (got ${bf}/${bs})`);
    betasUsable = false;
  } else if (bf <= bs) {
    error(
      findings,
      `beta_fast (${bf}) must be greater than beta_slow (${bs}) — as given, the ramp is inverted`,
    );
    betasUsable = false;
  } else if (betaFast !== undefined || betaSlow !== undefined) {
    ok(findings, `beta_fast ${bf} > beta_slow ${bs}`);
  }

  if (betasUsable) {
    const { low, high } = correctionRange(bf, bs, geom.rotaryDim, geom.base, geom.trainedCtx);
    const zones = yarnZones(low, high, geom.pairs);
    if (zones.blend === 0 && (zones.keep === 0 || zones.interpolate === 0)) {
      warn(
        findings,
        `correction range is degenerate for this geometry — every pair lands in one zone (ramp ${low}…${high})`,
      );
    } else {
      ok(
        findings,
        `ramp spans pairs ${low}…${high} of ${geom.pairs}: ${zones.keep} kept · ${zones.blend} blended · ${zones.interpolate} interpolated`,
      );
    }
  }

  if (s.attentionFactor === undefined) {
    if (s.factor !== undefined && s.factor > 1) {
      info(
        findings,
        `attention_factor unset — runtimes default to 0.1·ln(${s.factor}) + 1 = ${fmt2(yarnMscale(s.factor))}`,
      );
    }
  } else if (s.attentionFactor <= 0) {
    error(findings, `attention_factor ${s.attentionFactor} must be > 0`);
  } else if (s.attentionFactor > 10) {
    warn(findings, `attention_factor ${s.attentionFactor} is far outside the fitted range (~1.0–1.5)`);
  } else {
    ok(findings, `attention_factor ${s.attentionFactor} explicitly set`);
  }
}

function checkLlama3(geom: RopeGeometry, findings: Finding[]): void {
  const s = geom.scaling!;
  const lo = s.lowFreqFactor;
  const hi = s.highFreqFactor;
  if (lo === undefined || hi === undefined) {
    for (const [key, v] of [
      ["low_freq_factor", lo],
      ["high_freq_factor", hi],
    ] as const) {
      if (v === undefined && !s.invalidKeys.includes(key)) {
        error(findings, `llama3 scaling requires a numeric "${key}" and none is present`);
      }
    }
    return;
  }
  if (lo <= 0 || hi <= 0) {
    error(findings, `low/high_freq_factor must be > 0 (got ${lo}/${hi})`);
    return;
  }
  if (lo >= hi) {
    error(
      findings,
      `low_freq_factor (${lo}) must be less than high_freq_factor (${hi}) — the wavelength bands are inverted`,
    );
    return;
  }
  ok(findings, `low_freq_factor ${lo} < high_freq_factor ${hi}`);
  if (geom.declaredMax !== null && geom.declaredMax > geom.trainedCtx) {
    info(
      findings,
      `declared max ${fmtCtx(geom.declaredMax)} comes from long-context fine-tuning — the llama3 factor is a frequency divisor, not a reach multiplier`,
    );
  }
  const zones = llama3Zones(geom.base, geom.rotaryDim, lo, hi, geom.trainedCtx);
  ok(
    findings,
    `bands over ${geom.pairs} pairs: ${zones.keep} kept · ${zones.blend} blended · ${zones.interpolate} interpolated ÷${s.factor ?? "?"}`,
  );
}
