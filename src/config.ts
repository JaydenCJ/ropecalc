/**
 * config.json loading and normalization — key-driven, never name-driven.
 *
 * ropecalc reads the keys that determine RoPE geometry (`rope_theta`,
 * `head_dim` / `hidden_size` ÷ `num_attention_heads`, `partial_rotary_factor`,
 * `max_position_embeddings`, `rope_scaling`, with `text_config` nesting for
 * multimodal configs) and ignores everything else, so a new model that reuses
 * the keys works the day it drops. Structural impossibilities (odd rotary
 * dimension, no derivable head_dim) are hard `ConfigError`s; questionable
 * *values* inside `rope_scaling` are left for `check` to report as findings —
 * a validator that crashes on the config it is meant to diagnose is useless.
 */

import { readFileSync } from "node:fs";
import type { RopeGeometry, RopeScalingBlock } from "./types.js";

export class ConfigError extends Error {}

type Raw = Record<string, unknown>;

/** RoPE base assumed when a config omits `rope_theta` (the original RoPE default). */
export const DEFAULT_BASE = 10000;

function isObject(v: unknown): v is Raw {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function finiteNumber(v: unknown): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

/** Scaling-block keys ropecalc fully understands, mapped to camelCase fields. */
const SCALING_KEYS = [
  "factor",
  "original_max_position_embeddings",
  "beta_fast",
  "beta_slow",
  "attention_factor",
  "low_freq_factor",
  "high_freq_factor",
] as const;

/** Runtime-fork keys we recognize (and flag) but do not compute with. */
const FORK_KEYS = ["mscale", "mscale_all_dim", "truncate"] as const;

/** Parse a raw `rope_scaling` object. Pure; never throws on bad values. */
export function parseScalingBlock(raw: Raw): RopeScalingBlock {
  const ropeType = typeof raw["rope_type"] === "string" ? (raw["rope_type"] as string) : undefined;
  const legacyType = typeof raw["type"] === "string" ? (raw["type"] as string) : undefined;
  const block: RopeScalingBlock = {
    type: ropeType ?? legacyType ?? "",
    usedLegacyKey: ropeType === undefined && legacyType !== undefined,
    conflictingTypeKeys:
      ropeType !== undefined && legacyType !== undefined && ropeType !== legacyType,
    invalidKeys: [],
    forkKeys: [],
    unknownKeys: [],
  };

  const fields: Record<(typeof SCALING_KEYS)[number], keyof RopeScalingBlock> = {
    factor: "factor",
    original_max_position_embeddings: "originalMaxPositionEmbeddings",
    beta_fast: "betaFast",
    beta_slow: "betaSlow",
    attention_factor: "attentionFactor",
    low_freq_factor: "lowFreqFactor",
    high_freq_factor: "highFreqFactor",
  };

  for (const key of Object.keys(raw)) {
    if (key === "type" || key === "rope_type") continue;
    if ((SCALING_KEYS as readonly string[]).includes(key)) {
      const n = finiteNumber(raw[key]);
      if (n === undefined) block.invalidKeys.push(key);
      else (block[fields[key as (typeof SCALING_KEYS)[number]]] as number) = n;
    } else if ((FORK_KEYS as readonly string[]).includes(key)) {
      block.forkKeys.push(key);
    } else {
      block.unknownKeys.push(key);
    }
  }
  block.forkKeys.sort();
  block.unknownKeys.sort();
  block.invalidKeys.sort();
  return block;
}

/** Normalize an already-parsed config object. Pure and unit-testable. */
export function normalizeConfig(raw: unknown, source: string): RopeGeometry {
  if (!isObject(raw)) throw new ConfigError(`${source}: config root is not a JSON object`);
  const warnings: string[] = [];

  // Multimodal configs nest the text model under text_config; inner keys win.
  const merged: Raw = isObject(raw["text_config"]) ? { ...raw, ...raw["text_config"] } : raw;

  let headDim = finiteNumber(merged["head_dim"]);
  if (headDim === undefined) {
    const hidden = finiteNumber(merged["hidden_size"]);
    const heads = finiteNumber(merged["num_attention_heads"]);
    if (hidden === undefined || heads === undefined || heads < 1) {
      throw new ConfigError(
        `${source}: cannot determine head_dim (need head_dim, or hidden_size + num_attention_heads)`,
      );
    }
    headDim = hidden / heads;
    if (!Number.isInteger(headDim)) {
      throw new ConfigError(
        `${source}: hidden_size ${hidden} does not divide by num_attention_heads ${heads}`,
      );
    }
    warnings.push(`head_dim derived as hidden_size / num_attention_heads = ${headDim}`);
  }
  if (!Number.isInteger(headDim) || headDim < 2) {
    throw new ConfigError(`${source}: head_dim ${headDim} is not an integer ≥ 2`);
  }

  const prf = finiteNumber(merged["partial_rotary_factor"]) ?? 1;
  if (prf <= 0 || prf > 1) {
    throw new ConfigError(`${source}: partial_rotary_factor ${prf} is outside (0, 1]`);
  }
  const rotaryDim = Math.floor(headDim * prf);
  if (rotaryDim < 2 || rotaryDim % 2 !== 0) {
    throw new ConfigError(
      `${source}: rotary dimension ${rotaryDim} (head_dim ${headDim} × partial_rotary_factor ${prf}) must be even and ≥ 2`,
    );
  }

  let base = finiteNumber(merged["rope_theta"]);
  if (base === undefined) {
    base = DEFAULT_BASE;
    warnings.push(`rope_theta missing — assuming the RoPE default ${DEFAULT_BASE}`);
  }
  if (base <= 1) throw new ConfigError(`${source}: rope_theta ${base} must be > 1`);

  const declaredMaxRaw = finiteNumber(merged["max_position_embeddings"]);
  const declaredMax =
    declaredMaxRaw !== undefined && Number.isInteger(declaredMaxRaw) && declaredMaxRaw >= 1
      ? declaredMaxRaw
      : null;
  if (declaredMaxRaw !== undefined && declaredMax === null) {
    throw new ConfigError(
      `${source}: max_position_embeddings ${declaredMaxRaw} is not a positive integer`,
    );
  }

  const scaling = isObject(merged["rope_scaling"]) ? parseScalingBlock(merged["rope_scaling"]) : null;
  if (merged["rope_scaling"] !== undefined && merged["rope_scaling"] !== null && scaling === null) {
    throw new ConfigError(`${source}: rope_scaling is not a JSON object`);
  }

  // The context the frequencies were trained at: the scaling block's own
  // original_max_position_embeddings when present, else the declared max.
  const orig = scaling?.originalMaxPositionEmbeddings;
  const trainedCtx = orig !== undefined && orig >= 1 ? Math.floor(orig) : declaredMax;
  if (trainedCtx === null) {
    throw new ConfigError(
      `${source}: cannot determine trained context (need max_position_embeddings or rope_scaling.original_max_position_embeddings)`,
    );
  }

  return {
    source,
    headDim,
    rotaryDim,
    pairs: rotaryDim / 2,
    base,
    trainedCtx,
    declaredMax,
    scaling,
    warnings,
  };
}

/** Load and normalize a config.json from disk. */
export function loadConfig(path: string): RopeGeometry {
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch {
    throw new ConfigError(`cannot read ${path}`);
  }
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    throw new ConfigError(`${path}: invalid JSON (${(e as Error).message})`);
  }
  return normalizeConfig(raw, path);
}

/** Build geometry from `--dim/--base/--ctx` flags, for models without a config at hand. */
export function geometryFromFlags(opts: { dim: number; base?: number; ctx: number }): RopeGeometry {
  const warnings: string[] = [];
  if (!Number.isInteger(opts.dim) || opts.dim < 2 || opts.dim % 2 !== 0) {
    throw new ConfigError(`--dim ${opts.dim} must be an even integer ≥ 2`);
  }
  let base = opts.base;
  if (base === undefined) {
    base = DEFAULT_BASE;
    warnings.push(`--base not given — assuming the RoPE default ${DEFAULT_BASE}`);
  }
  if (base <= 1) throw new ConfigError(`--base ${base} must be > 1`);
  return {
    source: "flags",
    headDim: opts.dim,
    rotaryDim: opts.dim,
    pairs: opts.dim / 2,
    base,
    trainedCtx: opts.ctx,
    declaredMax: null,
    scaling: null,
    warnings,
  };
}
