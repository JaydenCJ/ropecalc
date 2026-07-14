/**
 * Ready-to-paste emitters. One method × one runtime → exactly the lines to
 * paste, nothing else on stdout (notes go to stderr via the CLI), so
 * `ropecalc plan … --emit hf | jq` and command substitution both work.
 *
 * Runtime quirks encoded here, so users don't have to know them:
 *  - HF transformers has no rope_type for static NTK — it is "just" a bigger
 *    rope_theta, emitted as such.
 *  - llama.cpp computes YaRN's 0.1·ln(s)+1 attention temperature itself
 *    (`--yarn-attn-factor` *multiplies* it), so the emit does not repeat it.
 *  - vLLM takes the whole rope_scaling block as inline JSON.
 */

import { roundTo } from "./units.js";
import type { MethodId, Plan } from "./types.js";

export type Runtime = "hf" | "llamacpp" | "vllm";

export const RUNTIMES: Runtime[] = ["hf", "llamacpp", "vllm"];

export class EmitError extends Error {}

/** The rope_scaling JSON block for methods that have one under HF semantics. */
function hfScalingBlock(plan: Plan, method: MethodId): Record<string, unknown> | null {
  if (method === "linear") {
    return { rope_type: "linear", factor: plan.factor };
  }
  if (method === "yarn") {
    const y = plan.yarn!;
    return {
      rope_type: "yarn",
      factor: y.factor,
      original_max_position_embeddings: plan.trainedCtx,
      beta_fast: y.betaFast,
      beta_slow: y.betaSlow,
    };
  }
  return null;
}

/** config.json patch (pure JSON on stdout). */
export function emitHf(plan: Plan, method: MethodId): string {
  const patch: Record<string, unknown> = { max_position_embeddings: plan.targetCtx };
  if (method === "ntk") {
    patch["rope_theta"] = plan.ntk!.scaledBase;
  } else {
    patch["rope_scaling"] = hfScalingBlock(plan, method);
  }
  return JSON.stringify(patch, null, 2);
}

/** llama.cpp server/CLI flags (one line). */
export function emitLlamaCpp(plan: Plan, method: MethodId): string {
  const ctx = `--ctx-size ${plan.targetCtx}`;
  if (method === "linear") {
    return `${ctx} --rope-scaling linear --rope-scale ${plan.factor}`;
  }
  if (method === "ntk") {
    return `${ctx} --rope-freq-base ${plan.ntk!.scaledBase}`;
  }
  return `${ctx} --rope-scaling yarn --rope-scale ${plan.factor} --yarn-orig-ctx ${plan.trainedCtx}`;
}

/** vLLM engine arguments (one line). */
export function emitVllm(plan: Plan, method: MethodId): string {
  const len = `--max-model-len ${plan.targetCtx}`;
  if (method === "ntk") {
    return `${len} --rope-theta ${plan.ntk!.scaledBase}`;
  }
  const block = hfScalingBlock(plan, method)!;
  return `${len} --rope-scaling '${JSON.stringify(block)}'`;
}

/** Dispatch. Throws EmitError when the plan has nothing to emit for the method. */
export function emit(plan: Plan, method: MethodId, runtime: Runtime): string {
  if (plan[method] === null) {
    throw new EmitError(
      `nothing to emit: the plan computed no ${method} parameters (factor ${plan.factor} needs no scaling)`,
    );
  }
  switch (runtime) {
    case "hf":
      return emitHf(plan, method);
    case "llamacpp":
      return emitLlamaCpp(plan, method);
    case "vllm":
      return emitVllm(plan, method);
  }
}

/** Human notes that accompany an emit — printed to stderr, never stdout. */
export function emitNotes(plan: Plan, method: MethodId, runtime: Runtime): string[] {
  const notes: string[] = [];
  if (method === "ntk") {
    if (runtime === "hf") {
      notes.push(
        `static NTK has no rope_type in transformers — it is a rope_theta override (${plan.base} → ${plan.ntk!.scaledBase})`,
      );
    }
    notes.push(`NTK reach shrinks near the target — consider planning with headroom`);
  }
  if (method === "yarn") {
    notes.push(
      `attention temperature ${roundTo(plan.yarn!.mscale, 4)} is applied by the runtime itself — do not add it twice`,
    );
    if (runtime === "llamacpp") {
      notes.push(`llama.cpp derives it from --rope-scale; --yarn-attn-factor multiplies it (leave at 1.0)`);
    }
  }
  if (method === "linear" && plan.factor > 2) {
    notes.push(`linear at ${plan.factor}× without fine-tuning degrades retrieval — see ropecalc plan`);
  }
  return notes;
}
