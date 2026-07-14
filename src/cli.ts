#!/usr/bin/env node
/**
 * ropecalc CLI — the only module that touches the filesystem or the process.
 *
 * Exit codes (script-friendly, documented in --help):
 *   0  success; check: the config is valid (warnings allowed unless --strict)
 *   1  check found errors (or warnings under --strict), or a --target is
 *      out of reach of the declared scaling
 *   2  usage or config error
 */

import { ConfigError, geometryFromFlags, loadConfig } from "./config.js";
import { parseArgs, positiveFloat, positiveInt, UsageError, type FlagSpec } from "./args.js";
import { computePlan } from "./plan.js";
import { checkConfig } from "./validate.js";
import { computeDims } from "./dims.js";
import { emit, EmitError, emitNotes, RUNTIMES, type Runtime } from "./emit.js";
import { methodById, METHODS } from "./methods.js";
import { renderCheck, renderDims, renderMethods, renderPlan } from "./render.js";
import { DEFAULT_BETA_FAST, DEFAULT_BETA_SLOW } from "./yarn.js";
import { parseCtx, UnitError } from "./units.js";
import { VERSION } from "./version.js";
import type { MethodId, RopeGeometry } from "./types.js";

const HELP = `ropecalc ${VERSION} — RoPE scaling parameters for context extension

Usage:
  ropecalc plan  <config.json> --target <ctx> [options]
                                          exact linear/NTK/YaRN parameters + a recommendation
  ropecalc check <config.json> [options]  validate an existing rope_scaling block
  ropecalc dims  <config.json> [options]  per-dimension wavelengths, zones and divisors
  ropecalc methods                        method reference (formulas + provenance)

Geometry (instead of a config file, for plan and dims):
  --dim <n>             rotary/head dimension, e.g. 128
  --ctx <ctx>           trained context, e.g. 4096
  --base <theta>        RoPE base (default 10000)

Options:
  --target <ctx>        target context, e.g. 32768 or 32k (dims: defaults to
                        the config's own declared scaling reach)
  --method <m>          linear | ntk | yarn (plan: narrow output / pick emit)
  --emit <runtime>      hf | llamacpp | vllm — print ready-to-paste settings
                        for the chosen (or recommended) method, JSON/flags only
  --finetune            plan as if you will fine-tune at the target length
  --beta-fast <n>       YaRN ramp upper rotation count (default 32)
  --beta-slow <n>       YaRN ramp lower rotation count (default 1)
  --strict              check: treat warnings as failures
  --all                 dims: print every pair (no elision)
  --json                machine-readable output
  --help, --version

Exit codes: 0 ok / valid · 1 check failed or target unreachable · 2 usage or config error`;

const COMMON_FLAGS: FlagSpec[] = [
  { name: "target", takesValue: true },
  { name: "dim", takesValue: true },
  { name: "ctx", takesValue: true },
  { name: "base", takesValue: true },
  { name: "method", takesValue: true },
  { name: "emit", takesValue: true },
  { name: "beta-fast", takesValue: true },
  { name: "beta-slow", takesValue: true },
  { name: "finetune", takesValue: false },
  { name: "strict", takesValue: false },
  { name: "all", takesValue: false },
  { name: "json", takesValue: false },
  { name: "help", takesValue: false },
  { name: "version", takesValue: false },
];

interface Io {
  out: (s: string) => void;
  err: (s: string) => void;
}

interface Parsed {
  values: Map<string, string>;
  flags: Set<string>;
  positionals: string[];
}

/** Resolve geometry from the positional config path or the --dim/--ctx/--base flags. */
function loadGeometry(parsed: Parsed, io: Io): RopeGeometry {
  const { values, positionals } = parsed;
  const path = positionals[1];
  const hasFlags = values.has("dim") || values.has("ctx") || values.has("base");
  if (path !== undefined && hasFlags) {
    throw new UsageError("give either a config.json or --dim/--ctx/--base flags, not both");
  }
  if (positionals.length > 2) throw new UsageError(`unexpected operand "${positionals[2]}"`);

  let geom: RopeGeometry;
  if (path !== undefined) {
    geom = loadConfig(path);
  } else {
    if (!values.has("dim") || !values.has("ctx")) {
      throw new UsageError("missing <config.json> operand (or both --dim and --ctx)");
    }
    geom = geometryFromFlags({
      dim: positiveInt("dim", values.get("dim")!),
      ctx: parseCtx(values.get("ctx")!),
      base: values.has("base") ? positiveFloat("base", values.get("base")!) : undefined,
    });
  }
  for (const w of geom.warnings) io.err(`note: ${w}`);
  return geom;
}

function parseBetas(values: Map<string, string>): { betaFast: number; betaSlow: number } {
  const betaFast = values.has("beta-fast")
    ? positiveFloat("beta-fast", values.get("beta-fast")!)
    : DEFAULT_BETA_FAST;
  const betaSlow = values.has("beta-slow")
    ? positiveFloat("beta-slow", values.get("beta-slow")!)
    : DEFAULT_BETA_SLOW;
  if (betaFast <= betaSlow) {
    throw new UsageError(`--beta-fast (${betaFast}) must be greater than --beta-slow (${betaSlow})`);
  }
  return { betaFast, betaSlow };
}

function parseMethod(values: Map<string, string>): MethodId | null {
  if (!values.has("method")) return null;
  const id = values.get("method")!;
  const m = methodById(id);
  if (!m || !m.plannable) {
    const plannable = METHODS.filter((x) => x.plannable).map((x) => x.id);
    throw new UsageError(`--method must be one of ${plannable.join(", ")}, got "${id}"`);
  }
  return id as MethodId;
}

function parseRuntime(values: Map<string, string>): Runtime | null {
  if (!values.has("emit")) return null;
  const r = values.get("emit")!;
  if (!(RUNTIMES as string[]).includes(r)) {
    throw new UsageError(`--emit must be one of ${RUNTIMES.join(", ")}, got "${r}"`);
  }
  return r as Runtime;
}

/** Run the CLI. Pure with respect to `io`; returns the process exit code. */
export function run(argv: string[], io: Io): number {
  const parsed = parseArgs(argv, COMMON_FLAGS);
  const { values, flags, positionals } = parsed;

  if (flags.has("version")) {
    io.out(VERSION);
    return 0;
  }
  if (flags.has("help") || positionals.length === 0) {
    io.out(HELP);
    return flags.has("help") ? 0 : 2;
  }

  const command = positionals[0]!;
  const json = flags.has("json");

  const reject = (flag: string, cmds: string) => {
    if (values.has(flag) || flags.has(flag)) {
      throw new UsageError(`--${flag} only applies to ${cmds}`);
    }
  };

  switch (command) {
    case "plan": {
      reject("strict", "check");
      reject("all", "dims");
      const geom = loadGeometry(parsed, io);
      if (!values.has("target")) throw new UsageError("plan requires --target <ctx>");
      const targetCtx = parseCtx(values.get("target")!);
      const { betaFast, betaSlow } = parseBetas(values);
      const method = parseMethod(values);
      const runtime = parseRuntime(values);
      const plan = computePlan(geom, {
        targetCtx,
        finetune: flags.has("finetune"),
        betaFast,
        betaSlow,
      });
      if (runtime !== null) {
        const chosen =
          method ??
          (plan.recommendation.method === "none" ? null : plan.recommendation.method);
        if (chosen === null) {
          throw new EmitError(
            `nothing to emit: factor ${plan.factor} needs no scaling (target within trained context)`,
          );
        }
        for (const n of emitNotes(plan, chosen, runtime)) io.err(`note: ${n}`);
        io.out(emit(plan, chosen, runtime));
        return 0;
      }
      io.out(json ? JSON.stringify(plan, null, 2) : renderPlan(plan, geom, method));
      return 0;
    }
    case "check": {
      reject("dim", "plan and dims (check needs the real config)");
      reject("ctx", "plan and dims (check needs the real config)");
      reject("base", "plan and dims (check needs the real config)");
      reject("method", "plan");
      reject("emit", "plan");
      reject("finetune", "plan");
      reject("all", "dims");
      reject("beta-fast", "plan and dims");
      reject("beta-slow", "plan and dims");
      const geom = loadGeometry(parsed, io);
      const targetCtx = values.has("target") ? parseCtx(values.get("target")!) : undefined;
      const result = checkConfig(geom, { targetCtx, strict: flags.has("strict") });
      io.out(json ? JSON.stringify(result, null, 2) : renderCheck(result, geom));
      return result.valid ? 0 : 1;
    }
    case "dims": {
      reject("strict", "check");
      reject("method", "plan");
      reject("emit", "plan");
      reject("finetune", "plan");
      const geom = loadGeometry(parsed, io);
      let targetCtx: number;
      if (values.has("target")) {
        targetCtx = parseCtx(values.get("target")!);
      } else if (geom.scaling !== null && geom.scaling.factor !== undefined && geom.scaling.factor >= 1) {
        // llama3's factor is a per-band frequency divisor, not a reach
        // multiplier — the reach is the declared max (same rule as `check`).
        targetCtx =
          geom.scaling.type === "llama3" && geom.declaredMax !== null
            ? geom.declaredMax
            : Math.round(geom.trainedCtx * geom.scaling.factor);
        io.err(`note: --target not given — using the config's declared reach ${targetCtx}`);
      } else {
        throw new UsageError("dims requires --target <ctx> (the config declares no scaling reach)");
      }
      // A config that already declares yarn betas is shown with them.
      const scalingBetas =
        geom.scaling !== null && geom.scaling.type === "yarn"
          ? { fast: geom.scaling.betaFast, slow: geom.scaling.betaSlow }
          : { fast: undefined, slow: undefined };
      if (!values.has("beta-fast") && scalingBetas.fast !== undefined) {
        values.set("beta-fast", String(scalingBetas.fast));
      }
      if (!values.has("beta-slow") && scalingBetas.slow !== undefined) {
        values.set("beta-slow", String(scalingBetas.slow));
      }
      const { betaFast, betaSlow } = parseBetas(values);
      const dims = computeDims(geom, { targetCtx, betaFast, betaSlow });
      io.out(json ? JSON.stringify(dims, null, 2) : renderDims(dims, flags.has("all")));
      return 0;
    }
    case "methods": {
      if (positionals.length > 1) throw new UsageError(`unexpected operand "${positionals[1]}"`);
      for (const name of [...values.keys(), ...flags]) {
        if (name !== "json") throw new UsageError(`--${name} does not apply to methods`);
      }
      io.out(json ? JSON.stringify({ tool: "ropecalc", methods: METHODS }, null, 2) : renderMethods());
      return 0;
    }
    default:
      throw new UsageError(`unknown command "${command}" — see ropecalc --help`);
  }
}

/** Entry point used by the bin shim. */
export function main(argv: string[]): number {
  const io: Io = {
    out: (s) => process.stdout.write(s + "\n"),
    err: (s) => process.stderr.write(s + "\n"),
  };
  try {
    return run(argv, io);
  } catch (err) {
    if (
      err instanceof UsageError ||
      err instanceof ConfigError ||
      err instanceof UnitError ||
      err instanceof EmitError
    ) {
      io.err(`ropecalc: ${err.message}`);
      return 2;
    }
    throw err;
  }
}

process.exitCode = main(process.argv.slice(2));
