# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-07-13

### Added

- The `plan` command: given a model's RoPE geometry and a target context,
  the exact parameters for all three extension recipes — linear/PI factor,
  static-NTK scaled base `b·s^(D/(D−2))`, and YaRN's correction range,
  zone counts and attention temperature — plus one recommendation with
  its reasoning stated (`--finetune` switches the policy).
- Key-driven config.json normalization: `rope_theta`, explicit or derived
  `head_dim`, `partial_rotary_factor` (partial-rotary models), nested
  `text_config` for multimodal configs, and full `rope_scaling` block
  parsing with legacy `type` / modern `rope_type` reconciliation.
- The `check` command: a rule engine auditing existing rope_scaling
  blocks — unknown types, missing or non-finite factors, inverted YaRN
  betas, out-of-range attention_factor, missing
  original_max_position_embeddings, factor-vs-declared-max mismatches,
  fork-specific and unknown keys, dynamic-NTK KV-cache drift — with
  ok/info/warn/error findings, a VALID/INVALID verdict, `--strict`, and
  a `--target` reachability gate.
- llama3-type block validation with the honest semantics: the factor is
  a per-band frequency divisor, not a reach multiplier, so the declared
  max (from long-context fine-tuning) is taken as the reach.
- The `dims` command: per-dimension-pair wavelengths, trained-context
  rotation counts, YaRN zone assignment (keep/blend/interp) and the
  frequency divisor each method applies, with readable elision around
  the ramp (`--all` for every pair).
- The `methods` command: a reference of all five understood schemes with
  formulas and provenance (kaiokendev/Chen et al., bloc97, emozilla,
  Peng et al. arXiv:2309.00071, Meta's Llama-3.1 bands).
- `--emit hf|llamacpp|vllm`: paste-ready output per runtime — a pure-JSON
  config patch, llama.cpp flags, or vLLM engine args — with runtime
  quirks (NTK as a rope_theta override, llama.cpp's self-derived YaRN
  temperature) explained on stderr, never stdout.
- Flag-only geometry (`--dim/--base/--ctx`) for planning without a
  config file, and `--beta-fast/--beta-slow` overrides.
- Script-friendly contract: `--json` on every command, byte-identical
  output for identical inputs, exit codes 0 (ok/valid) / 1 (check failed
  or target unreachable) / 2 (usage or config error).
- Five committed example configs (classic 4k 7B, high-base 8B, sound
  YaRN ×4, llama3 ×8, and a deliberately broken block) that double as
  regression fixtures.
- Test suite: 89 node:test tests (unit + CLI integration, every expected
  number hand-derived in comments) and an end-to-end `scripts/smoke.sh`
  against the bundled examples.

[0.1.0]: https://github.com/JaydenCJ/ropecalc/releases/tag/v0.1.0
