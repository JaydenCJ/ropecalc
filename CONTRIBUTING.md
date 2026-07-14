# Contributing to ropecalc

Issues, discussions and pull requests are all welcome — this project aims
to stay small, zero-dependency at runtime, fully offline, and exact about
math that people currently copy from forum posts.

## Getting started

Requirements: Node.js >= 22.13 (for the stable `node:test` runner used by the suite).

```bash
git clone https://github.com/JaydenCJ/ropecalc.git
cd ropecalc
npm install            # installs typescript, the only devDependency
npm run build          # compile TypeScript to dist/
npm test               # build + 89 node:test tests
bash scripts/smoke.sh  # end-to-end CLI check against examples/
```

`scripts/smoke.sh` exercises the real CLI (the flagship 4k→16k plan and its
numbers, both emit formats, VALID/INVALID check verdicts and their exit
codes, the target-reach gate, dims zones, JSON determinism) against the
bundled example configs and must print `SMOKE OK`.

## Before you open a pull request

1. `npx tsc -p tsconfig.json --noEmit` — the tree must type-check clean (strict mode is enforced).
2. `npm test` — all tests must pass.
3. `bash scripts/smoke.sh` — must print `SMOKE OK`.
4. Add tests for behavior changes; keep logic in pure, unit-testable
   modules (every compute function takes a geometry and returns plain
   data — only `cli.ts` touches the filesystem or the process).
5. New expected values in tests must come with the paper arithmetic in a
   comment — no "expected = whatever the code printed" snapshots. A wrong
   constant here becomes someone's broken 128k deployment.

## Ground rules

- **No runtime dependencies.** The zero-dependency install is a core
  feature; adding one needs justification in the PR and will usually be
  declined.
- No network calls, ever — ropecalc reads one local JSON file and prints
  numbers. It must work on an air-gapped box.
- Determinism is API: same config and flags, byte-identical output and
  exit code — no clocks, no randomness, no locale-dependent formatting.
- Stay key-driven: support new scaling schemes by reading their config
  keys, never by matching model names against a hardcoded list.
- Formulas must cite their source (paper, post, or reference
  implementation) in the module header and in `docs/rope-math.md`; where
  runtimes disagree, follow HF transformers' semantics and document the
  divergence.
- Code comments and doc comments are written in English.

## Reporting bugs

Please include: `ropecalc --version` output, the exact command line, the
config.json (or the relevant keys), and the number you expected with how
you derived it. Discrepancy reports against a real runtime's computed
inv_freq tensors are especially valuable — say which runtime and version
you compared against.

## Security

Do not open public issues for security problems (e.g. a crafted
config.json that hangs the parser); use GitHub private vulnerability
reporting on this repository instead.
