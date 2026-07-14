#!/usr/bin/env bash
# Smoke test for ropecalc: exercises the real CLI end to end against the
# committed example configs. No network, idempotent, runs from a clean
# checkout (after `npm install`). Prints "SMOKE OK" on success.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

fail() {
  echo "SMOKE FAIL: $1" >&2
  exit 1
}

BASE7B=examples/base-10k-7b.json
YARN=examples/yarn-128k.json
LLAMA3=examples/llama3-extended.json
BROKEN=examples/broken-rope.json

# 1. Build (idempotent).
npm run build >/dev/null 2>&1 || fail "npm run build failed"
CLI="node $ROOT/dist/cli.js"
echo "[smoke] build ok"

# 2. --version matches package.json; --help documents every command.
PKG_VERSION="$(node -p "require('$ROOT/package.json').version")"
CLI_VERSION="$($CLI --version)"
[ "$CLI_VERSION" = "$PKG_VERSION" ] || fail "--version mismatch: $CLI_VERSION != $PKG_VERSION"
HELP="$($CLI --help)"
for word in plan check dims methods --target --emit "Exit codes"; do
  echo "$HELP" | grep -q -- "$word" || fail "--help missing $word"
done
echo "[smoke] --help/--version ok ($CLI_VERSION)"

# 3. Error handling: bad flags, commands and inputs exit 2.
set +e
$CLI plan "$BASE7B" --target 16k --frobnicate >/dev/null 2>&1; [ $? -eq 2 ] || { set -e; fail "unknown flag should exit 2"; }
$CLI plan "$BASE7B" >/dev/null 2>&1; [ $? -eq 2 ] || { set -e; fail "missing --target should exit 2"; }
$CLI plan does-not-exist.json --target 16k >/dev/null 2>&1; [ $? -eq 2 ] || { set -e; fail "missing file should exit 2"; }
printf '{not json' > "$WORKDIR/bad.json"
$CLI check "$WORKDIR/bad.json" >/dev/null 2>&1; [ $? -eq 2 ] || { set -e; fail "invalid JSON should exit 2"; }
$CLI plan "$BASE7B" --target 16q >/dev/null 2>&1; [ $? -eq 2 ] || { set -e; fail "bad target should exit 2"; }
set -e
echo "[smoke] error handling ok (exit 2)"

# 4. The flagship question: extend the classic 4k/10000-base 7B to 16k.
PLAN="$($CLI plan "$BASE7B" --target 16k)" || fail "flagship plan should exit 0"
for want in "factor 4.00×" "rope_theta 10000 → 40889.9" "ramp pairs 20…46 of 64" "mscale 1.14" "recommend yarn"; do
  echo "$PLAN" | grep -qF "$want" || fail "plan output missing: $want"
done
echo "[smoke] plan ok (4k → 16k, all three methods + recommendation)"

# 5. Emits are paste-ready: llama.cpp flags exact, HF patch is pure JSON.
EMIT="$($CLI plan "$BASE7B" --target 16k --emit llamacpp 2>/dev/null)"
[ "$EMIT" = "--ctx-size 16384 --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 4096" ] \
  || fail "llamacpp emit wrong: $EMIT"
$CLI plan "$BASE7B" --target 16k --emit hf --method ntk 2>/dev/null | node -e "
  const p = JSON.parse(require('fs').readFileSync(0, 'utf8'));
  if (p.rope_theta !== 40889.9) throw new Error('rope_theta: ' + p.rope_theta);
  if (p.max_position_embeddings !== 16384) throw new Error('max_position_embeddings');
" || fail "hf ntk emit is not clean JSON with the right numbers"
echo "[smoke] emit ok (llamacpp + hf)"

# 6. check: the sound YaRN config validates, the broken one fails with exit 1.
GOOD="$($CLI check "$YARN")" || fail "yarn example should exit 0"
echo "$GOOD" | grep -qF "ramp spans pairs 23…40 of 64" || fail "check missing the ramp finding"
echo "$GOOD" | grep -qF "VALID" || fail "check missing VALID verdict"
$CLI check "$LLAMA3" | grep -qF "29 kept · 6 blended · 29 interpolated" || fail "llama3 band counts wrong"
set +e
$CLI check "$BROKEN" > "$WORKDIR/broken.txt"; RC=$?
set -e
[ "$RC" -eq 1 ] || fail "broken config should exit 1, got $RC"
grep -qF "ramp is inverted" "$WORKDIR/broken.txt" || fail "inverted-beta finding missing"
grep -qF "INVALID — 2 errors" "$WORKDIR/broken.txt" || fail "INVALID verdict missing"
echo "[smoke] check ok (VALID exit 0, INVALID exit 1, llama3 bands)"

# 7. check --target gates on reach: 128k fits the declared 4×, 256k does not.
$CLI check "$YARN" --target 128k >/dev/null || fail "128k should be within reach"
set +e
$CLI check "$YARN" --target 256k >/dev/null; RC=$?
set -e
[ "$RC" -eq 1 ] || fail "256k beyond reach should exit 1, got $RC"
echo "[smoke] target gate ok (128k yes, 256k no)"

# 8. dims: zones visible when elided, all 64 pairs with --all.
DIMS="$($CLI dims "$BASE7B" --target 16k)"
for zone in keep blend interp; do
  echo "$DIMS" | grep -q "$zone" || fail "dims missing zone $zone"
done
ALLROWS="$($CLI dims "$BASE7B" --target 16k --all | grep -c '^ *[0-9]\+ ')"
[ "$ALLROWS" -eq 64 ] || fail "dims --all should print 64 pairs, got $ALLROWS"
echo "[smoke] dims ok (zones + 64 pairs)"

# 9. --json is valid, structurally intact and byte-identical across runs.
A="$($CLI plan "$BASE7B" --target 16k --json)"
B="$($CLI plan "$BASE7B" --target 16k --json)"
[ "$A" = "$B" ] || fail "plan --json is not deterministic"
echo "$A" | node -e "
  const p = JSON.parse(require('fs').readFileSync(0, 'utf8'));
  if (p.tool !== 'ropecalc') throw new Error('tool');
  if (p.factor !== 4) throw new Error('factor: ' + p.factor);
  if (p.yarn.low !== 20 || p.yarn.high !== 46) throw new Error('yarn ramp');
  if (p.yarn.mscale !== 1.1386) throw new Error('mscale: ' + p.yarn.mscale);
" || fail "plan --json is not structurally intact"
echo "[smoke] --json + determinism ok"

# 10. methods reference carries its receipts.
$CLI methods | grep -qF "arXiv:2309.00071" || fail "methods missing the YaRN citation"
echo "[smoke] methods ok"

echo "SMOKE OK"
