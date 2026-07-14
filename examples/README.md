# Example configs

Five hand-written `config.json` files spanning the rope_scaling space
ropecalc understands. The shapes are realistic (they match well-known public
model classes dimension-for-dimension) but the files are examples, not copies
of any distribution — a config is just geometry.

| File | Geometry | rope_scaling | Why it's here |
|---|---|---|---|
| `base-10k-7b.json` | head_dim 128, base 10000, ctx 4096 | none | the classic 7B every scaling trick was invented on |
| `base-500k-8b.json` | head_dim 128, base 500000, ctx 8192 | none | a modern high-base 8B, the usual extension candidate |
| `yarn-128k.json` | head_dim 128, base 1000000, ctx 32768 | yarn ×4 → 128k | a sound YaRN block; `check` gives it a clean bill |
| `llama3-extended.json` | head_dim 128, base 500000, ctx 8192 | llama3 ×8 | wavelength bands; factor ≠ reach, and `check` knows why |
| `broken-rope.json` | head_dim 128, base 10000, ctx 4096 | yarn, sabotaged | inverted betas + negative attention_factor: exit 1 |

The test suite and `scripts/smoke.sh` pin these numbers, so the examples
double as regression fixtures.

## Things to try

```bash
# The classic: stretch a 4k model to 16k, see all three recipes priced out:
ropecalc plan examples/base-10k-7b.json --target 16k

# Paste-ready llama.cpp flags for the recommended method:
ropecalc plan examples/base-10k-7b.json --target 16k --emit llamacpp

# Audit a downloaded config before trusting its declared 128k:
ropecalc check examples/yarn-128k.json --target 128k

# What went wrong in a hand-edited block (exit code 1):
ropecalc check examples/broken-rope.json

# Which dimensions YaRN touches, pair by pair:
ropecalc dims examples/base-10k-7b.json --target 16k

# Everything is scriptable:
ropecalc plan examples/base-500k-8b.json --target 64k --json | node -pe \
  'JSON.parse(require("fs").readFileSync(0,"utf8")).ntk.scaledBase'
```

To analyze your own model, point ropecalc at the `config.json` you already
have next to the weights — or skip the file entirely with
`--dim 128 --base 10000 --ctx 4096`. ropecalc itself never touches the
network.
