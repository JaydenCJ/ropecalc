/**
 * Context-length parsing and deterministic number formatting.
 *
 * Context lengths accept the notation people actually use for them:
 * `4096`, `16k` (= 16 × 1024 = 16384), `1m` (= 1024 × 1024). All formatting
 * goes through `toFixed`, which is locale-independent — identical inputs
 * must render byte-identical output on every machine.
 */

export class UnitError extends Error {}

/** Parse a context length: plain integer, `<n>k` (×1024) or `<n>m` (×1024²). */
export function parseCtx(text: string): number {
  const t = text.trim().toLowerCase();
  const m = /^(\d+(?:\.\d+)?)([km]?)$/.exec(t);
  if (!m) throw new UnitError(`cannot parse context length "${text}" (try 4096, 16k, 1m)`);
  const value = Number(m[1]);
  const mult = m[2] === "k" ? 1024 : m[2] === "m" ? 1024 * 1024 : 1;
  const n = value * mult;
  if (!Number.isInteger(n) || n < 1) {
    throw new UnitError(`context length "${text}" is not a positive whole number of tokens`);
  }
  return n;
}

/** Render a context length: `131072 (128k)` when it is a clean multiple of 1024, else the plain number. */
export function fmtCtx(n: number): string {
  if (n >= 1024 && n % 1024 === 0) return `${n} (${n / 1024}k)`;
  return String(n);
}

/** Render just the compact form: `128k` when clean, else the plain number. */
export function fmtCtxShort(n: number): string {
  if (n >= 1024 && n % 1024 === 0) return `${n / 1024}k`;
  return String(n);
}

/** Fixed two-decimal rendering (factors, divisors, mscale). */
export function fmt2(x: number): string {
  return x.toFixed(2);
}

/**
 * Adaptive magnitude rendering for wavelengths and rotation counts, which
 * span ~6.28 to millions in one table: 2 decimals under 10, 1 decimal under
 * 1000, whole numbers above.
 */
export function fmtAdaptive(x: number): string {
  if (x < 10) return x.toFixed(2);
  if (x < 1000) return x.toFixed(1);
  return String(Math.round(x));
}

/** Round to at most `places` decimals, returning a number (for JSON emits: no 3.9999999996). */
export function roundTo(x: number, places: number): number {
  const p = Math.pow(10, places);
  return Math.round(x * p) / p;
}
