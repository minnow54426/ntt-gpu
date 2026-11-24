// Exact, small teaching models. These do not execute CUDA or estimate timings.
export const mod = (x, q) => ((x % q) + q) % q;
export function power(base, exponent, q) {
  let value = 1;
  for (let i = 0; i < exponent; i++) value = value * base % q;
  return value;
}
export function reverseBits(value, width) {
  let result = 0;
  for (let bit = 0; bit < width; bit++) { result = result * 2 + value % 2; value = Math.floor(value / 2); }
  return result;
}
export function butterflyTrace(input = [1, 2, 3, 4, 5, 6, 7, 8]) {
  if (input.length !== 8 || input.some(x => !Number.isSafeInteger(x))) throw Error('Use eight integer coefficients.');
  const q = 17, psi = 3, n = 8, states = [input.map(x => mod(x, q))], stages = [];
  for (let stage = 0; stage < 3; stage++) {
    const d = n / 2 ** (stage + 1), before = states.at(-1), after = [...before], pairs = [];
    for (let g = 0; g < n / (2 * d); g++) {
      const exponent = reverseBits(n / (2 * d) + g, 3), zeta = power(psi, exponent, q);
      for (let j = 0; j < d; j++) {
        const u = g * 2 * d + j, v = u + d, a = before[u], b = before[v], t = b * zeta % q;
        const x = mod(a + t, q), y = mod(a - t, q);
        after[u] = x; after[v] = y;
        pairs.push({u, v, a, b, t, x, y, zeta, exponent, d});
      }
    }
    states.push(after); stages.push(pairs);
  }
  return {q, psi, states, stages};
}
export function memoryPair(stage = 0, block = 0, thread = 0) {
  const clamp = (x, max) => Math.max(0, Math.min(max, Math.trunc(Number(x)) || 0));
  stage = clamp(stage, 15); block = clamp(block, 255); thread = clamp(thread, 255);
  const distance = 32768 / 2 ** stage, column = stage < 8;
  const localDistance = column ? distance / 256 : distance;
  const group = Math.floor(thread / localDistance), offset = thread % localDistance;
  const localU = group * 2 * localDistance + offset, localV = localU + localDistance;
  const address = i => column ? i * 256 + block : block * 256 + i;
  return {stage, block, thread, distance, column, active: thread < 128,
    load: address(thread), u: thread < 128 ? address(localU) : null,
    v: thread < 128 ? address(localV) : null};
}
export const coefficients = [1, -2, 20, 100];
export const primes = [17, 41, 73];
export function towerModel(count = 3, format = 'coefficient') {
  count = Math.max(1, Math.min(3, Math.trunc(Number(count)) || 1));
  const roots = [2, 3, 10], towers = primes.slice(0, count).map((q, k) => {
    const values = coefficients.map(x => mod(x, q));
    return format === 'evaluation' ? values.map((_, slot) => {
      const point = power(roots[k], 2 * reverseBits(slot, 2) + 1, q);
      return values.reduce((sum, a, i) => (sum + a * power(point, i, q)) % q, 0);
    }) : values;
  });
  return {count, format, towers, bytes: count * 65536 * 8, sampleBytes: count * 4 * 8};
}
export function transferModel(mode = 'resident', step = 0, towers = 3) {
  step = Math.max(0, Math.min(5, Math.trunc(Number(step)) || 0));
  towers = Math.max(1, Math.min(3, Math.trunc(Number(towers)) || 1));
  const bytes = towers * 65536 * 8;
  const units = mode === 'separate' ? [0, 2, 4, 7, 9, 9] : [0, 2, 2, 2, 2, 3];
  return {mode, step, bytes, copiedBytes: units[step] * bytes,
    totalBytes: (mode === 'separate' ? 9 : 3) * bytes};
}
