// Run the actual inline explorer script without a browser dependency.
// This verifies displayed indices, not CUDA execution or performance.
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const {runInNewContext} = require('node:vm');
const {test} = require('node:test');

class Element {
  children = [];
  dataset = {};
  attributes = {};
  listeners = {};
  textContent = '';
  value = '';
  appendChild(child) { this.children.push(child); }
  replaceChildren() { this.children = []; }
  setAttribute(name, value) { this.attributes[name] = value; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  click() { this.listeners.click(); }
  fill(value) { this.value = value; this.listeners.input(); }
}

function explorer() {
  const elements = new Map();
  const get = selector => {
    if (!elements.has(selector)) elements.set(selector, new Element());
    return elements.get(selector);
  };
  const buttons = () => [...get('[data-stages="far"]').children, ...get('[data-stages="near"]').children];
  const root = {
    querySelector:get,
    querySelectorAll:() => buttons(),
  };
  const source = readFileSync(join(__dirname,'../../docs/ntt-65536-work-distribution.html'),'utf8');
  const script = source.match(/<script>([\s\S]*?)<\/script>/)[1];
  runInNewContext(script, {document:{getElementById:() => root, createElement:() => new Element()}});
  return {get, buttons};
}

test('65536: all stages and boundary thread/block selections match the CPU pair schedule', () => {
  const page = explorer();
  for (const n of [65536]) {
    assert.equal(page.buttons().length, Math.log2(n));
    for (let stage=0; stage<Math.log2(n); stage++) {
      page.buttons()[stage].click();
      const distance = n / 2 ** (stage+1);
      const far = distance >= 256;
      const rows = n/256;
      const pairCount = far ? rows/2 : 128;
      for (const block of [0,3,far ? 255 : rows-1]) {
        page.get('[data-block]').fill(String(block));
        for (const thread of [0,5,21,pairCount-1,pairCount,200,255]) {
          page.get('[data-thread]').fill(String(thread));
          const selection = page.get('[data-selection]').textContent;
          if (thread >= pairCount) {
            assert.match(selection,/no butterfly/);
            assert.match(page.get('[data-twiddle]').textContent,/No twiddle read/);
          } else {
            // Independent form of the kernel mapping: insert the distance bit.
            const localDistance = far ? distance/256 : distance;
            const u = (thread & (localDistance-1)) + Math.floor(thread/localDistance)*2*localDistance;
            const upper = far ? u*256+block : block*256+u;
            assert.ok(selection.includes('global ' + upper + ' ↔ ' + (upper+distance)),selection);
            const cpuOffset = upper % (2*distance);
            const exponent = (n/distance)*cpuOffset;
            assert.equal(page.get('[data-twiddle]').textContent,
              'CPU offset J = ' + cpuOffset + ' · c = ψ^' + distance + ' · w = ψ^' + exponent + ' (mod q)');
            const cells = page.get('[data-matrix] tbody').children.flatMap(row => row.children);
            assert.equal(cells.find(cell => cell.className === 'matrix-upper').textContent,String(upper));
            assert.equal(cells.find(cell => cell.className === 'matrix-lower').textContent,String(upper+distance));
          }
        }
      }
    }
  }
});

test('65536 loads inactive butterfly threads and clamps input boundaries', () => {
  const page = explorer();
  assert.match(page.get('[data-transform-summary]').textContent,/N = 65536.*16 stages.*524,288 butterflies/);
  assert.match(page.get('[data-selection]').textContent,/global 1283 ↔ 34051/);
  page.get('[data-block]').fill('255');
  page.get('[data-thread]').fill('200');
  assert.match(page.get('[data-selection]').textContent,/no butterfly/);
  assert.match(page.get('[data-thread-work]').textContent,/load\/store: global position 51455/);
  page.buttons().find(button => button.textContent === '128').click();
  assert.equal(page.get('[data-block]').value,'255');
  assert.equal(page.get('[data-block]').max,'255');
  assert.match(page.get('[data-thread-work]').textContent,/load\/store: global position 65480/);
  page.get('[data-block]').fill('999');
  assert.equal(page.get('[data-block]').value,'255');
  page.get('[data-thread]').fill('-1');
  assert.equal(page.get('[data-thread]').value,'0');
  page.get('[data-thread]').fill('999');
  assert.equal(page.get('[data-thread]').value,'255');
});

test('kernel-shaped mapping covers each global index once at every stage', () => {
  for (const n of [65536]) {
    let butterflies = 0;
    for (let d=n/2; d>=1; d/=2) {
      const seen = new Uint8Array(n);
      const far = d>=256;
      const length = far ? n/256 : 256;
      const localDistance = far ? d/256 : d;
      for (let block=0; block<(far ? 256 : n/256); block++) {
        for (let p=0; p<length/2; p++) {
          const u = Math.floor(p/localDistance)*2*localDistance + p%localDistance;
          const upper = far ? block+256*u : block*256+u;
          assert.ok(upper+d<n);
          assert.equal(seen[upper]++,0);
          assert.equal(seen[upper+d]++,0);
          butterflies++;
        }
      }
      assert.ok(seen.every(count => count===1));
    }
    assert.equal(butterflies,n/2*Math.log2(n));
  }
});
