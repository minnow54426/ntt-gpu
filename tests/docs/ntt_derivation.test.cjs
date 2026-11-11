// Validate the real lesson's twiddle explorer against independent direct NTT sums.
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const {runInNewContext} = require('node:vm');
const {test} = require('node:test');

function explorer() {
  const nodes = new Map();
  const get = selector => {
    if (!nodes.has(selector)) nodes.set(selector, {
      value:'', textContent:'', listeners:{},
      addEventListener(name, callback) { this.listeners[name] = callback; }
    });
    return nodes.get(selector);
  };
  const source = readFileSync(join(__dirname,'../../docs/butterfly-explained.html'),'utf8');
  const script = source.match(/<script id="handwritten-twiddle-script">([\s\S]*?)<\/script>/)[1];
  runInNewContext(script,{document:{getElementById:() => ({querySelector:get})}});
  const field = name => get('[data-note-'+name+']');
  return {
    field,
    set(name, value) {
      const node = field(name); node.value = String(value);
      node.listeners[['mode','n'].includes(name) ? 'change' : 'input']();
    },
    read() {
      return Object.fromEntries(['omega','psi','upper','lower','distance'].map(name =>
        [name,field(name).textContent === '—' ? null : Number(field(name).textContent)]));
    }
  };
}

function pow(base, exponent, q) {
  let value=1;
  for (let e=0;e<exponent;e++) value=value*base%q;
  return value;
}
function reverse(index,n) {
  return parseInt(index.toString(2).padStart(Math.log2(n),'0').split('').reverse().join(''),2);
}
function root(n,q) {
  for (let psi=2;psi<q;psi++) if (pow(psi,n,q)===q-1) return psi;
  throw new Error('No primitive 2N-th root');
}
function direct(input,q,psi,negacyclic) {
  return input.map((_,k) => input.reduce((sum,a,i) =>
    (sum+a*pow(psi, (negacyclic ? 2*k+1 : k)*i,q))%q,0));
}

for (const mode of ['dit','repo']) {
  test(mode+': every displayed butterfly reproduces independent direct transforms', () => {
    const page=explorer();page.set('mode',mode);
    for (const [n,q] of [[4,17],[8,17],[16,97]]) {
      page.set('n',n);
      const psi=root(n,q), omega=psi*psi%q;
      const patterns=[
        ...Array.from({length:n},(_,i) => Array.from({length:n},(_,j) => Number(i===j))),
        Array(n).fill(q-1),
        Array.from({length:n},(_,i) => (i*7+3)%q),
      ];
      for (const input of patterns) {
        const work=mode==='dit' ? input.map((_,k) => input[reverse(k,n)]) : [...input];
        for (let stage=0;stage<Math.log2(n);stage++) {
          page.set('stage',stage);
          const length=mode==='dit' ? 2**(stage+1) : n/2**stage;
          const seen=Array(n).fill(0);
          for (let group=0;group<n/length;group++) {
            page.set('group',group);
            for (let j=0;j<length/2;j++) {
              page.set('j',j);
              const {upper,lower,omega:exponent,psi:psiExponent,distance}=page.read();
              assert.equal(seen[upper]++,0);assert.equal(seen[lower]++,0);
              const w=pow(omega,exponent,q);
              const x=work[upper], y=work[lower];
              if (mode==='dit') {
                assert.equal(psiExponent,null);
                const t=y*w%q;
                work[upper]=(x+t)%q;work[lower]=(x-t+q)%q;
              } else {
                assert.equal(w,pow(psi,psiExponent,q));
                const cv=pow(psi,distance,q)*y%q;
                work[upper]=(x+cv)%q;work[lower]=(x-cv+q)*w%q;
              }
            }
          }
          assert.ok(seen.every(count=>count===1));
        }
        const expected=direct(input,q,mode==='dit' ? omega : psi,mode==='repo');
        assert.deepEqual(work,mode==='dit' ? expected : expected.map((_,k)=>expected[reverse(k,n)]));
      }
    }
  });
}

test('65536 boundary stages, local offsets, and group ownership clamp correctly', () => {
  const page=explorer();page.set('n',65536);page.set('stage',15);page.set('j',32767);
  assert.deepEqual(page.read(),{omega:32767,psi:null,upper:32767,lower:65535,distance:32768});
  page.set('mode','repo');page.set('stage',0);page.set('j',32767);
  assert.deepEqual(page.read(),{omega:32767,psi:65534,upper:32767,lower:65535,distance:32768});
  page.set('stage',999);page.set('group',99999);page.set('j',99999);
  assert.deepEqual(page.read(),{omega:0,psi:0,upper:65534,lower:65535,distance:1});
  page.set('j',-1);page.set('group',-4);page.set('stage',NaN);
  assert.equal(page.field('stage').value,'0');
  assert.equal(page.field('j').value,'0');assert.equal(page.field('group').value,'0');
  page.set('n',4);
  assert.equal(page.field('stage').max,'1');assert.equal(page.field('group').max,'0');
});

test('stage numbering changes the distance direction and labels the folded twist', () => {
  const page=explorer();page.set('n',16);page.set('stage',0);
  assert.equal(page.read().distance,1);
  assert.match(page.field('context').textContent,/输入先按位逆序/);
  page.set('mode','repo');
  assert.equal(page.read().distance,8);
  assert.match(page.field('result').textContent,/c = ψ\^d = ψ\^8/);
  assert.match(page.field('context').textContent,/自然输入/);
});
