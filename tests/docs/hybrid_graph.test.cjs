const {test}=require('node:test');
const assert=require('node:assert/strict');
const {layout}=require('../../docs/site-assets/hybrid-graph.js');
test('D8 explorer preserves full-key P positions and covers every active tower once per digit',()=>{
  for(let L=1;L<=5;L++){
    const g=layout(L);assert.deepEqual(g.keyIndices.slice(L),[5,6,7]);
    assert.deepEqual(g.keyIndices.slice(0,L),Array.from({length:L},(_,i)=>i));
    assert.equal(g.parts.flatMap(p=>p.own).length,L);
    let forward=0;
    for(const p of g.parts){assert.deepEqual([...p.own,...p.complement].sort(),[...g.active].sort());assert.equal(new Set([...p.own,...p.complement]).size,L+3);forward+=p.complement.length;}
    assert.equal(g.inverseNtts,L+6);assert.equal(g.forwardNtts,forward+2*L);
    assert.equal(g.digitBytes,g.parts.length*g.active.length*65536*8);
  }
});
test('full, shortened-last-digit, exact-boundary and single-tower cases match the executable fixture geometry',()=>{
  assert.deepEqual(layout(5).parts.map(p=>p.own),[['q0','q1','q2'],['q3','q4']]);
  assert.deepEqual(layout(4).parts.map(p=>p.own),[['q0','q1','q2'],['q3']]);
  assert.deepEqual(layout(3).parts.map(p=>p.own),[['q0','q1','q2']]);
  assert.deepEqual(layout(1).parts.map(p=>p.own),[['q0']]);
  assert.deepEqual([layout(5).inverseNtts,layout(5).forwardNtts],[11,21]);
  assert.equal(layout(999).L,5);assert.equal(layout(-1).L,1);assert.equal(layout(NaN).L,1);
});
