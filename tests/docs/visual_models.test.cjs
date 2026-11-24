const {test} = require('node:test');
const assert = require('node:assert/strict');
const models = import('../../docs/site-assets/visual-models.mjs');

test('3D butterfly teaching values equal independent direct negacyclic evaluation', async () => {
  const {butterflyTrace} = await models;
  const pow=(b,e,q)=>Number(BigInt(b)**BigInt(e)%BigInt(q));
  const patterns=[Array(8).fill(0),Array(8).fill(16),[1,-2,20,100,0,16,17,-18],
    ...Array.from({length:8},(_,i)=>Array.from({length:8},(_,j)=>Number(i===j)))];
  for(const input of patterns){
    const {states,stages}=butterflyTrace(input);
    for(const pairs of stages){const touched=new Set();for(const p of pairs){touched.add(p.u);touched.add(p.v);assert.ok(p.x>=0&&p.x<17&&p.y>=0&&p.y<17);}assert.equal(touched.size,8);}
    const expected=Array.from({length:8},(_,slot)=>{
      const k=parseInt(slot.toString(2).padStart(3,'0').split('').reverse().join(''),2);
      const value=input.reduce((sum,a,i)=>sum+BigInt(a)*BigInt(pow(3,(2*k+1)*i,17)),0n);
      return Number((value%17n+17n)%17n);
    });
    assert.deepEqual(states.at(-1),expected);
  }
  assert.throws(()=>butterflyTrace([1,2]));
});

test('all sixteen displayed memory stages cover every address exactly once', async () => {
  const {memoryPair}=await models;
  for(let stage=0;stage<16;stage++){
    const visits=new Uint8Array(65536);
    for(let block=0;block<256;block++)for(let thread=0;thread<128;thread++){
      const p=memoryPair(stage,block,thread);
      assert.equal(p.u^p.v,32768>>stage);
      assert.equal(visits[p.u]++,0);assert.equal(visits[p.v]++,0);
      if(stage<8){assert.equal(p.u%256,block);assert.equal(p.v%256,block);}
      else {assert.equal(Math.floor(p.u/256),block);assert.equal(Math.floor(p.v/256),block);}
    }
    assert.ok(visits.every(count=>count===1));
    for(const thread of [128,255]){
      const p=memoryPair(stage,255,thread);assert.equal(p.active,false);assert.equal(p.u,null);assert.equal(p.v,null);assert.ok(p.load>=0&&p.load<65536);
    }
  }
  assert.deepEqual(memoryPair(NaN,-1,900),memoryPair(0,0,255));
});

test('prime towers show canonical residues, exact evaluations and full-size payloads', async () => {
  const {towerModel}=await models;
  assert.deepEqual(towerModel().towers,[[1,15,3,15],[1,39,20,18],[1,71,20,27]]);
  assert.deepEqual(towerModel(1,'evaluation').towers,[[10,16,3,9]]);
  for(let count=1;count<=3;count++){
    const m=towerModel(count),e=towerModel(count,'evaluation');
    assert.equal(m.bytes,count*524288);assert.equal(m.sampleBytes,count*32);assert.equal(e.bytes,m.bytes);
    assert.equal(m.towers.length,count);assert.equal(e.towers.length,count);
  }
});

test('transfer totals agree with the host API call boundaries and remain monotone', async () => {
  const {transferModel}=await models;
  const S=3*65536*8;
  const separateCalls=[{input:1,output:1},{input:1,output:1},{input:2,output:1},{input:1,output:1}];
  assert.equal(transferModel('separate',5).totalBytes,separateCalls.reduce((bytes,c)=>bytes+(c.input+c.output)*S,0));
  assert.equal(transferModel('resident',5).totalBytes,3*S);
  for(const mode of ['resident','separate']){
    let prior=0;
    for(let step=0;step<=5;step++){const m=transferModel(mode,step);assert.ok(m.copiedBytes>=prior);assert.ok(m.copiedBytes<=m.totalBytes);prior=m.copiedBytes;}
    assert.equal(prior,transferModel(mode,5).totalBytes);
  }
});
