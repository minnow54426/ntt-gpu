const {test}=require('node:test');
const assert=require('node:assert/strict');
const {trace,power,Q,PSI}=require('../../docs/lesson-examples.js');
const mod=x=>((x%Q)+Q)%Q;
const rev=k=>parseInt(k.toString(2).padStart(4,'0').split('').reverse().join(''),2);
function direct(input){return input.map((_,k)=>input.reduce((sum,a,i)=>mod(sum+mod(a)*power((2*rev(k)+1)*i)),0));}
test('complete 16-point lesson matches direct odd-root evaluation for all basis vectors and signed inputs',()=>{
  assert.equal(power(16),Q-1);assert.equal(power(32),1);assert.equal(PSI,19);
  const inputs=[...Array.from({length:16},(_,i)=>Array.from({length:16},(_,j)=>Number(i===j))),Array(16).fill(96),Array.from({length:16},(_,i)=>i+1),Array.from({length:16},(_,i)=>i*17-201)];
  for(const input of inputs){const model=trace(input);assert.deepEqual(model.states.at(-1),direct(input));
    assert.deepEqual(model.stages.map(ps=>ps[0].d),[8,4,2,1]);
    for(let s=0;s<4;s++){const seen=[];for(const p of model.stages[s]){seen.push(p.u,p.v);assert.equal(p.a,model.states[s][p.u]);assert.equal(p.b,model.states[s][p.v]);assert.equal(p.x,model.states[s+1][p.u]);assert.equal(p.y,model.states[s+1][p.v]);}assert.deepEqual(seen.sort((a,b)=>a-b),Array.from({length:16},(_,i)=>i));}
  }
});
test('both two-stage groupings stay closed and reproduce the full transform intermediate values',()=>{
  const full=trace(Array.from({length:16},(_,i)=>i+1));
  for(const [indices,stages] of [[[3,7,11,15],[0,1]],[[12,13,14,15],[2,3]]]){
    const work=new Map(indices.map(i=>[i,full.states[stages[0]][i]]));
    for(const s of stages){const pairs=full.stages[s].filter(p=>indices.includes(p.u));assert.equal(pairs.length,2);
      for(const p of pairs){assert.ok(indices.includes(p.v));const a=work.get(p.u),b=work.get(p.v);work.set(p.u,mod(a+p.c*b));work.set(p.v,mod((a-p.c*b)*p.w));}
      for(const i of indices)assert.equal(work.get(i),full.states[s+1][i]);
    }
  }
});
test('invalid lesson inputs are rejected without computing a partial network',()=>{assert.throws(()=>trace([1,2]));assert.throws(()=>trace(Array(16).fill(NaN)));assert.throws(()=>trace(Array(16).fill(1.5)));});
test('each intermediate layer equals cyclic DIF after removing the remaining local coefficient twist',()=>{
  for(const input of [Array.from({length:16},(_,i)=>i+1),Array(16).fill(96),Array.from({length:16},(_,i)=>i%3===0?1:0)]){
    const folded=trace(input);let cyclic=input.map((a,i)=>mod(a*power(i)));
    for(let s=0,d=8;s<4;s++,d/=2){const next=[...cyclic];
      for(let base=0;base<16;base+=2*d)for(let j=0;j<d;j++){
        const u=base+j,v=u+d;next[u]=mod(cyclic[u]+cyclic[v]);next[v]=mod((cyclic[u]-cyclic[v])*power(16/d*j));
      }
      for(let i=0;i<16;i++)assert.equal(next[i],mod(folded.states[s+1][i]*power(i%d)));
      cyclic=next;
    }
  }
});
