/* D8 fixture geometry. Kept separate from the cryptographic implementation. */
(function () {
  'use strict';
  function layout(level) {
    const L = Math.max(1, Math.min(5, Math.trunc(Number(level)) || 1));
    const M=5, K=3, alpha=3, beta=Math.ceil(L/alpha);
    const active=Array.from({length:L},(_,i)=>'q'+i).concat(['p0','p1','p2']);
    return {L,M,K,alpha,beta,active,keyIndices:active.map((_,i)=>i<L?i:i+M-L),
      parts:Array.from({length:beta},(_,d)=>{
        const start=d*alpha,end=Math.min(start+alpha,L);
        return {start,end,own:active.slice(start,end),complement:active.filter((_,i)=>i<start||i>=end)};
      }),
      inverseNtts:L+2*K,forwardNtts:beta*(L+K)-L+2*L,
      digitBytes:beta*(L+K)*65536*8,macBytes:2*(L+K)*65536*8};
  }
  if(typeof module!=='undefined'&&module.exports)module.exports={layout};
  if(typeof document==='undefined')return;
  document.querySelectorAll('[data-hybrid-explorer]').forEach(root=>{
    const select=root.querySelector('select'),out=root.querySelector('[data-hybrid-layout]');
    const el=(tag,value)=>{const n=document.createElement(tag);n.textContent=value;return n;};
    function render(){
      const g=layout(select.value);out.replaceChildren();
      out.appendChild(el('p',g.L+' active Q towers; '+g.beta+' active digit'+(g.beta===1?'':'s')+'. The digit width stays at 3, and P keeps all three towers.'));
      const wrap=document.createElement('div');wrap.className='site-table-scroll';wrap.tabIndex=0;wrap.setAttribute('role','region');wrap.setAttribute('aria-label','Active towers and full evaluation-key indices');
      const table=document.createElement('table'),head=document.createElement('thead'),body=document.createElement('tbody');
      const headers=document.createElement('tr');['Array','Tower order'].forEach(v=>{const th=el('th',v);th.scope='col';headers.appendChild(th);});head.appendChild(headers);
      const rows=[['Ciphertext QlP',g.active.join(' | ')],['Evaluation-key indices',g.keyIndices.join(' | ')]];
      g.parts.forEach((p,d)=>{rows.push(['Digit '+d+' original towers',p.own.join(', ')]);rows.push(['Digit '+d+' complement',p.complement.join(', ')]);});
      rows.forEach(row=>{const tr=document.createElement('tr');row.forEach(v=>tr.appendChild(el('td',v)));body.appendChild(tr);});
      table.append(head,body);wrap.appendChild(table);out.appendChild(wrap);
      out.appendChild(el('p','One KeySwitchCore: '+g.inverseNtts+' inverse NTTs and '+g.forwardNtts+' forward NTTs, counted per tower. Digit outputs request '+g.digitBytes.toLocaleString('en-US')+' bytes; the two QlP accumulators request '+g.macBytes.toLocaleString('en-US')+' bytes. These are array payloads, excluding keys and temporary storage.'));
    }
    select.addEventListener('change',render);render();
  });
})();
