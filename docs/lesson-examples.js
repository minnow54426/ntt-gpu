/* The teaching model uses the repository's folded negacyclic DIF convention. */
(function (scope) {
  'use strict';
  const Q = 97, PSI = 19, N = 16;
  const mod = x => ((x % Q) + Q) % Q;
  function power(e) { let x=1; for(let i=0;i<e;i++) x=x*PSI%Q; return x; }
  function trace(input) {
    if (input.length!==N || input.some(x=>!Number.isSafeInteger(x))) throw new Error('需要 16 个整数。');
    const states=[input.map(mod)], stages=[];
    for(let d=8;d>=1;d/=2) {
      const before=states.at(-1), after=[...before], pairs=[];
      for(let base=0;base<N;base+=2*d) for(let j=0;j<d;j++) {
        const u=base+j,v=u+d,c=power(d),e=N/d*j,w=power(e);
        const a=before[u],b=before[v],cv=mod(c*b),x=mod(a+cv),y=mod((a-cv)*w);
        after[u]=x;after[v]=y;pairs.push({u,v,a,b,c,e,w,cv,x,y,d,j});
      }
      states.push(after);stages.push(pairs);
    }
    return {states,stages};
  }
  if(typeof module!=='undefined' && module.exports) module.exports={trace,power,Q,PSI};
  if(typeof document==='undefined') return;
  const style=document.createElement('style');
  style.textContent=`.lesson16{border:1px solid #c9d3bf;border-radius:10px;padding:18px;margin:24px 0;background:#fafbf5;color:#253e31;min-width:0}.lesson16 label{display:block;font-size:14px;margin:10px 0}.lesson16 input{box-sizing:border-box;width:100%;padding:9px;font:14px monospace;border:1px solid #b6c3ae;background:white;color:#253e31}.lesson16 button,.lesson16 select{padding:8px;margin:4px;border:1px solid #a5b39c;border-radius:5px;background:white;color:#253e31;font-size:14px}.lesson16 button[aria-pressed=true]{background:#254d39;color:white}.lesson16 .lesson-scroll{overflow:auto;max-width:100%;margin:14px 0}.lesson16 svg{display:block;min-width:720px;width:100%}.lesson16 table{border-collapse:collapse;min-width:680px;width:100%;font:14px/1.7 monospace}.lesson16 th,.lesson16 td{padding:7px;border-bottom:1px solid #dbe2d4;text-align:left;white-space:nowrap}.lesson16 .lesson-calc{padding:12px;background:#e8efde;line-height:1.9;overflow-wrap:anywhere}.lesson16 .katex-display{overflow-x:auto;overflow-y:hidden;font-size:15px}.lesson16 [role=status]{font-size:13px}.lesson16 .lesson-group{border-left:3px solid #638755;padding-left:12px;margin:20px 0}.lesson16 p{font-size:15px;line-height:1.8}.lesson16 h4{margin:16px 0 8px}`;
  document.head.appendChild(style);
  const text=(tag,content)=>{const el=document.createElement(tag);el.textContent=content;return el;};
  document.querySelectorAll('[data-lesson-sixteen]').forEach(root=>{
    root.classList.add('lesson16');
    root.innerHTML=`<h3>16 点完整演算：四层、32 个蝶形</h3><p>模数 q=97，本原 32 次根 ψ=19，ψ¹⁶≡−1。使用仓库的 negacyclic DIF：自然系数输入，位逆序求值输出。所有数字均已取模。选择 stage 和配对可查看乘子与算式。</p><label>16 个输入系数（空格或逗号分隔）<input data-input aria-label="16 个输入系数" value="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"></label><p role="status" data-status></p><div data-stages role="group" aria-label="选择 stage"></div><label>本层的八个蝶形<select data-pair aria-label="选择蝶形"></select></label><div class="lesson-calc" data-calc aria-live="polite"></div><p>网络图与明细表可左右滑动；每列是该层结束后的全局数组。</p><div class="lesson-scroll" tabindex="0" role="region" aria-label="可横向滚动的四层网络"><svg viewBox="0 0 850 620" role="img" aria-label="16 点四层完整蝶形网络，每列标注当前数组值"></svg></div><details><summary>本层全部八个蝶形的输入、因子和输出</summary><div class="lesson-scroll matrix-scroll"><table><thead><tr><th>位置 u ↔ v</th><th>输入 a, b</th><th>c</th><th>w</th><th>输出 x, y</th></tr></thead><tbody data-rows></tbody></table></div></details><div data-groups></div><p>这是将 16 点分成 4×4 的教学示例。N=65536 时对应 256×256，前八层归列，后八层归行；实现仍使用同一个全局 N 的乘子。</p>`;
    const get=s=>root.querySelector(s);let stage=0,pair=0,model=trace(Array.from({length:N},(_,i)=>i+1));
    const stages=get('[data-stages]');
    for(let s=0;s<4;s++){const b=text('button','Stage '+s+' · d='+2**(3-s));b.type='button';b.onclick=()=>{stage=s;render();};stages.appendChild(b);}
    function grouped(title,indices,stageIds){
      const box=document.createElement('div');box.className='lesson-group';box.appendChild(text('h4',title));
      box.appendChild(text('p','全局位置：'+indices.join(', ')+'。进入这两层的数值：['+indices.map(i=>model.states[stageIds[0]][i]).join(', ')+']。'));
      for(const s of stageIds){
        const selected=model.stages[s].filter(p=>indices.includes(p.u)&&indices.includes(p.v));
        box.appendChild(text('p','Stage '+s+'：'+selected.map(p=>p.u+'↔'+p.v+'，('+p.a+', '+p.b+') → ('+p.x+', '+p.y+')；c='+p.c+'，w='+p.w).join('； ')));
      }
      const last=stageIds.at(-1)+1;box.appendChild(text('p','两层后：['+indices.map(i=>model.states[last][i]).join(', ')+']。'));return box;
    }
    function render(){
      [...stages.children].forEach((b,i)=>b.setAttribute('aria-pressed',String(i===stage)));
      const pairs=model.stages[stage];get('[data-pair]').replaceChildren(...pairs.map((p,i)=>{const o=text('option',p.u+' ↔ '+p.v);o.value=i;return o;}));get('[data-pair]').value=pair;
      const p=pairs[pair];
      get('[data-calc]').textContent='位置 '+p.u+' ↔ '+p.v+'：输入 ('+p.a+', '+p.b+')。' +
        '\\[c=\\psi^{'+p.d+'}='+p.c+',\\quad w=\\psi^{'+p.e+'}='+p.w+'\\]' +
        '\\[cb\\equiv '+p.cv+'\\pmod{97}\\]' +
        '\\[x=('+p.a+'+'+p.cv+')\\bmod97='+p.x+'\\]' +
        '\\[y=('+p.a+'-'+p.cv+')\\cdot'+p.w+'\\bmod97='+p.y+'\\]';
      if(typeof renderMathInElement==='function')renderMathInElement(get('[data-calc]'),{delimiters:[{left:'\\[',right:'\\]',display:true}],throwOnError:true});
      get('[data-rows]').replaceChildren(...pairs.map(p=>{const tr=document.createElement('tr');[p.u+' ↔ '+p.v,p.a+', '+p.b,p.c,p.w,p.x+', '+p.y].forEach(v=>tr.appendChild(text('td',v)));return tr;}));
      const svg=get('svg'),ns='http://www.w3.org/2000/svg';svg.replaceChildren();
      function node(tag,attrs,label){const el=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));if(label!==undefined)el.textContent=label;svg.appendChild(el);}
      const xs=[65,240,415,590,765], y=i=>60+34*i;
      for(let s=0;s<4;s++)for(const p of model.stages[s])for(const [a,b] of [[p.u,p.u],[p.u,p.v],[p.v,p.u],[p.v,p.v]])node('line',{x1:xs[s]+22,y1:y(a),x2:xs[s+1]-22,y2:y(b),stroke:s===stage&&(p===pairs[pair])?'#b64d16':'#9bac92','stroke-width':s===stage&&p===pairs[pair]?2.5:0.7,opacity:s===stage?1:0.45});
      for(let col=0;col<5;col++){
        node('text',{x:xs[col],y:22,'text-anchor':'middle','font-size':14,fill:'#253e31'},col===0?'输入':'Stage '+(col-1));
        if(col)node('text',{x:xs[col],y:40,'text-anchor':'middle','font-size':12,fill:'#53654a'},'d='+2**(4-col));
        for(let i=0;i<N;i++){
          node('rect',{x:xs[col]-19,y:y(i)-12,width:38,height:23,rx:5,fill:'#fafbf5'});
          node('text',{x:xs[col],y:y(i)+5,'text-anchor':'middle','font-size':15,fill:'#253e31'},model.states[col][i]);
          if(col===0)node('text',{x:25,y:y(i)+4,'font-size':12,fill:'#637359'},i);
        }
      }
      get('[data-groups]').replaceChildren(text('h4','把相邻两层放在一起：为什么可分列、再分行？'),text('p','先按 4×4 看全局数组。列组必须收齐跨行的伙伴；经过前两层后，后两层的伙伴都在同一连续行内。行组读取的是中间结果。'),grouped('前两层 · 第 3 列（跨度为 4）',[3,7,11,15],[0,1]),grouped('后两层 · 第 3 行（连续四点）',[12,13,14,15],[2,3]));
      get('[data-status]').textContent='输入有效 · 四层完成 · 输出按 4 位位逆序排列。';
    }
    get('[data-pair]').onchange=()=>{pair=Number(get('[data-pair]').value);render();};
    get('[data-input]').oninput=()=>{try{const raw=get('[data-input]').value.trim().split(/[\s,，]+/).map(Number);model=trace(raw);get('[data-input]').removeAttribute('aria-invalid');render();}catch(e){get('[data-input]').setAttribute('aria-invalid','true');get('[data-status]').textContent=e.message+' 图中保留上一次有效计算。';}};
    render();
  });
})(typeof globalThis!=='undefined'?globalThis:this);
