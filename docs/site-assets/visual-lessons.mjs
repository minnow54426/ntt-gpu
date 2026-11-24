import {butterflyTrace, memoryPair, towerModel, transferModel, coefficients, primes, reverseBits, power} from './visual-models.mjs';

const ink = '#2b1d10', green = '#1f5d3e', orange = '#b5421a', muted = '#6b593f', paper = '#efe4cd';
const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const text = (x, y, value, size = 18, fill = ink, anchor = 'middle') => `<text x="${x}" y="${y}" text-anchor="${anchor}" font-size="${size}" fill="${fill}">${escape(value)}</text>`;
const line = (x1, y1, x2, y2, color = green, width = 2) => `<path d="M${x1} ${y1} L${x2} ${y2}" fill="none" stroke="${color}" stroke-width="${width}"/>`;
const circle = (x, y, r, fill = green) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}"/>`;
const table = (headers, rows) => '<div class="site-table-scroll" tabindex="0" role="region" aria-label="Exact demonstration values"><table><thead><tr>' + headers.map(h => `<th scope="col">${escape(h)}</th>`).join('') + '</tr></thead><tbody>' + rows.map(row => '<tr>' + row.map(cell => `<td>${escape(cell)}</td>`).join('') + '</tr>').join('') + '</tbody></table></div>';
const trace = butterflyTrace();
const mib = bytes => (bytes / 1048576).toLocaleString('en-US', {maximumFractionDigits:2}) + ' MiB';

function presentation(kind, state) {
  if (kind === 'butterfly') {
    const step = state.step, pair = trace.stages[Math.max(0, step - 1)][state.pair];
    let svg = '';
    for (let s = 0; s < 3; s++) for (const p of trace.stages[s]) {
      const selected = step === s + 1 && p === pair;
      for (const [u, v] of [[p.u,p.u],[p.u,p.v],[p.v,p.u],[p.v,p.v]]) svg += line(65+s*190, 60+u*36, 255+s*190, 60+v*36, selected ? orange : '#ad9d80', selected ? 3 : 1);
    }
    for (let col = 0; col < 4; col++) {
      svg += text(65+col*190, 25, col ? 'Stage '+col : 'Input', 18);
      for (let i = 0; i < 8; i++) {
        const selected = step && (col === step || col === step-1) && (i === pair.u || i === pair.v);
        svg += circle(65+col*190, 60+i*36, 14, selected ? orange : col <= step ? green : '#d2c4a8');
        if (col <= step) svg += text(65+col*190, 66+i*36, trace.states[col][i], 16, '#faf4e6');
      }
    }
    const formula = step ? `Pair ${pair.u} and ${pair.v}: a=${pair.a}, b=${pair.b}. The shared factor is ζ=3^${pair.exponent} mod 17=${pair.zeta}. First t=b×ζ mod 17=${pair.t}. Then (${pair.a}+${pair.t}) mod 17=${pair.x}, and (${pair.a}−${pair.t}) mod 17=${pair.y}. Both results are in 0…16.` : 'These are eight coefficient residues modulo 17, in ordinary array order. No stage has run yet. Advance one step: each of the four independent pairs shares one multiplication between its two outputs.';
    return {svg, caption: 'Columns are stages; height is array position. Orange marks the selected pair. Pale nodes have not run yet. The 3D angle changes only the viewpoint.',
      explanation: formula + (step === 3 ? ' All three stages are complete. Output slots are evaluations in bit-reversed order, not new coefficients.' : ''),
      values: table(['Slot', step === 0 ? 'Coefficient' : 'After stage '+step, ...(step === 3 ? ['Evaluation point'] : [])], trace.states[step].map((v,i) => [i,v,...(step === 3 ? ['3^'+(2*reverseBits(i,3)+1)+' mod 17 = '+power(3,2*reverseBits(i,3)+1,17)] : [])])),
      prompt: step === 0 ? 'Try Stage 1. Which two input positions does Pair 1 combine?' : 'Try another pair. Each array position belongs to exactly one pair in this stage.', pair};
  }
  if (kind === 'memory') {
    const p = memoryPair(state.step, state.block, state.thread), xy = i => [170+(i%256)*1.1, 35+Math.floor(i/256)*1.1];
    let svg = '';
    // The outline is the complete 256×256 array, not a sampled grid.
    for (let i=0;i<=256;i+=16) svg += line(170+i*1.1,35,170+i*1.1,316.6,'#c8b898',1)+line(170,35+i*1.1,451.6,35+i*1.1,'#c8b898',1);
    svg += p.column ? line(...xy(p.block),...xy(65280+p.block),green,4) : line(...xy(p.block*256),...xy(p.block*256+255),green,4);
    if (p.active) { svg += line(...xy(p.u),...xy(p.v),orange,3); for(const i of [p.u,p.v]) svg += circle(...xy(i),6,orange); }
    svg += text(310,22,'256 columns',18)+text(82,175,'256 rows',18)+text(575,120,p.column?'Column pass':'Row pass',21,green)+text(575,160,'Block '+p.block,19)+text(575,195,'Thread '+p.thread,19)+text(575,235,p.active?'2 paired positions':'No pair',19,orange)+text(310,340,'Flattened index = row × 256 + column',17);
    const explanation = `Forward stage ${p.stage+1} has global distance ${p.distance.toLocaleString('en-US')}. Block ${p.block} owns ${p.column ? 'column' : 'row'} ${p.block}, containing 256 positions. Thread ${p.thread} loads and stores position ${p.load}. ` + (p.active ? `Its butterfly reads positions ${p.u} and ${p.v}, separated by ${p.distance}. Each pair uses two shared-memory entries after the block has loaded its group.` : 'It has no butterfly: only threads 0–127 compute pairs. It still participates in loading, storing and every block barrier.');
    return {svg, caption:'Green marks the block-owned row or column; orange marks one pair. The labelled diagram groups 16 rows or columns; 3D draws all 65,536 positions. Spacing is not a bandwidth measurement.', explanation,
      values:table(['Item','Exact value'],[['Pass',p.column?'1 · columns':'2 · rows'],['Global distance',p.distance],['Load/store index',p.load],['Upper pair index',p.u ?? 'No butterfly'],['Lower pair index',p.v ?? 'No butterfly'],['Barrier participation','All 256 threads']]),
      prompt:'Compare Stage 8 and Stage 9, then set thread to 128. Explain why it still reaches the barrier.', memory:p};
  }
  if (kind === 'towers') {
    const m = towerModel(state.count,state.format);let svg = '';
    for (let k=0;k<m.count;k++) {
      const x = 120 + k*235;svg += text(x,32,'q'+k+' = '+primes[k],21,green);
      for(let i=0;i<4;i++) {
        const y=55+i*63, active=i===state.slot;
        svg += `<rect x="${x-75}" y="${y}" width="150" height="48" rx="5" fill="${active?orange:green}"/>` + text(x,y+31,'slot '+i+': '+m.towers[k][i],19,'#faf4e6');
      }
    }
    const meaning = state.format === 'coefficient' ? `Selected coefficient a${state.slot}=${coefficients[state.slot]}. Its remainders under the visible primes are ${m.towers.map(t=>t[state.slot]).join(', ')}.` : `Selected evaluation slot ${state.slot} contains ${m.towers.map(t=>t[state.slot]).join(', ')} under the visible primes. Each prime has its own root, so it also has its own evaluation points.`;
    return {svg, caption:'Each column is one prime tower. Orange marks the same array position across towers. The diagram shows all four values of a small example, not all 65,536 values of a production tower.',
      explanation: `${meaning} This N=4 example stores ${m.sampleBytes} bytes. At N=65,536, ${m.count} × 65,536 × 8 = ${m.bytes.toLocaleString('en-US')} bytes = ${mib(m.bytes)} per polynomial. Changing format changes values, not the number of words. Parameters, other polynomials and temporary buffers require additional storage.`,
      values:table(['Slot',...primes.slice(0,m.count).map(q=>'mod '+q)],coefficients.map((_,i)=>[i,...m.towers.map(t=>t[i])])),
      prompt:'Select coefficient position 1 to see how −2 becomes a nonnegative remainder. Then switch to evaluations and notice that the byte count stays the same.', towers:m};
  }
  const m=transferModel(state.mode,state.step), labels=['CPU inputs','Upload inputs','Forward transforms','Pointwise product','Inverse transform','CPU result'];
  const operations=['Waiting for upload','Inputs A and B','NTT(A), NTT(B)','NTT(A) × NTT(B)','Inverse NTT of product','Ready for next work'];
  const svg = `<rect x="25" y="75" width="170" height="170" rx="7" fill="#e3d4b3" stroke="${green}"/><rect x="260" y="60" width="220" height="200" rx="7" fill="${green}"/><rect x="545" y="75" width="150" height="170" rx="7" fill="#e3d4b3" stroke="${green}"/>`+
    line(195,160,260,160,orange,4)+line(480,160,545,160,orange,4)+text(110,115,'CPU',24)+text(110,155,'A and B',20)+text(110,195,'2 × 1.5 MiB',18)+text(370,100,'GPU',24,'#faf4e6')+text(370,150,state.step===2?'NTT(A), NTT(B)':state.step===3?'Pointwise product':state.step===4?'Inverse NTT':state.step===0?'Waiting':state.step===1?'Input arrays':'Result ready',18,'#faf4e6')+text(370,205,state.mode==='resident'?'Intermediates stay here':'Returns at each call',16,'#faf4e6')+text(620,115,'CPU',24)+text(620,155,'Result C',20)+text(620,195,'1.5 MiB',18)+text(360,305,labels[state.step]+' · copied so far: '+mib(m.copiedBytes),20);
  return {svg, caption:'Read left to right: input arrays, device operations, result array. Links mark a memory boundary; they do not encode measured duration. Separate calls also return intermediate arrays to CPU memory.',
    explanation: `${labels[state.step]}. ${operations[state.step]}. Counted so far: ${mib(m.copiedBytes)} of polynomial payload. ` + (state.mode==='resident' ? 'Keeping arrays on the GPU uploads two payloads and downloads one: 4.5 MiB in total. The baseline fused multiply has the same payload count, but a different API and lifetime behavior.' : 'Separate forward(A), forward(B), pointwise and inverse host-vector calls move 2 + 2 + 3 + 2 payloads: 13.5 MiB in total. Extra copies at each call return are included in the count.') + ' This excludes setup constants and allocations. Fewer bytes alone do not establish a speedup.',
    values:table(['Boundary or total','Polynomial payload'],state.mode==='resident'?[['Input upload','2 × 1.5 MiB'],['Intermediate CPU copies','0'],['Final download','1 × 1.5 MiB'],['Total',mib(m.totalBytes)]]:[['forward(A) + forward(B)','4 × 1.5 MiB'],['pointwise(A, B)','3 × 1.5 MiB'],['inverse(product)','2 × 1.5 MiB'],['Total',mib(m.totalBytes)]]),
    prompt:'Finish the steps, switch the call pattern, and explain where the extra six polynomial copies come from.', transfers:m};
}

// Three.js is imported only when requested. The exact SVG model is the default.
let threePromise;
async function create3D(container, onLost) {
  const T = await (threePromise ??= import('../vendor/three/three.module.min.js'));
  const renderer = new T.WebGLRenderer({antialias:true,alpha:false});
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1,1.5));renderer.setClearColor(paper);
  renderer.domElement.setAttribute('aria-hidden','true');container.append(renderer.domElement);
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(38,1,0.1,100);
  let group = new T.Group();scene.add(group);let currentKind='';
  function disposeGroup() {
    group.traverse(o=>{o.geometry?.dispose();if(o.material){o.material.map?.dispose();o.material.dispose();}});
    scene.remove(group);group=new T.Group();scene.add(group);
  }
  function label(value,x,y,z,size=.35,color=ink) {
    const canvas=document.createElement('canvas'),ctx=canvas.getContext('2d');ctx.font='500 56px sans-serif';
    canvas.width=Math.ceil(ctx.measureText(String(value)).width)+24;canvas.height=80;
    ctx.font='500 56px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillStyle=color;ctx.fillText(String(value),canvas.width/2,40);
    const texture=new T.CanvasTexture(canvas);texture.colorSpace=T.SRGBColorSpace;
    const sprite=new T.Sprite(new T.SpriteMaterial({map:texture,depthTest:false}));sprite.position.set(x,y,z);
    sprite.userData.label={aspect:canvas.width/80,pixels:Math.max(21,size*65)};group.add(sprite);
  }
  function point(x,y,z,color=green,r=.1) {const o=new T.Mesh(new T.SphereGeometry(r,10,8),new T.MeshBasicMaterial({color}));o.position.set(x,y,z);group.add(o);}
  function edge(a,b,color=green) {const g=new T.BufferGeometry().setFromPoints([new T.Vector3(...a),new T.Vector3(...b)]);group.add(new T.Line(g,new T.LineBasicMaterial({color})));}
  function box(x,y,z,w,h,d,color=green) {const mesh=new T.Mesh(new T.BoxGeometry(w,h,d),new T.MeshBasicMaterial({color}));mesh.position.set(x,y,z);group.add(mesh);}
  function draw(kind,state,model) {
    disposeGroup();currentKind=kind;
    if(kind==='butterfly') {
      const y=i=>2.5-i*.72;
      for(let s=0;s<3;s++)for(const p of trace.stages[s])for(const [a,b] of [[p.u,p.u],[p.u,p.v],[p.v,p.u],[p.v,p.v]]) edge([-4.5+s*3,y(a),0],[-1.5+s*3,y(b),0],state.step===s+1&&p===model.pair?orange:'#ad9d80');
      for(let s=0;s<4;s++) {label(s?'Stage '+s:'Input',-4.5+s*3,3.3,0);for(let i=0;i<8;i++){const active=state.step&&(s===state.step||s===state.step-1)&&(i===model.pair.u||i===model.pair.v);point(-4.5+s*3,y(i),0,active?orange:s<=state.step?green:'#c8b898',.24);if(s<=state.step)label(trace.states[s][i],-4.5+s*3,y(i),0,.25,'#faf4e6');}}
    } else if(kind==='memory') {
      const p=model.memory,positions=new Float32Array(65536*3),colors=new Float32Array(65536*3),normal=new T.Color('#b4a382'),owned=new T.Color(green);
      for(let i=0;i<65536;i++){const row=Math.floor(i/256),col=i%256;positions.set([(col/255-.5)*8,0,(row/255-.5)*8],i*3);colors.set((p.column?col===p.block:row===p.block)?[owned.r,owned.g,owned.b]:[normal.r,normal.g,normal.b],i*3);}
      const geometry=new T.BufferGeometry();geometry.setAttribute('position',new T.BufferAttribute(positions,3));geometry.setAttribute('color',new T.BufferAttribute(colors,3));group.add(new T.Points(geometry,new T.PointsMaterial({size:.025,vertexColors:true})));
      const pos=i=>[(i%256/255-.5)*8,.08,(Math.floor(i/256)/255-.5)*8];
      const first=p.column?p.block:p.block*256,last=p.column?65280+p.block:p.block*256+255;edge(pos(first),pos(last),green);
      if(p.active){for(const i of [p.u,p.v])point(...pos(i),orange,.12);edge(pos(p.u),pos(p.v),orange);}
      label('256 columns',0,-.7,-5.3,.4);label('256 rows',-5.3,-.3,0,.4);
    } else if(kind==='towers') {
      const m=model.towers;
      for(let k=0;k<m.count;k++){const x=(k-(m.count-1)/2)*3;label('q'+k+' = '+primes[k],x,2.7,0,.42);for(let i=0;i<4;i++){box(x,1.6-i*1.1,0,2,.7,.55,i===state.slot?orange:green);label('slot '+i+': '+m.towers[k][i],x,1.6-i*1.1,.4,.3,'#faf4e6');}}
    } else {
      box(-3.8,0,0,2,2,.6,'#d6c5a2');box(0,0,0,3,3,1,green);box(3.8,0,0,2,2,.6,'#d6c5a2');
      edge([-2.8,0,.3],[-1.5,0,.3],orange);edge([1.5,0,.3],[2.8,0,.3],orange);
      label('CPU inputs',-3.8,1.6,0,.35);label('GPU',0,2.2,0,.45);label('CPU result',3.8,1.6,0,.35);
      label('A + B',-3.8,0,.5,.4);label('C',3.8,0,.5,.4);label(['Waiting','A + B','NTT A, NTT B','Pointwise','Inverse NTT','Result ready'][state.step],0,0,.8,.4,'#faf4e6');
      label('Copied: '+mib(model.transfers.copiedBytes),0,-2.2,0,.42);
    }
  }
  function render(angle) {
    const rect=container.getBoundingClientRect();if(!rect.width||!rect.height)return;
    renderer.setSize(rect.width,rect.height,false);camera.aspect=rect.width/rect.height;
    // Fit the entire model at phone widths as well as desktop widths.
    const distance=Math.max(currentKind==='memory'?16:12,16/camera.aspect),radians=angle*Math.PI/180;
    if(currentKind==='memory')camera.position.set(Math.sin(radians)*distance*.5,distance*.85,Math.cos(radians)*distance*.65);
    else camera.position.set(Math.sin(radians)*distance*.35,1,Math.cos(radians*.35)*distance);
    camera.lookAt(0,0,0);camera.updateProjectionMatrix();
    group.traverse(o=>{if(o.userData.label){const {pixels,aspect}=o.userData.label;const height=2*camera.position.distanceTo(o.position)*Math.tan(19*Math.PI/180)*pixels/rect.height;o.scale.set(height*aspect,height,1);}});
    renderer.render(scene,camera);
  }
  renderer.domElement.addEventListener('webglcontextlost',event=>{event.preventDefault();onLost();});
  return {draw,render,dispose(){disposeGroup();renderer.dispose();renderer.domElement.remove();}};
}

for(const root of document.querySelectorAll('[data-visual]')) {
  const kind=root.dataset.visual,fields=[...root.querySelectorAll('[data-field]')],initial=Object.fromEntries(fields.map(f=>[f.dataset.field,f.value]));
  const svg=root.querySelector('[data-visual-svg]'),viewport=root.querySelector('[data-visual-webgl]'),button=root.querySelector('[data-3d]'),status=root.querySelector('[data-renderer-status]'),camera=root.querySelector('[data-camera]');
  let state,model,view=null,show3D=false;
  const numeric=new Set(['step','pair','block','thread','count','slot']);
  function read() {
    state=Object.fromEntries(fields.map(f=>{
      let value=f.value;
      if(numeric.has(f.dataset.field)) {const max=f.tagName==='SELECT'?Number(f.options[f.options.length-1].value):Number(f.max);value=Math.max(0,Math.min(max,Math.trunc(Number(value))||0));f.value=String(value);}
      return [f.dataset.field,numeric.has(f.dataset.field)?Number(value):value];
    }));
  }
  function draw() {
    read();model=presentation(kind,state);svg.innerHTML=model.svg;
    root.querySelector('[data-visual-caption]').textContent=model.caption;
    root.querySelector('[data-visual-explanation]').textContent=model.explanation;
    root.querySelector('[data-visual-values]').innerHTML=model.values;
    root.querySelector('[data-visual-prompt]').textContent=model.prompt;
    const step=fields.find(f=>f.dataset.field==='step');
    for(const [selector,disabled] of [['[data-previous]',!step||step.selectedIndex===0],['[data-next]',!step||step.selectedIndex===step.options.length-1]])root.querySelector(selector).disabled=disabled;
    if(!step)root.querySelector('.visual-step-buttons').hidden=true;
    if(view&&show3D){view.draw(kind,state,model);view.render(Number(camera.value));}
  }
  function fallback(message) {
    show3D=false;svg.removeAttribute('hidden');viewport.hidden=true;button.setAttribute('aria-pressed','false');button.textContent='Open 3D view';root.querySelector('[data-camera-control]').hidden=true;status.textContent=message;
  }
  root.querySelectorAll('[data-visual-controls]').forEach(e=>e.hidden=false);
  status.textContent='Labelled diagram active. Optional 3D loads only when you open it. Every control works in diagram view.';
  fields.forEach(f=>f.addEventListener(f.tagName==='SELECT'?'change':'input',draw));
  for(const [selector,delta]of [['[data-previous]',-1],['[data-next]',1]])root.querySelector(selector).addEventListener('click',()=>{const field=fields.find(f=>f.dataset.field==='step');field.selectedIndex=Math.max(0,Math.min(field.options.length-1,field.selectedIndex+delta));draw();});
  root.querySelector('[data-reset]').addEventListener('click',()=>{fields.forEach(f=>f.value=initial[f.dataset.field]);camera.value='25';draw();});
  camera.addEventListener('input',()=>{if(view&&show3D)view.render(Number(camera.value));});
  button.addEventListener('click',async()=>{
    if(show3D){fallback('Labelled diagram active. All values and controls are unchanged.');return;}
    button.disabled=true;status.textContent='Loading the local Three.js renderer…';viewport.hidden=false;
    try {
      if(!view)view=await create3D(viewport,()=>{const lost=view;view=null;lost?.dispose();fallback('3D context was lost. The labelled diagram and exact controls remain available.');});
      show3D=true;svg.setAttribute('hidden','');button.setAttribute('aria-pressed','true');button.textContent='Use labelled diagram';root.querySelector('[data-camera-control]').hidden=false;
      view.draw(kind,state,model);view.render(Number(camera.value));status.textContent='Three.js 3D view active. Use Rotate view to change the angle. There is no automatic motion.';
    } catch(error) {view?.dispose();view=null;threePromise=null;viewport.replaceChildren();fallback('3D is unavailable in this browser. The labelled diagram, exact values and all controls still work.');}
    finally {button.disabled=false;}
  });
  const observer=new ResizeObserver(()=>{if(view&&show3D)view.render(Number(camera.value));});observer.observe(viewport);
  window.addEventListener('pagehide',event=>{if(!event.persisted){observer.disconnect();view?.dispose();view=null;}});
  draw();
}
