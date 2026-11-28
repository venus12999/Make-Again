/* ================= 植物装饰 · 花体字 · 花瓣（参考紫藤文件夹视频） =================
   每个文件夹一种植物：信件=紫藤，物品=铃兰，照片=樱花，记忆=薰衣草。
   目前用代码绘制；拿到透明底 PNG 素材后可在 FLORA_IMG 里替换成图片（见策划案）。 */
const FLORA = {
  letters:  { script: 'Letters',   unit: '封',   glyph: 'wisteria', petal: ['#CDB3F5', '#A78BCF', '#B9A3E3', '#C4ACEC'] },
  items:    { script: 'Keepsakes', unit: '件',   glyph: 'lily',     petal: ['#F1DFE6', '#E6CCD8', '#FBEEF1', '#EBD5DE'] },
  photos:   { script: 'Photos',    unit: '张',   glyph: 'sakura',   petal: ['#F2B3CD', '#E79BBB', '#F8CFE0', '#EEB0C8'] },
  memories: { script: 'Memories',  unit: '段',   glyph: 'lavender', petal: ['#A9B2EC', '#8E98DC', '#C3C9F4', '#9FA6E4'] }
};
function rngOf(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }
function bez(p0, p1, p2, p3, t) { const u = 1 - t; return [u*u*u*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t*t*t*p3[0], u*u*u*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t*t*t*p3[1]]; }

/* 生成一株植物的结构（以文件夹宽度 fw 为单位，坐标为文件夹内部坐标） */
function floraSpec(kind, g, seed) {
  const R = rngOf(seed), fw = g.fw, ft = g.bh - g.fh, S = { kind, items: [] };
  const vine = [[.1*fw, ft + .04*fw], [.12*fw, -.86*fw], [1.02*fw, -.9*fw], [1.16*fw, ft - .02*fw]];
  S.vine = vine;
  if (kind === 'letters') {           // 紫藤：藤蔓 + 下垂花串
    for (let i = 0; i < 6; i++) S.items.push({ t: .12 + i*.14, type: 'leaf', side: i % 2 ? 1 : -1, size: fw*(.09 + R()*.04) });
    const N = 13;
    for (let i = 0; i < N; i++) { const t = .22 + i*(.76/(N-1)) + (R()-.5)*.03;
      S.items.push({ t, type: 'raceme', len: fw*(.34 + R()*.2 + (t > .8 ? .2 : 0)), front: t > .86, ph: R()*6.28 }); }
  } else if (kind === 'items') {      // 铃兰：宽叶 + 弯茎小铃铛
    S.vine = null;
    [[.28, -.05], [.62, .08], [.45, -.12]].forEach(([x, r]) => S.items.push({ type: 'blade', x: x*fw, y: ft + .05*fw, h: fw*(.85 + R()*.15), r, w: fw*.16, t: 0 }));
    for (let i = 0; i < 4; i++) S.items.push({ type: 'lilystem', x: (.22 + i*.18)*fw, y: ft + .04*fw, h: fw*(.66 + R()*.2), lean: .18 + R()*.1, bells: 8 + (R()*3|0), t: .15 + i*.14, front: i === 3, ph: R()*6.28 });
  } else if (kind === 'photos') {     // 樱花：枝条 + 五瓣花团
    S.vine = [[.05*fw, ft + .02*fw], [.2*fw, -.78*fw], [.9*fw, -.82*fw], [1.18*fw, ft - .14*fw]];
    for (let i = 0; i < 16; i++) { const t = .1 + i*.055; S.items.push({ t, type: 'blossom', n: 4 + (R()*4|0), spread: fw*(.08 + R()*.06), front: t > .9, ph: R()*6.28 }); }
    for (let i = 0; i < 4; i++) S.items.push({ t: .2 + i*.2, type: 'twig', ang: (R()-.5)*1.6, len: fw*(.12 + R()*.08), ph: R()*6.28 });
  } else {                            // 薰衣草：从夹口竖起的花穗
    S.vine = null;
    for (let i = 0; i < 15; i++) { const x = (.12 + i*.052 + (R()-.5)*.03)*fw;
      S.items.push({ type: 'spike', x, y: ft + .06*fw, h: fw*(.62 + R()*.38), lean: (x/fw - .5)*.5 + (R()-.5)*.12, t: .05 + i*.05 + R()*.05, front: i === 14, ph: R()*6.28 }); }
  }
  return S;
}


/* ---------- 紫藤（算法取自「Make Again 紫藤动效 · 弧形花串」） ---------- */
const WPAL = { letters: [[140,112,222],[80,90,24]], items: [[212,186,200],[42,60,46]], photos: [[216,124,168],[36,88,62]], memories: [[100,114,208],[104,98,42]] };
function wisteriaSpec(g, seed, kind) {
  let sd = seed >>> 0; const rnd = () => ((sd = (Math.imul(sd, 1664525) + 1013904223) >>> 0) / 4294967296);
  const SZ = .68, k = g.fw / 354, ft = g.bh - g.fh, M = (X, Y) => ({ x: (X - 270) * k, y: ft + (Y - 264) * k });
  const vine = u => M(314 + 322 * u, 256 - 92 * Math.sin(Math.PI * u * .88));
  const clusters = [.03, .15, .26, .36, .48, .58, .67, .75, .84, .94, 1].map((u, i) => {
    const root = vine(u), front = i >= 8, length = [49, 68, 95, 112, 102, 83, 114, 124, 162, 193, 155][i] * k * 1.08 * SZ;
    const flowers = Array.from({ length: Math.round(length / k / SZ * 2.0) }, () => { const depth = rnd(), spread = (rnd() + rnd() + rnd() - 1.5) * 1.27;
      return { depth, spread, size: (1.5 + rnd() * 5.5) * (1 - .73 * depth) * k * 1.12 * (SZ + .12), light: rnd(), delay: rnd() * .13 }; });
    const curve = front ? [{ reach: 33, curl: -24, dip: 5 }, { reach: 65, curl: -19, dip: 13 }, { reach: 89, curl: 13, dip: 21 }][i - 8] : { reach: 9, curl: 0, dip: 0 };
    return { u, root, front, length, flowers, curve: { reach: curve.reach * k * SZ, curl: curve.curl * k * SZ, dip: curve.dip * k * SZ }, phase: rnd() * 6.28 };
  });
  return { sz: SZ, kind, pal: WPAL[kind] || WPAL.letters, wis: true, vine, clusters, k, items: clusters.map(c => ({ t: c.u, type: 'raceme', len: c.length, front: c.front })) };
}
function wStem(c, d, time, sw) { const reach = c.curve.reach * (1 - (1 - d) ** 2.5), tip = c.curve.curl * d ** 3, sway = sw * Math.sin(time * 1.77 + c.phase + d * 2) * 8 * c.length / 120 * d ** 2;
  return { x: c.root.x + reach + tip + sway, y: c.root.y + c.length * (c.front ? d ** 1.32 : d) - Math.sin(Math.PI * d) * c.curve.dip }; }
function drawWisteria(x, f, which, time) {
  const S = f.fspec, p = f.grow ?? 0, T = p * 2.15 + (p >= 1 ? 1 : 0), sw = f.swayK || 0, sm = v => { v = clamp(v, 0, 1); return v * v * (3 - 2 * v) };
  if (which === 'b') { const pr = sm(T / .47); x.beginPath(); for (let i = 0; i <= 64; i++) { const q = S.vine(i / 64 * pr); i ? x.lineTo(q.x, q.y) : x.moveTo(q.x, q.y) }
    x.strokeStyle = 'rgba(150,132,175,.55)'; x.lineWidth = 1.1; x.stroke();
  }
  for (const c of S.clusters) { if (c.front !== (which === 'f')) continue; const tc = T - .19 - c.u * .63;
    for (const fl of c.flowers) { const a = sm((tc - fl.depth * .73 - fl.delay) / .24); if (!a) continue;
      const d = fl.depth, ce = wStem(c, d, time, sw), si = wStem(c, Math.min(1, d + .016), time, sw), tg = Math.atan2(si.y - ce.y, si.x - ce.x), sp = fl.spread * (18 * (1 - d) ** .75 + 1) * S.k * S.sz;
      const px = ce.x + Math.sin(tg) * sp, py = ce.y - Math.cos(tg) * sp, L = fl.light;
      const [B, D] = S.pal; x.fillStyle = `rgba(${B[0] + Math.round(D[0] * L)},${B[1] + Math.round(D[1] * L)},${B[2] + Math.round(D[2] * L)},${a * (.62 + .33 * L)})`;
      x.beginPath(); x.ellipse(px, py, fl.size * .82, fl.size, fl.spread * .35, 0, 7); x.fill();
      if (L > .62) { x.fillStyle = `rgba(252,240,255,${a * .55})`; x.beginPath(); x.arc(px - fl.size * .19, py - fl.size * .2, fl.size * .53, 0, 7); x.fill() } } }
}

/* ---------- 绘制各类元素 ---------- */
function drawLeaflet(x, cx, cy, ang, len, col) { x.save(); x.translate(cx, cy); x.rotate(ang); x.fillStyle = col; x.beginPath(); x.ellipse(len/2, 0, len/2, len*.2, 0, 0, 7); x.fill(); x.restore(); }
function drawRaceme(x, ax, ay, len, p, sway, fw) {
  const n = Math.round(len / (fw*.02)); x.save();
  x.strokeStyle = 'rgba(120,140,100,.55)'; x.lineWidth = fw*.006; x.beginPath(); x.moveTo(ax, ay);
  const end = Math.max(1, Math.round(n*p));
  for (let k = 0; k <= end; k++) { const f = k/n; x.lineTo(ax + sway*f*f*len*.3, ay + f*len); } x.stroke();
  for (let k = 0; k < end; k++) {
    const f = k/n, cx = ax + sway*f*f*len*.3, cy = ay + f*len, w = fw*.075*(1 - f*.72), r = fw*.022*(1 - f*.45);
    for (let j = 0; j < 3; j++) {
      const ox = (j - 1)*w*.55 + Math.sin(k*2.3 + j)*w*.15, oy = Math.cos(k*1.7 + j)*r*.4;
      const g = x.createRadialGradient(cx + ox - r*.3, cy + oy - r*.35, r*.1, cx + ox, cy + oy, r*1.1);
      const top = [156, 134, 200], tip = [236, 228, 250], m = Math.min(1, f*1.2);
      const c = top.map((v, i) => Math.round(v + (tip[i] - v)*m));
      g.addColorStop(0, `rgba(255,255,255,.95)`); g.addColorStop(.35, `rgba(${c[0]+20},${c[1]+20},${c[2]+10},.95)`); g.addColorStop(1, `rgba(${c[0]-25},${c[1]-25},${c[2]-15},.92)`);
      x.fillStyle = g; x.beginPath(); x.ellipse(cx + ox, cy + oy, r*1.05, r*.85, (j - 1)*.5, 0, 7); x.fill();
    }
  }
  x.restore();
}
function drawFlower5(x, cx, cy, r, rot, a = 1) {
  x.save(); x.translate(cx, cy); x.rotate(rot); x.globalAlpha = a;
  for (let i = 0; i < 5; i++) { x.save(); x.rotate(i/5*Math.PI*2);
    const g = x.createLinearGradient(0, 0, 0, -r); g.addColorStop(0, '#F3B6CB'); g.addColorStop(1, '#FDEEF3');
    x.fillStyle = g; x.beginPath(); x.moveTo(0, 0); x.bezierCurveTo(r*.55, -r*.35, r*.42, -r*.95, r*.1, -r); x.lineTo(0, -r*.86); x.lineTo(-r*.1, -r); x.bezierCurveTo(-r*.42, -r*.95, -r*.55, -r*.35, 0, 0); x.fill(); x.restore(); }
  x.fillStyle = '#E58BA8'; x.beginPath(); x.arc(0, 0, r*.18, 0, 7); x.fill();
  x.fillStyle = '#F7D48C'; for (let i = 0; i < 6; i++) { const a2 = i*1.05; x.beginPath(); x.arc(Math.cos(a2)*r*.3, Math.sin(a2)*r*.3, r*.05, 0, 7); x.fill(); }
  x.restore();
}
function drawBell(x, cx, cy, r) {
  const g = x.createRadialGradient(cx - r*.3, cy - r*.4, r*.1, cx, cy, r*1.2); g.addColorStop(0, '#FFFFFF'); g.addColorStop(1, '#DDE6D5');
  x.fillStyle = g; x.beginPath(); x.moveTo(cx - r, cy + r*.6); x.quadraticCurveTo(cx - r*1.05, cy - r*.9, cx, cy - r); x.quadraticCurveTo(cx + r*1.05, cy - r*.9, cx + r, cy + r*.6);
  for (let i = 0; i < 6; i++) { const xx = cx + r - (i + .5)*(2*r/6); x.quadraticCurveTo(xx + r/6, cy + r*.95, xx, cy + r*.6); }
  x.closePath(); x.fill(); x.strokeStyle = 'rgba(150,170,140,.35)'; x.lineWidth = r*.08; x.stroke();
}
function drawFlora(f, which, time) {
  const cv = which === 'f' ? f.ff : f.fb, x = cv.getContext('2d'), L = f.flayout, S = f.fspec, fw = L.fw, p = f.grow ?? 0;
  x.setTransform(1, 0, 0, 1, 0, 0); x.clearRect(0, 0, cv.width, cv.height);
  x.setTransform(L.dpr, 0, 0, L.dpr, L.ox*L.dpr, L.oy*L.dpr);
  if (S.wis) { drawWisteria(x, f, which, time); return }
  const pv = Math.min(1, p*1.5), swayAmp = f.swaying ? 1 : 0;
  const sw = (ph, k = 1) => swayAmp*Math.sin(time*1.25 + ph)*.22*k;
  // 藤 / 枝
  if (S.vine && which === 'b') {
    const v = S.vine, n = 48, end = Math.round(n*pv);
    x.lineCap = 'round';
    if (S.kind === 'photos') { x.strokeStyle = '#7C6A86'; x.lineWidth = fw*.026; } else { x.strokeStyle = 'rgba(112,128,96,.75)'; x.lineWidth = fw*.011; }
    x.beginPath(); for (let i = 0; i <= end; i++) { const q = bez(...v, i/n); i ? x.lineTo(q[0], q[1]) : x.moveTo(q[0], q[1]); } x.stroke();
    if (S.kind !== 'photos') { x.strokeStyle = 'rgba(112,128,96,.45)'; x.lineWidth = fw*.005; x.beginPath(); for (let i = 0; i <= end; i++) { const q = bez(...v, i/n); i ? x.lineTo(q[0] + Math.sin(i*.9)*fw*.012, q[1] + Math.cos(i*.7)*fw*.01) : x.moveTo(q[0], q[1]); } x.stroke(); }
  }
  for (const it of S.items) {
    if (!!it.front !== (which === 'f')) continue;
    const ip = clamp((pv - it.t)*3.2, 0, 1); if (ip <= 0) continue;
    if (it.type === 'leaf') { const q = bez(...S.vine, it.t); for (let k = 0; k < 5; k++) drawLeaflet(x, q[0] + k*it.size*.18, q[1] - k*it.size*.12, -1.2*it.side + k*.15 - .4, it.size*.55*ip, `rgba(${118 + k*6},${150 + k*4},${108},.8)`); }
    else if (it.type === 'raceme') { const q = bez(...S.vine, it.t); drawRaceme(x, q[0], q[1], it.len, ip, sw(it.ph, 1 + it.t), fw); }
    else if (it.type === 'blossom') { const q = bez(...S.vine, it.t), R = rngOf(Math.round(it.t*1000));
      for (let k = 0; k < it.n; k++) { const a = R()*6.28, d = R()*it.spread; drawFlower5(x, q[0] + Math.cos(a)*d + sw(it.ph)*fw*.05, q[1] + Math.sin(a)*d*.8, fw*(.045 + R()*.025)*ip, R()*6.28); }
      for (let k = 0; k < 3; k++) { x.fillStyle = 'rgba(244,190,210,.9)'; x.beginPath(); x.arc(q[0] + (R()-.5)*it.spread*2, q[1] + (R()-.5)*it.spread, fw*.012*ip, 0, 7); x.fill(); } }
    else if (it.type === 'twig') { const q = bez(...S.vine, it.t); x.strokeStyle = '#8A7894'; x.lineWidth = fw*.012; x.beginPath(); x.moveTo(q[0], q[1]); const e = [q[0] + Math.cos(it.ang - 1.2)*it.len*ip, q[1] + Math.sin(it.ang - 1.2)*it.len*ip]; x.lineTo(...e); x.stroke(); if (ip > .6) drawFlower5(x, e[0], e[1], fw*.04*(ip - .6)*2.5, it.ph); }
    else if (it.type === 'blade') { x.save(); x.translate(it.x, it.y); x.rotate(it.r + sw(it.ph || 0, .3)); const h = it.h*ip, g = x.createLinearGradient(0, 0, 0, -h); g.addColorStop(0, 'rgba(120,160,120,.85)'); g.addColorStop(1, 'rgba(170,205,160,.85)'); x.fillStyle = g; x.beginPath(); x.moveTo(-it.w*.2, 0); x.quadraticCurveTo(-it.w*.75, -h*.55, 0, -h); x.quadraticCurveTo(it.w*.75, -h*.55, it.w*.2, 0); x.fill(); x.strokeStyle = 'rgba(240,250,235,.5)'; x.lineWidth = fw*.004; x.beginPath(); x.moveTo(0, 0); x.lineTo(0, -h*.95); x.stroke(); x.restore(); }
    else if (it.type === 'lilystem') { const h = it.h*ip, lean = it.lean + sw(it.ph, .4), top = [it.x + Math.sin(lean)*h*.6, it.y - h]; x.strokeStyle = 'rgba(120,155,105,.9)'; x.lineWidth = fw*.009; x.beginPath(); x.moveTo(it.x, it.y); x.quadraticCurveTo(it.x, it.y - h*.8, top[0], top[1]); x.quadraticCurveTo(top[0] + h*.35, top[1] - h*.05, top[0] + h*.45, top[1] + h*.18); x.stroke();
      for (let k = 0; k < it.bells; k++) { const f = k/it.bells; if (f > ip*1.2) break; const bx = top[0] + h*.45*f + Math.sin(f*3)*fw*.01, by = top[1] - h*.04*Math.sin(f*Math.PI) + h*.18*f*f; x.strokeStyle = 'rgba(120,155,105,.8)'; x.lineWidth = fw*.004; x.beginPath(); x.moveTo(bx, by); x.lineTo(bx, by + fw*.025); x.stroke(); drawBell(x, bx, by + fw*.05, fw*(.033 - f*.01)); } }
    else if (it.type === 'spike') { const h = it.h*ip, lean = it.lean + sw(it.ph, .35), tx = it.x + Math.sin(lean)*h, ty = it.y - Math.cos(lean)*h; x.strokeStyle = 'rgba(118,140,104,.9)'; x.lineWidth = fw*.006; x.beginPath(); x.moveTo(it.x, it.y); x.quadraticCurveTo(it.x + Math.sin(lean)*h*.3, it.y - h*.6, tx, ty); x.stroke();
      const nb = 14; for (let k = 0; k < nb; k++) { const f = k/nb; if (f > ip) break; const bx = tx - Math.sin(lean)*h*.45*f, by = ty + Math.cos(lean)*h*.45*f, r = fw*.018*(1 - f*.3);
        for (const sd of [-1, 1]) { const g = x.createRadialGradient(bx + sd*r*.6, by, 0, bx + sd*r*.6, by, r*1.3); g.addColorStop(0, '#C9B8EC'); g.addColorStop(1, '#7E6AB7'); x.fillStyle = g; x.beginPath(); x.ellipse(bx + sd*r*.7, by, r*.8, r*1.15, sd*.5 + lean, 0, 7); x.fill(); } } }
  }
}
function layoutFlora(f) {
  const g = G(), fw = g.fw, ox = .4*fw, oy = 1.05*fw, w = 1.82*fw, h = g.bh + 1.4*fw, dpr = Math.min(2, devicePixelRatio || 1);
  f.flayout = { fw, ox, oy, dpr };
  for (const cv of [f.fb, f.ff]) { cv.width = Math.round(w*dpr); cv.height = Math.round(h*dpr); Object.assign(cv.style, { left: -ox + 'px', top: -oy + 'px', width: w + 'px', height: h + 'px' }); }
  if (!f.fspec) f.fspec = wisteriaSpec(g, [784, 1931, 4127, 9203][folders.indexOf(f) % 4], f.key);
  const t = f.el.querySelector('.ftext'), cfg = FLORA[f.key];
  t.innerHTML = `<span class="fscript">${cfg.script}</span><span class="fsub">${f.name} · ${f.photos.length} ${cfg.unit}</span><svg class="fglyph" viewBox="0 0 20 28"><path d="M10 2v24M10 6c-3 1-4 3-4 5M10 6c3 1 4 3 4 5M10 12c-2.6 1-3.4 2.6-3.4 4.4M10 12c2.6 1 3.4 2.6 3.4 4.4M10 18c-2 .8-2.6 2-2.6 3.4M10 18c2 .8 2.6 2 2.6 3.4"/></svg>`;
  Object.assign(t.style, { left: .085*fw + 'px', top: (g.bh - g.fh*.56) + 'px', width: .83*fw + 'px', fontSize: fw*.15 + 'px' });
  if (f.grow === undefined) f.grow = 1;   // 一打开就是长好的样子，不再等它慢慢长出来
  drawFlora(f, 'b', 0); drawFlora(f, 'f', 0);
}
/* 生长：文件夹第一次被选中时，藤蔓画出、花串依次长出（约 1.6s） */
function growFlora(f) { if (f.growing || f.grow >= 1) return; f.growing = true; f.growStart = performance.now() - f.grow*1600; floraTick() }
let floraRaf = 0;
function floraTick() {
  if (floraRaf) return;
  floraRaf = requestAnimationFrame(function loop(now) {
    floraRaf = 0; let busy = false; const t = now/1000;
    (folders||[]).forEach((f, i) => { if (!f.fb || f.el.classList.contains('off')) return;
      if (f.growing) { f.grow = Math.min(1, (now - f.growStart)/1600); if (f.grow >= 1) f.growing = false; busy = true }
      /* 摆动幅度不再一下开/关，而是慢慢淡入淡出：打开很久再收起，花从静止平滑地开始摆，不会跳一下 */
      const want = mode === 'home' && i === raisedIdx && f.grow >= .6 ? 1 : 0, k0 = f.swayK || 0, dt = Math.min(.05, (now - (f._swT || now))/1000); f._swT = now;
      f.swayP = Math.max(0, Math.min(1, (f.swayP || 0) + (want ? 1 : -1)*dt/1.4)); { const q = f.swayP; f.swayK = q*q*(3 - 2*q) }   // 1.4 秒 smoothstep 淡入，起步速度为 0
      const moving = f.swayK > 0 || k0 !== f.swayK; if (moving || want) busy = true;
      if (f.growing || moving) { f.swaying = f.swayK > 0; drawFlora(f, 'b', t); drawFlora(f, 'f', t) } });
    if (busy && mode !== 'detail') floraRaf = requestAnimationFrame(loop);
  });
}
setInterval(() => { if (mode === 'home' && !floraRaf) floraTick() }, 400);

/* ---------- 花瓣粒子 ---------- */
const petalCv = document.createElement('canvas'); petalCv.id = 'petals'; let petals = [], petalRaf = 0, petalLast = 0;
function petalLayout() { const dpr = Math.min(2, devicePixelRatio || 1); petalCv.width = W*dpr; petalCv.height = H*dpr; petalCv.style.width = W + 'px'; petalCv.style.height = H + 'px'; petalCv._dpr = dpr }
function foldToScreen(f, lx, ly) { const r = f.el.getBoundingClientRect(), pr = phone.getBoundingClientRect(), k = r.width/G().fw; return [r.left - pr.left + lx*k, r.top - pr.top + ly*k] }
function emitPetal(kind, x, y, vx, vy, big) {
  if (petals.length > 260) petals.shift();
  const pal = FLORA[kind].petal, s = (big ? 1.25 : 1)*(3.2 + Math.random()*3.4);
  petals.push({ x, y, vx, vy, rot: Math.random()*6.28, vr: (Math.random() - .5)*5, s, col: pal[(Math.random()*pal.length)|0], kind, ph: Math.random()*6.28, life: 0 });
  if (!petalRaf) { petalLast = performance.now(); petalRaf = requestAnimationFrame(petalStep) }
}
function petalBurst(f, lx, ly, n, power = 1) {
  const [x, y] = foldToScreen(f, lx, ly);
  for (let i = 0; i < n; i++) { const a = -Math.PI/2 + (Math.random() - .5)*2.4, sp = (140 + Math.random()*260)*power;
    emitPetal(f.key, x + (Math.random() - .5)*40, y + (Math.random() - .5)*30, Math.cos(a)*sp, Math.sin(a)*sp, true) }
}
function petalStep(now) {
  const dt = Math.min(.033, (now - petalLast)/1000); petalLast = now; const x = petalCv.getContext('2d'), d = petalCv._dpr || 1;
  x.setTransform(d, 0, 0, d, 0, 0); x.clearRect(0, 0, W, H);
  petals = petals.filter(p => p.y < H + 30 && p.x > -40 && p.x < W + 40 && p.life < 14);
  for (const p of petals) {
    p.life += dt; p.vy += 55*dt; p.vx *= Math.pow(.35, dt); p.vy *= Math.pow(.55, dt);
    const fall = Math.min(p.vy, 38 + p.s*3); p.vy = Math.max(p.vy, -400); if (p.vy > fall) p.vy = fall;
    p.x += (p.vx + Math.sin(p.life*2.2 + p.ph)*22)*dt; p.y += p.vy*dt; p.rot += p.vr*dt;
    x.save(); x.translate(p.x, p.y); x.rotate(p.rot); x.scale(1, .55 + .45*Math.abs(Math.sin(p.life*3 + p.ph)));
    x.globalAlpha = Math.min(1, p.life*4)*Math.min(1, (14 - p.life)/2)*.95;
    const gr = x.createLinearGradient(-p.s, -p.s*.55, p.s*.8, p.s*.65); gr.addColorStop(0, p.col); gr.addColorStop(.67, '#F3EAFB'); gr.addColorStop(1, '#FDF8FF'); x.fillStyle = gr;
    const S2 = p.s; x.beginPath(); { x.moveTo(-S2, -.05*S2); x.bezierCurveTo(-.56*S2, -.32*S2, -.03*S2, -.96*S2, .55*S2, -.72*S2); x.bezierCurveTo(1.26*S2, -.44*S2, 1.08*S2, .33*S2, .58*S2, .64*S2); x.bezierCurveTo(-.02*S2, .97*S2, -.71*S2, .44*S2, -S2, -.05*S2); }
    x.fill(); x.restore();
  }
  petalRaf = petals.length ? requestAnimationFrame(petalStep) : 0;
}
/* 平时：当前文件夹的花偶尔飘落一两片 */
setInterval(() => {
  if (mode !== 'home' || !folders) return; const f = folders[raisedIdx]; if (!f || !f.fspec || f.grow < 1) return;
  const S = f.fspec, cand = S.items.filter(i => i.type === 'raceme' || i.type === 'blossom' || i.type === 'lilystem' || i.type === 'spike');
  const it = cand[(Math.random()*cand.length)|0] || {}, fw = G().fw; let lx, ly;
  if (S.wis) { const c = S.clusters[2 + ((Math.random() * 9) | 0)], q = wStem(c, Math.random() * .75, performance.now() / 1000, 1); lx = q.x; ly = q.y }
  else if (S.vine && it.t !== undefined) { const q = bez(...S.vine, it.t); lx = q[0]; ly = q[1] + (it.len || fw*.1)*.7 } else { lx = it.x; ly = it.y - it.h*.85 }
  const [x, y] = foldToScreen(f, lx, ly); emitPetal(f.key, x, y, (Math.random() - .5)*30, 10 + Math.random()*20);
}, 650);
