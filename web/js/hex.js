/* Nền công nghệ cao, hiệu ứng mạnh:
   - sóng năng lượng lan từ lõi tổ ong ra ngoài định kỳ, thắp sáng cả viền các ô tối;
   - ô sáng "thở" lệch pha, thỉnh thoảng một ô bùng sáng (flare);
   - bấm chuột bất kỳ đâu: gợn sóng lan ra từ điểm bấm; rê chuột: các ô quanh con trỏ sáng lên;
   - đường mạch cam có xung sáng chạy kèm đuôi mờ và nút phát sáng ở đầu đường;
   - tia lửa (ember) bay lên từ cụm tổ ong; dải quét sáng chậm chạy ngang nền.
   Lớp tĩnh vẽ một lần vào canvas phụ; chỉ phần động vẽ mỗi khung (~30 fps).
   Tắt chuyển động nếu bật "giảm chuyển động" của hệ điều hành; tự dừng khi tab bị ẩn. */
(() => {
  const cv = document.getElementById('hexbg');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const OR = '255,122,26', OR2 = '255,168,64', HOT = '255,214,150';
  const SQ = Math.sqrt(3);
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let seed = 7;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

  let W = 0, H = 0, dpr = 1, R = 40, base = null, cells = [], lit = [], lines = [], embers = [], ox = 0, oy = 0, reach = 600;
  const ripples = [];                                  // gợn sóng do bấm chuột: {x, y, t0}
  const flares = [];                                   // ô bùng sáng: {h, t0}
  const mouse = { x: -999, y: -999 };

  function hexPath(c, x, y, r) {
    c.beginPath();
    for (let i = 0; i < 6; i++) {
      const a = Math.PI / 3 * i, px = x + r * Math.cos(a), py = y + r * Math.sin(a);
      i ? c.lineTo(px, py) : c.moveTo(px, py);
    }
    c.closePath();
  }

  function build() {
    dpr = Math.min(devicePixelRatio || 1, 2);
    W = innerWidth; H = innerHeight;
    cv.width = W * dpr; cv.height = H * dpr;
    cv.style.width = W + 'px'; cv.style.height = H + 'px';
    seed = 7; cells = []; lit = []; lines = []; embers = [];
    R = Math.max(34, Math.min(58, W / 26));
    ox = W * 0.97; oy = H * 0.52; reach = Math.min(W * 0.5, 820);

    base = document.createElement('canvas'); base.width = W * dpr; base.height = H * dpr;
    const b = base.getContext('2d'); b.setTransform(dpr, 0, 0, dpr, 0, 0);
    const bg = b.createLinearGradient(0, 0, W, H);
    bg.addColorStop(0, '#070A0D'); bg.addColorStop(1, '#10151A');
    b.fillStyle = bg; b.fillRect(0, 0, W, H);
    const glow = b.createRadialGradient(ox, oy, 0, ox, oy, reach * 1.1);   // hào quang cam nền phía sau cụm
    glow.addColorStop(0, 'rgba(255,110,10,.20)'); glow.addColorStop(0.5, 'rgba(255,110,10,.06)'); glow.addColorStop(1, 'rgba(255,110,10,0)');
    b.fillStyle = glow; b.fillRect(0, 0, W, H);

    const s = 11;                                   // lưới tổ ong nhỏ mờ phủ toàn nền
    b.strokeStyle = 'rgba(255,255,255,.04)'; b.lineWidth = 1;
    for (let c = 0, x = 0; x < W + s * 2; c++, x += s * 1.5)
      for (let y = (c % 2) * (SQ * s / 2); y < H + s * 2; y += SQ * s) { hexPath(b, x, y, s - 1); b.stroke(); }

    for (let c = -2, x = W - R * 12; x < W + R * 2; c++, x += R * 1.5) {
      for (let y = ((c % 2 + 2) % 2) * (SQ * R / 2) - SQ * R; y < H + SQ * R; y += SQ * R) {
        const d = Math.hypot(x - ox, (y - oy) * 1.15), k = 1 - d / reach;
        if (k <= 0) continue;
        hexPath(b, x, y, R - 3);
        b.fillStyle = `rgba(13,17,21,${0.6 + 0.32 * k})`; b.fill();
        b.strokeStyle = `rgba(${OR},${0.12 + 0.24 * k})`; b.lineWidth = 1.2; b.stroke();
        const cell = { x, y, k, d, ph: rnd() * 6.283, sp: 0.6 + rnd() * 1.2, lit: rnd() < 0.2 + k * 0.62 };
        cells.push(cell); if (cell.lit) lit.push(cell);
      }
    }

    const defs = [[0.20, 0.00, 0.62, 0.40], [0.34, 0.00, 0.50, 0.60], [0.74, 0.00, 0.58, 0.32], [0.47, 0.00, 0.40, 0.22], [0.88, 0.00, 0.66, 0.26]];
    for (const [fy, sx, ex, al] of defs) {
      const y = H * fy, x0 = W * sx, bend = W * (ex - 0.1), dy = (fy < 0.5 ? 1 : -1) * 36;
      const pts = [[x0, y], [bend, y], [bend + Math.abs(dy), y + dy], [W * ex, y + dy]];
      let len = 0; for (let i = 1; i < pts.length; i++) len += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
      lines.push({ pts, al, len, sp: 90 + rnd() * 90, off: rnd() * 500, ph: rnd() * 6 });
    }
    for (let i = 0; i < 46; i++) embers.push(newEmber(true));
  }

  function newEmber(init) {
    const x = W - rnd() * Math.min(W * 0.42, 620), y = init ? rnd() * H : H + 10;
    return { x, y, vy: 14 + rnd() * 34, vx: (rnd() - 0.5) * 10, r: 0.8 + rnd() * 1.8, ph: rnd() * 6.28, sp: 1 + rnd() * 3 };
  }

  function linePath(pts) {
    ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  }

  let lastFlare = 0, prevT = 0;
  function frame(t) {
    const sec = t / 1000, dt = Math.min(0.1, (t - prevT) / 1000 || 0.033); prevT = t;
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalCompositeOperation = 'source-over';
    ctx.drawImage(base, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // dải quét sáng chậm chạy ngang nền
    const sx = ((sec * 90) % (W + 600)) - 300, sg = ctx.createLinearGradient(sx - 160, 0, sx + 160, 0);
    sg.addColorStop(0, 'rgba(255,140,40,0)'); sg.addColorStop(0.5, 'rgba(255,140,40,.045)'); sg.addColorStop(1, 'rgba(255,140,40,0)');
    ctx.fillStyle = sg; ctx.fillRect(sx - 160, 0, 320, H);

    // nguồn sóng: lõi (định kỳ) + gợn do bấm chuột
    const period = reach * 1.7, ring = (sec * 300) % period;
    for (let i = ripples.length - 1; i >= 0; i--) if (sec - ripples[i].t0 > 2.2) ripples.splice(i, 1);

    // đường mạch + xung sáng có đuôi
    ctx.lineCap = 'round';
    for (const L of lines) {
      linePath(L.pts); ctx.setLineDash([]);
      ctx.strokeStyle = `rgba(${OR},${L.al * 0.5})`; ctx.lineWidth = 1.1; ctx.stroke();
      const head = (sec * L.sp + L.off) % (L.len + 160);
      for (const [dash, a, w] of [[120, 0.22, 3], [60, 0.55, 2.2], [22, 1, 1.8]]) {
        linePath(L.pts); ctx.setLineDash([dash, L.len + 200]); ctx.lineDashOffset = -(head - dash);
        ctx.shadowColor = `rgba(${OR},1)`; ctx.shadowBlur = 14;
        ctx.strokeStyle = `rgba(${HOT},${Math.min(1, a * (0.5 + L.al))})`; ctx.lineWidth = w; ctx.stroke();
      }
      ctx.setLineDash([]); ctx.shadowBlur = 0;
      const end = L.pts[3], np = 0.5 + 0.5 * Math.sin(sec * 3 + L.ph);   // nút phát sáng cuối đường
      ctx.beginPath(); ctx.arc(end[0], end[1], 3 + np * 2, 0, 6.283);
      ctx.shadowColor = `rgba(${OR},1)`; ctx.shadowBlur = 16; ctx.fillStyle = `rgba(${OR2},${0.5 + 0.5 * np})`; ctx.fill(); ctx.shadowBlur = 0;
    }

    // sóng lõi / gợn chuột thắp sáng viền mọi ô (kể cả ô tối)
    for (const c of cells) {
      let w = Math.exp(-Math.pow(c.d - ring, 2) / (2 * 62 * 62)) * (0.4 + 0.6 * c.k);
      for (const r of ripples) {
        const rd = Math.hypot(c.x - r.x, c.y - r.y), rr = (sec - r.t0) * 520;
        w = Math.max(w, Math.exp(-Math.pow(rd - rr, 2) / (2 * 48 * 48)) * Math.max(0, 1 - (sec - r.t0) / 2.2));
      }
      c.w = w;
      if (w > 0.06 && !c.lit) {
        ctx.strokeStyle = `rgba(${OR2},${Math.min(1, w * 1.1)})`; ctx.lineWidth = 1.6;
        ctx.shadowColor = `rgba(${OR},1)`; ctx.shadowBlur = 14 * w; hexPath(ctx, c.x, c.y, R - 3); ctx.stroke(); ctx.shadowBlur = 0;
      }
    }

    // thỉnh thoảng một ô bùng sáng
    if (!reduce && lit.length && sec - lastFlare > 0.55) { flares.push({ h: lit[(rnd() * lit.length) | 0], t0: sec }); lastFlare = sec; }
    for (let i = flares.length - 1; i >= 0; i--) if (sec - flares[i].t0 > 1.1) flares.splice(i, 1);

    // ô sáng: thở + chuột + sóng + flare
    for (const h of lit) {
      const pulse = reduce ? 0.7 : 0.5 + 0.5 * Math.sin(sec * h.sp + h.ph);
      const near = Math.max(0, 1 - Math.hypot(h.x - mouse.x, h.y - mouse.y) / (R * 4.2));
      let fl = 0; for (const f of flares) if (f.h === h) fl = Math.max(fl, 1 - (sec - f.t0) / 1.1);
      const e = Math.min(1, (0.35 + 0.5 * pulse) * (0.55 + 0.45 * h.k) + near * 0.7 + h.w * 0.9 + fl);
      const r = R - 3 + near * 5 + h.w * 3 + fl * 2;
      ctx.globalCompositeOperation = 'lighter';
      const g = ctx.createRadialGradient(h.x, h.y, 2, h.x, h.y, R);
      g.addColorStop(0, `rgba(${fl > 0.2 ? HOT : OR2},${0.10 + 0.42 * e})`); g.addColorStop(1, `rgba(${OR},${0.02 + 0.12 * e})`);
      hexPath(ctx, h.x, h.y, r); ctx.fillStyle = g; ctx.fill();
      ctx.globalCompositeOperation = 'source-over';
      ctx.shadowColor = `rgba(${OR},1)`; ctx.shadowBlur = 8 + 26 * e;
      ctx.strokeStyle = `rgba(${fl > 0.2 ? HOT : OR2},${0.4 + 0.6 * e})`; ctx.lineWidth = 1.5 + near + fl; hexPath(ctx, h.x, h.y, r); ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = `rgba(${OR},${0.22 + 0.62 * e})`; ctx.lineWidth = 1; hexPath(ctx, h.x, h.y, R * 0.66 + near * 3 + h.w * 3); ctx.stroke();
    }

    // tia lửa bay lên
    ctx.globalCompositeOperation = 'lighter';
    for (const m of embers) {
      if (!reduce) { m.y -= m.vy * dt; m.x += m.vx * dt + Math.sin(sec * 1.3 + m.ph) * 0.15; if (m.y < -10) Object.assign(m, newEmber(false)); }
      const a = 0.25 + 0.55 * (0.5 + 0.5 * Math.sin(sec * m.sp + m.ph)), fade = Math.min(1, m.y / (H * 0.25), (H - m.y + 20) / 80);
      ctx.beginPath(); ctx.arc(m.x, m.y, m.r, 0, 6.283);
      ctx.fillStyle = `rgba(${OR2},${a * Math.max(0, fade)})`; ctx.shadowColor = `rgba(${OR},1)`; ctx.shadowBlur = 8; ctx.fill();
    }
    ctx.shadowBlur = 0; ctx.globalCompositeOperation = 'source-over';

    const v = ctx.createLinearGradient(0, 0, W, 0);  // vignette để chữ luôn dễ đọc
    v.addColorStop(0, 'rgba(7,10,13,.55)'); v.addColorStop(0.6, 'rgba(7,10,13,0)');
    ctx.fillStyle = v; ctx.fillRect(0, 0, W, H);
  }

  let last = 0;
  function loop(now) {
    if (now - last >= 32) { last = now; frame(now); }   // ~30 fps
    requestAnimationFrame(loop);
  }

  addEventListener('pointermove', e => { mouse.x = e.clientX; mouse.y = e.clientY; }, { passive: true });
  addEventListener('pointerleave', () => { mouse.x = mouse.y = -999; });
  addEventListener('pointerdown', e => { if (ripples.length < 6) ripples.push({ x: e.clientX, y: e.clientY, t0: performance.now() / 1000 }); }, { passive: true });
  let rt;
  addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { build(); if (reduce) frame(0); }, 120); });
  build();
  if (reduce) frame(0); else requestAnimationFrame(loop);
})();
