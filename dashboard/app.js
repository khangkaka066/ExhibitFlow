'use strict';

// Synthetic floor-space observations. Never presented as tracker output.
const zones = [
  {id:'A',name:'Sảnh đón',subtitle:'Entrance',color:'#b4c0ab',rect:[65,275,210,120]},
  {id:'B',name:'Di sản',subtitle:'Heritage gallery',color:'#87a983',rect:[65,55,210,205]},
  {id:'C',name:'Nghệ thuật',subtitle:'Art gallery',color:'#4f8068',rect:[290,55,230,205]},
  {id:'D',name:'Tương tác',subtitle:'Interactive space',color:'#9cb179',rect:[535,55,160,205]},
  {id:'E',name:'Góc nghỉ',subtitle:'Rest area',color:'#c8b98a',rect:[440,275,255,120]},
];
let visitors = [];
let sessionEnd = 120;
const DATA_URL = 'data/mock_tracks.csv';

function zoneForPoint(x, y) {
  return zones.find(zone => {
    const [left, top, width, height] = zone.rect;
    return x >= left && x <= left + width && y >= top && y <= top + height;
  })?.id || null;
}
function parseCsv(text) {
  const [header, ...lines] = text.trim().split(/\r?\n/);
  const fields = header.split(',');
  return lines.filter(Boolean).map(line => Object.fromEntries(line.split(',').map((value, index) => [fields[index], Number(value)])));
}
function makeVisitors(rows) {
  const groups = new Map();
  rows.forEach(row => {
    const x = 55 + row.x / 1280 * 650, y = 45 + row.y / 720 * 360;
    const observation = {time: row.timestamp, x, y, zone: zoneForPoint(x, y)};
    if (!groups.has(row.track_id)) groups.set(row.track_id, []);
    groups.get(row.track_id).push(observation);
  });
  return [...groups].map(([id, observations]) => {
    observations.sort((a, b) => a.time - b.time);
    const visits = [];
    let current = null;
    observations.forEach(observation => {
      if (observation.zone !== current?.zone) {
        if (current && current.zone) visits.push({zone: current.zone, start: current.start, end: observation.time});
        current = {zone: observation.zone, start: observation.time};
      }
    });
    if (current?.zone) visits.push({zone: current.zone, start: current.start, end: observations.at(-1).time + 0.1});
    return {id, visits, observations};
  }).filter(visitor => visitor.visits.length);
}
async function loadMockTracks() {
  try {
    const response = await fetch(DATA_URL);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const rows = parseCsv(await response.text());
    visitors = makeVisitors(rows);
    sessionEnd = Math.max(1, Math.ceil(Math.max(...rows.map(row => row.timestamp))));
    $('session').innerHTML = '<option>Mock tracks CSV · cam_01</option>';
    document.querySelector('.demo-notice p').innerHTML = '<strong>Dữ liệu mock từ CSV</strong> · Dashboard đọc <code>dashboard/data/mock_tracks.csv</code>; zone được ánh xạ minh họa từ tọa độ ảnh, không phải kết quả calibration CAVIAR.';
    render();
  } catch (error) {
    document.querySelector('.demo-notice p').textContent = `Không tải được mock CSV (${error.message}). Hãy chạy python3 -m http.server 8080 từ repo root.`;
  }
}
const $ = (id)=>document.getElementById(id);
const zoneById = id=>zones.find(z=>z.id===id);
const fmt = seconds=>`${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(Math.floor(seconds%60)).padStart(2,'0')}`;
const state = {window:'all',zone:'all',layer:'heat',visitor:1,time:0};
let timer = null;
const bounds = ()=>state.window==='first'?[0,sessionEnd/2]:state.window==='second'?[sessionEnd/2,sessionEnd]:[0,sessionEnd];
const clipped = visitor=>{
  const [lo,hi]=bounds();
  return visitor.visits.map(v=>({...v,start:Math.max(v.start,lo),end:Math.min(v.end,hi)})).filter(v=>v.end>v.start);
};
const included = ()=>visitors.filter(v=>clipped(v).some(s=>state.zone==='all'||s.zone===state.zone));
function statsFor(id){
  const visits=visitors.flatMap(v=>clipped(v).filter(s=>s.zone===id).map(s=>({...s,id:v.id})));
  const seconds=visits.reduce((sum,v)=>sum+v.end-v.start,0);
  return {count:new Set(visits.map(v=>v.id)).size,visits:visits.length,seconds,mean:visits.length?seconds/visits.length:0};
}
function transitions(){
  const [lo,hi]=bounds(),counts={};
  visitors.forEach(v=>v.visits.slice(1).forEach((s,i)=>{
    const from=v.visits[i].zone;
    if(s.start>=lo&&s.start<hi&&from!==s.zone&&(state.zone==='all'||from===state.zone||s.zone===state.zone)){
      const key=from+s.zone;counts[key]=(counts[key]||0)+1;
    }
  }));
  return counts;
}
function badge(stats){
  if(stats.count===0)return '<span class="badge normal">Chưa có dữ liệu</span>';
  if(stats.count<=2)return '<span class="badge quiet">Ít ghé</span>';
  if(stats.mean>=30)return '<span class="badge hot">● Nóng</span>';
  return '<span class="badge normal">Bình thường</span>';
}
function renderMetrics(){
  const selected=zones.filter(z=>state.zone==='all'||z.id===state.zone);
  const stats=selected.map(z=>statsFor(z.id));
  const seconds=stats.reduce((n,s)=>n+s.seconds,0),visits=stats.reduce((n,s)=>n+s.visits,0);
  const popular=selected.slice().sort((a,b)=>statsFor(b.id).seconds-statsFor(a.id).seconds)[0];
  const countTransitions=Object.values(transitions()).reduce((a,b)=>a+b,0);
  const items=[['Track quan sát',included().length,'ID','ID duy nhất trong phiên / camera','♧'],['Dừng trung bình',visits?Math.round(seconds/visits):0,'giây','Trên mỗi lượt ghé zone trong bộ lọc','◷'],['Tổng chuyển tiếp',countTransitions,'lần','Các lần chuyển zone có hướng','⇄'],['Dừng nhiều nhất',popular?popular.id:'—',popular?popular.name:'','Theo tổng thời gian quan sát tại zone','⌁']];
  $('metrics').innerHTML=items.map(([name,value,unit,note,icon])=>`<article class="metric"><div class="metric-label">${name}<span class="metric-icon">${icon}</span></div><div class="metric-value">${value}<small>${unit}</small></div><div class="metric-note">${note}</div></article>`).join('');
  $('zone-rows').innerHTML=selected.map(z=>{const s=statsFor(z.id);return `<tr><td><button class="zone-button" data-zone="${z.id}" aria-label="Chọn zone ${z.name}"><span class="zone-dot" style="background:${z.color}"></span><span>${z.id} · ${z.name}<small>${z.subtitle}</small></span></button></td><td>${s.count}</td><td>${s.visits?Math.round(s.mean)+' giây':'—'}</td><td>${badge(s)}</td></tr>`;}).join('');
  const focus=state.zone==='all'?popular:zoneById(state.zone),s=statsFor(focus.id);
  $('spotlight-name').textContent=`${focus.id} · ${focus.name}`;
  $('spotlight').innerHTML=`<div class="spotlight-total">${s.count}<small>ID ghé</small></div><p class="spotlight-subtitle">${state.zone==='all'?'Zone có tổng thời gian dừng cao nhất':'Khu vực đang được chọn'}</p><div class="detail-line"><span>Dừng TB / lượt</span><strong>${Math.round(s.mean)} giây</strong></div><div class="detail-line"><span>Tổng thời gian</span><strong>${s.seconds} giây</strong></div><div class="detail-line"><span>Mức quan tâm</span>${badge(s)}</div>`;
  $('insight-text').textContent=`${focus.name} có ${s.visits} lượt ghé trong khoảng đang chọn. So sánh số lượt và thời gian dừng để tìm zone thu hút hoặc ít được ghé.`;
}
function renderTransitions(){
  const counts=transitions(),entries=Object.entries(counts).sort((a,b)=>b[1]-a[1]),max=Math.max(1,...Object.values(counts));
  $('transition-bars').innerHTML=entries.length?entries.slice(0,3).map(([key,n])=>`<div class="transition-row"><span class="route">${key[0]} → ${key[1]} · ${zoneById(key[1]).name}</span><div class="bar-track"><div class="bar-fill" style="width:${100*n/max}%"></div></div><strong>${n} lần</strong></div>`).join(''):'<p class="empty">Chưa có chuyển tiếp trong bộ lọc này.</p>';
  $('transition-matrix').innerHTML=`<table class="matrix"><caption class="sr-only">Số lần chuyển tiếp có hướng giữa các zone</caption><thead><tr><th scope="col">Từ / Đến</th>${zones.map(z=>`<th scope="col" title="${z.name}">${z.id}</th>`).join('')}</tr></thead><tbody>${zones.map(a=>`<tr><th scope="row" title="${a.name}">${a.id}</th>${zones.map(b=>{const n=counts[a.id+b.id]||0;return `<td title="${a.name} → ${b.name}: ${n} lần" style="background:${n?`rgba(100,145,91,${.12+.5*n/max})`:'#f6f8f2'}">${a.id===b.id?'—':n}</td>`;}).join('')}</tr>`).join('')}</tbody></table>`;
}
// One synthetic observation per second, always inside a defined zone.
function pointAt(visitor,t){
  const observation = visitor.observations.reduce((best, candidate) =>
    !best || Math.abs(candidate.time - t) < Math.abs(best.time - t) ? candidate : best, null);
  return observation && Math.abs(observation.time - t) <= .55 && observation.zone ? observation : null;
}
function renderMap(){
  const [lo,hi]=bounds();
  let svg=`<defs><pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r=".6" fill="#dce3d5"/></pattern><radialGradient id="heat"><stop stop-color="#397348" stop-opacity=".48"/><stop offset=".45" stop-color="#86ab63" stop-opacity=".34"/><stop offset="1" stop-color="#dce49e" stop-opacity="0"/></radialGradient></defs><rect width="760" height="450" fill="url(#grid)"/><path d="M55 45 H705 V405 H55 Z" fill="#fff" stroke="#b8c3b0" stroke-width="4"/><text x="345" y="360" fill="#9fa994" font-size="10" letter-spacing="2">LỐI ĐI</text><path d="M312 380v-36m-5 5 5-5 5 5" stroke="#9eae94" fill="none"/><text x="115" y="425" fill="#819275" font-size="9" letter-spacing="2">↑ LỐI VÀO / RA</text>`;
  const maxSeconds=Math.max(1,...zones.map(z=>statsFor(z.id).seconds));
  zones.forEach(z=>{
    const [x,y,w,h]=z.rect,dim=state.zone!=='all'&&state.zone!==z.id;
    svg+=`<g class="zone-shape" data-zone="${z.id}" role="button" tabindex="0" aria-label="Chọn zone ${z.id}: ${z.name}" aria-pressed="${state.zone===z.id}" opacity="${dim?.38:1}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3" fill="#f1f3eb" stroke="${state.zone===z.id?'#216953':'#cbd4c0'}" stroke-width="${state.zone===z.id?3:1.5}"/>`;
    if(state.layer==='heat'){
      // Per-zone grid occupancy accumulated as observed person-seconds.
      const cells={};
      visitors.forEach(v=>{for(let t=lo;t<hi;t++){const p=pointAt(v,t);if(p&&p.zone===z.id){const key=`${Math.floor((p.x-x)/35)},${Math.floor((p.y-y)/35)}`;cells[key]=(cells[key]||0)+1;}}});
      const max=Math.max(1,...Object.values(cells));
      svg+=`<svg x="${x}" y="${y}" width="${w}" height="${h}" overflow="hidden">${Object.entries(cells).map(([key,n])=>{const [cx,cy]=key.split(',').map(Number);return `<ellipse cx="${cx*35+17}" cy="${cy*35+17}" rx="57" ry="48" fill="url(#heat)" opacity="${(.3+.7*n/max)*(statsFor(z.id).seconds/maxSeconds)}"/>`;}).join('')}</svg>`;
    }
    // Stylized exhibit plinths, not a measured CAVIAR floor plan.
    svg+=`<rect x="${x+w*.16}" y="${y+20}" width="${w*.3}" height="13" rx="2" fill="#e2e6d9" stroke="#d2d9c8"/><rect x="${x+w*.64}" y="${y+20}" width="${w*.18}" height="13" rx="2" fill="#e2e6d9" stroke="#d2d9c8"/><text x="${x+15}" y="${y+h-25}" fill="#4d6749" font-size="12" font-weight="600">${z.id} · ${z.name}</text><text x="${x+15}" y="${y+h-10}" fill="#94a18a" font-size="8">${z.subtitle}</text></g>`;
  });
  if(state.layer==='journey'){
    const v=visitors.find(v=>v.id===state.visitor),segments=[];
    if(v){
      let previous=null,points=[];
      for(let t=lo;t<=Math.min(state.time,hi-1);t++){
        const p=pointAt(v,t);
        if(!p){if(points.length)segments.push(points);points=[];previous=null;continue;}
        if(previous&&previous.zone!==p.zone){if(points.length)segments.push(points);points=[];svg+=`<path d="M${previous.x},${previous.y} L${p.x},${p.y}" stroke="#547b55" stroke-width="2" stroke-dasharray="4 5" opacity=".6" fill="none"/>`;}
        points.push(`${p.x},${p.y}`);previous=p;
      }
      if(points.length)segments.push(points);
      svg+=segments.map(points=>`<polyline points="${points.join(' ')}" stroke="#216953" stroke-width="3" fill="none" stroke-linecap="round"/>`).join('');
      const p=state.time<hi?pointAt(v,state.time):null;
      if(p)svg+=`<circle cx="${p.x}" cy="${p.y}" r="11" fill="#216953" opacity=".14"/><circle cx="${p.x}" cy="${p.y}" r="5" fill="#216953" stroke="#fff" stroke-width="2"/><rect x="${p.x+10}" y="${p.y-24}" width="68" height="20" rx="4" fill="#216953"/><text x="${p.x+17}" y="${p.y-10}" fill="#fff" font-size="9">Track ${String(v.id).padStart(3,'0')}</text>`;
    }
  }
  $('map').innerHTML=svg;
  $('layer-description').textContent=state.layer==='heat'?'Mật độ thời gian quan sát · dữ liệu mẫu':state.layer==='journey'?'Nét đứt = nối giữa zone, chưa xác thực lối đi':'5 khu vực · Chọn bằng chuột hoặc bàn phím';
  document.querySelector('.heat-legend').style.visibility=state.layer==='heat'?'visible':'hidden';
  document.querySelectorAll('[data-layer]').forEach(b=>{const active=b.dataset.layer===state.layer;b.classList.toggle('selected',active);b.setAttribute('aria-pressed',String(active));});
}
function renderJourney(){
  const available=included();
  if(!available.some(v=>v.id===state.visitor))state.visitor=available[0]?.id??null;
  $('visitor').innerHTML=available.map(v=>`<option value="${v.id}">Track ${String(v.id).padStart(3,'0')} · cam_01</option>`).join('');
  $('visitor').value=String(state.visitor);
  const visitor=available.find(v=>v.id===state.visitor),stops=visitor?clipped(visitor):[];
  $('journey-summary').textContent=visitor?`${new Set(stops.map(s=>s.zone)).size} zone · ${stops.reduce((n,s)=>n+s.end-s.start,0)} giây quan sát. Giữ toàn bộ hành trình trong khoảng thời gian đã chọn.`:'Không có hành trình trong bộ lọc này.';
  $('journey-stops').innerHTML=stops.length?stops.map(s=>`<div class="stop"><span class="stop-dot"></span><strong>${s.zone} · ${zoneById(s.zone).name}</strong><small>${fmt(s.start)}–${fmt(s.end)} · ${s.end-s.start}s</small></div>`).join(''):'<p class="empty">Không có dữ liệu.</p>';
  $('play').disabled=$('show-journey').disabled=!visitor;
}
function syncTime(){
  const [lo,hi]=bounds();state.time=Math.max(lo,Math.min(hi,state.time));
  $('scrubber').min=lo;$('scrubber').max=hi;$('scrubber').value=state.time;
  $('playback-time').textContent=`${fmt(state.time)} / ${fmt(hi)}`;
}
function stop(){clearInterval(timer);timer=null;$('play').textContent='▶';$('play').setAttribute('aria-label','Phát hành trình');}
function render(){renderMetrics();renderTransitions();renderJourney();syncTime();renderMap();}
function setZone(id){state.zone=id;$('zone-filter').value=id;stop();render();}
$('zone-filter').insertAdjacentHTML('beforeend',zones.map(z=>`<option value="${z.id}">${z.id} · ${z.name}</option>`).join(''));
$('zone-filter').addEventListener('change',e=>setZone(e.target.value));
$('time-window').addEventListener('change',e=>{stop();state.window=e.target.value;state.time=bounds()[0];render();});
document.addEventListener('click',e=>{
  const z=e.target.closest('[data-zone]');if(z)setZone(z.dataset.zone);
  const layer=e.target.closest('[data-layer]');if(layer){state.layer=layer.dataset.layer;if(state.layer!=='journey')stop();renderMap();}
});
$('map').addEventListener('keydown',e=>{const z=e.target.closest('[data-zone]');if(z&&['Enter',' '].includes(e.key)){e.preventDefault();setZone(z.dataset.zone);$('map').querySelector(`[data-zone="${z.dataset.zone}"]`).focus();}});
$('visitor').addEventListener('change',e=>{stop();state.visitor=Number(e.target.value);state.time=bounds()[0];render();});
$('scrubber').addEventListener('input',e=>{stop();state.time=Number(e.target.value);state.layer='journey';syncTime();renderMap();});
$('play').addEventListener('click',()=>{
  if(timer){stop();return;}state.layer='journey';const [lo,hi]=bounds();if(state.time>=hi)state.time=lo;
  $('play').textContent='Ⅱ';$('play').setAttribute('aria-label','Tạm dừng hành trình');renderMap();
  timer=setInterval(()=>{state.time=Math.min(hi,state.time+1);syncTime();renderMap();if(state.time>=hi)stop();},1000);
});
$('show-journey').addEventListener('click',()=>{stop();state.layer='journey';state.time=bounds()[1];syncTime();renderMap();$('floor-map').scrollIntoView({behavior:'smooth'});});
$('reset').addEventListener('click',()=>{stop();Object.assign(state,{window:'all',zone:'all',layer:'heat',visitor:1,time:0});$('zone-filter').value='all';$('time-window').value='all';render();});
document.querySelectorAll('nav a').forEach(a=>a.addEventListener('click',()=>{document.querySelectorAll('nav a').forEach(link=>link.classList.toggle('active',link===a));}));
window.addEventListener('pagehide',stop);
loadMockTracks();
