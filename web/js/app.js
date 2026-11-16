"use strict";
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const clamp=(v,a,b)=>Math.min(b,Math.max(a,v)), sleep=ms=>new Promise(r=>setTimeout(r,ms));
const fmt=t=>{t=Math.max(0,t|0);return Math.floor(t/60)+':'+String(t%60).padStart(2,'0')};
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const BEH={
  touching:{c:'#C98A00',t:'Chạm hiện vật'}, holding:{c:'#E0601C',t:'Cầm / nhấc'}, stealing:{c:'#D32F2F',t:'Nghi lấy cắp'},
  candidate:{c:'#7B6CE0',t:'Cần xác nhận'}, normal:{c:'#9A9088',t:'Bình thường'}
};
const evColor=e=>e.decision==='ANOMALY'?(BEH[e.behavior]||BEH.stealing).c:e.decision==='CANDIDATE'?BEH.candidate.c:BEH.normal.c;
const evLabel=e=>e.decision==='ANOMALY'?(BEH[e.behavior]||BEH.stealing).t:e.decision==='CANDIDATE'?BEH.candidate.t:BEH.normal.t;

/* ---------- Dữ liệu mẫu (cùng schema backend sẽ trả) ---------- */
function makeMock(D=60){
  const k=D/60, S=v=>v*k;
  const arts=[{id:1,name:'Bình gốm Bát Tràng',fx:.2,fy:.35},{id:2,name:'Tượng Chăm',fx:.78,fy:.3},
              {id:3,name:'Tranh sơn mài',fx:.25,fy:.78},{id:4,name:'Trống đồng',fx:.75,fy:.75}];
  const P={
    1:[[0,.5,.97],[8,.5,.6],[20,.55,.55],[36,.8,.65],[40,.75,.82],[43,.75,.82],[52,.5,.7],[60,.15,.97]],
    2:[[2,.05,.9],[10,.2,.6],[14,.2,.42],[19,.2,.42],[28,.45,.4],[36,.6,.3],[44,.95,.5]],
    3:[[8,.95,.9],[16,.85,.6],[22,.75,.82],[26,.75,.82],[34,.5,.9],[42,.3,.95],[48,.1,.97]],
    4:[[18,.05,.5],[26,.15,.7],[31,.25,.85],[38,.25,.85],[46,.45,.6],[56,.5,.97]],
    5:[[30,.6,.97],[38,.7,.6],[47,.78,.37],[53,.78,.37],[56,.92,.6],[60,.99,.9]],
    6:[[0,.9,.5],[12,.6,.45],[24,.5,.3],[40,.35,.45],[60,.05,.5]],
    7:[[34,.1,.95],[44,.35,.55],[58,.6,.4]]};
  const tracks=Object.entries(P).map(([id,w])=>({id:+id,pts:w.map(p=>[S(p[0]),p[1],p[2]])}));
  const ev=(id,track,art,a,b,behavior,decision,conf,reason)=>({id,track,art,start:S(a),end:S(b),behavior,decision,conf,reason,review:null});
  const events=[
    ev(1,2,1,14,19,'touching','ANOMALY',.91,'Bàn tay phải đặt lên thân bình gốm khoảng 5 giây; khu vực không có biển cho phép chạm.'),
    ev(2,4,3,31,38,'holding','ANOMALY',.87,'Khách nhấc khung tranh khỏi giá treo bằng hai tay trong nhiều giây.'),
    ev(3,5,2,47,53,'stealing','ANOMALY',.93,'Khách đưa vật nhỏ vào túi áo khoác, nhìn quanh trước khi rời đi nhanh.'),
    ev(4,3,4,22,26,'uncertain','CANDIDATE',.52,'Tay ở gần trống đồng nhưng bị che một phần, chưa đủ chắc chắn để kết luận.'),
    ev(5,1,4,40,43,'normal','NORMAL',.88,'Chỉ đứng xem và chỉ tay về phía hiện vật, không có tiếp xúc.')];
  return {meta:{sequence_id:'demo_01',camera_id:'cam_01',duration:D,fps:25,floor:{w:8,h:6,unit:'m'}},arts,tracks,events};
}

/* ---------- Chuyển result.json của pipeline sang dữ liệu web ---------- */
function fromResult(res,imgFor){
  const v=res.video,fps=v.fps||25,nw=Math.min((((res.config||{}).thresholds||{}).CFG||{}).width||960,v.width);
  const arts=(res.artifacts||[]).map(a=>({id:a.id,name:a.label||('Hiện vật '+a.id),box:a.box,pedestal:!!a.pedestal}));
  const al=Object.fromEntries((res.alerts||[]).map(a=>[a.event_id,a]));
  const why={NORMAL:'VLM đánh giá hành vi bình thường.',CANDIDATE:'VLM chưa đủ chắc chắn để kết luận.',ANOMALY:'VLM kết luận có hành vi bất thường.'};
  const events=(res.events||[]).map(e=>{const a=al[e.event_id];
    return{id:e.event_id,track:e.track_id,art:e.artifact_id,start:e.start_time_s,end:e.end_time_s,
      behavior:e.decision==='CANDIDATE'?'uncertain':e.behavior,decision:e.decision,conf:e.vlm_confidence==null?(a?a.confidence:null):e.vlm_confidence,
      reason:(a&&a.evidence)||e.vlm_evidence||why[e.decision]||'',review:null,img:imgFor?imgFor(e):null}});
  return{real:true,raw:res,meta:{sequence_id:v.key,camera_id:v.name,duration:v.n_frames/fps,fps,width:v.width,height:v.height,nw},arts,tracks:[],events};
}

/* ---------- Tính toán dẫn xuất ---------- */
let DATA=null, D=60, DER=null;
function posAt(tr,t){const p=tr.pts;if(t<p[0][0]||t>p[p.length-1][0])return null;
  let i=1;while(i<p.length-1&&p[i][0]<t)i++;const a=p[i-1],b=p[i],u=b[0]===a[0]?0:(t-a[0])/(b[0]-a[0]);
  return{fx:a[1]+(b[1]-a[1])*u+Math.sin(t*1.7+tr.id)*.004,fy:a[2]+(b[2]-a[2])*u+Math.cos(t*1.3+tr.id)*.004}}
function activeAt(t){const o=[];for(const tr of DATA.tracks){const p=posAt(tr,t);if(p)o.push({id:tr.id,...p})}return o}
function derive(){
  const W=48,H=36,heat=new Float32Array(W*H),dwell={},journey={},occ=[];
  DATA.arts.forEach(a=>dwell[a.id]={total:0,vis:new Set()});
  for(const tr of DATA.tracks){journey[tr.id]=[];const seen={};
    for(let t=tr.pts[0][0];t<=tr.pts[tr.pts.length-1][0];t+=.5*D/60){const p=posAt(tr,t);if(!p)continue;
      const gx=p.fx*(W-1),gy=p.fy*(H-1);
      for(let y=Math.max(0,gy-3|0);y<Math.min(H,gy+4);y++)for(let x=Math.max(0,gx-3|0);x<Math.min(W,gx+4);x++)heat[y*W+x]+=Math.exp(-((x-gx)**2+(y-gy)**2)/3);
      for(const a of DATA.arts)if(Math.hypot(p.fx-a.fx,p.fy-(a.fy+.07))<.1){dwell[a.id].total+=.5*D/60;dwell[a.id].vis.add(tr.id);
        (seen[a.id]=seen[a.id]||{art:a.id,t0:t,dur:0}).dur+=.5*D/60}}
    journey[tr.id]=Object.values(seen).filter(s=>s.dur>=2*D/60).sort((a,b)=>a.t0-b.t0)}
  for(let t=0;t<D;t+=D/60)occ.push(activeAt(t).length);
  let mx=1e-6;heat.forEach(v=>mx=Math.max(mx,v));
  DER={W,H,heat,mx,dwell,journey,occ};
}

/* ---------- Vẽ chung ---------- */
function fit(c){const r=c.getBoundingClientRect(),d=window.devicePixelRatio||1,w=Math.round(r.width*d),h=Math.round(r.height*d);
  if(c.width!==w||c.height!==h){c.width=w;c.height=h}const x=c.getContext('2d');x.setTransform(d,0,0,d,0,0);return{x,w:r.width,h:r.height}}
const proj=(fx,fy)=>{const s=.45+.55*fy;return{x:.5+(fx-.5)*s,y:.24+.7*fy,s}};
function rr(x,X,Y,W,H,R){x.beginPath();x.roundRect(X,Y,W,H,R)}
function activeEvents(t){return DATA.events.filter(e=>t>=e.start&&t<=e.end)}

function drawReal(t){
  const {x,w,h}=fit($('#scene'));x.clearRect(0,0,w,h);
  const M=DATA.meta,hasVid=!!S.videoURL,vw=(hasVid&&vid.videoWidth)||M.width,vh=(hasVid&&vid.videoHeight)||M.height;
  const sc=Math.min(w/vw,h/vh),ox=(w-vw*sc)/2,oy=(h-vh*sc)/2,k=vw*sc/M.nw;
  if(!hasVid){x.fillStyle='#18232b';x.fillRect(ox,oy,vw*sc,vh*sc);x.fillStyle='#A9B4BA';x.font='13px Segoe UI';x.textAlign='center';
    x.fillText('Chưa chọn video — chỉ hiển thị cảnh báo và vị trí hiện vật',w/2,h/2);x.textAlign='left'}
  const aev=activeEvents(t);
  for(const a of DATA.arts){const e=aev.find(e=>e.art===a.id&&e.decision!=='NORMAL'),n=aev.find(e=>e.art===a.id);
    if((!S.layers.art&&!e)||(a.pedestal&&!e))continue;
    const [x1,y1,x2,y2]=a.box,X=ox+x1*k,Y=oy+y1*k,col=e?evColor(e):n?'#9A9088':'#5B9BEA';
    x.strokeStyle=col;x.lineWidth=e?3:2;x.strokeRect(X,Y,(x2-x1)*k,(y2-y1)*k);
    if(S.layers.labels||e){const txt=e?`${a.name} · ${evLabel(e)}`:a.name;x.font='600 11px Segoe UI';const tw=x.measureText(txt).width+10;
      x.fillStyle=col;x.fillRect(X,Y-18,tw,18);x.fillStyle='#fff';x.fillText(txt,X+5,Y-5)}}
}
function drawScene(t){
  if(DATA.real)return drawReal(t);
  const {x,w,h}=fit($('#scene')),hasVid=!!S.videoURL;
  x.clearRect(0,0,w,h);
  if(!hasVid){
    x.fillStyle='#111920';x.fillRect(0,0,w,h);
    const g=x.createLinearGradient(0,0,0,h*.26);g.addColorStop(0,'#1f2b34');g.addColorStop(1,'#2a3843');x.fillStyle=g;x.fillRect(0,0,w,h*.24);
    const f=x.createLinearGradient(0,h*.24,0,h);f.addColorStop(0,'#18232b');f.addColorStop(1,'#111920');x.fillStyle=f;x.fillRect(0,h*.24,w,h);
    x.strokeStyle='rgba(255,255,255,.05)';x.lineWidth=1;
    for(let i=0;i<=10;i++){const a=proj(i/10,0),b=proj(i/10,1);x.beginPath();x.moveTo(a.x*w,a.y*h);x.lineTo(b.x*w,b.y*h);x.stroke()}
    for(let j=0;j<=8;j++){const a=proj(0,j/8),b=proj(1,j/8);x.beginPath();x.moveTo(a.x*w,a.y*h);x.lineTo(b.x*w,b.y*h);x.stroke()}
    for(const a of DATA.arts){const p=proj(a.fx,a.fy),pw=w*.075*p.s,ph=h*.11*p.s;
      x.fillStyle='#33434f';x.fillRect(p.x*w-pw/2,p.y*h-ph,pw,ph);x.fillStyle='#cbd5ee';
      x.beginPath();x.ellipse(p.x*w,p.y*h-ph-pw*.28,pw*.3,pw*.34,0,0,7);x.fill()}
  }
  const act=activeAt(t),aev=activeEvents(t),L=S.layers;
  // vẽ silhouette khi chưa có video thật
  const boxOf=p=>{const q=proj(p.fx,p.fy),bh=h*.34*q.s,bw=bh*.42;return{x:q.x*w-bw/2,y:q.y*h-bh,w:bw,h:bh,fx:q.x*w,fy:q.y*h}};
  if(!hasVid)for(const p of act){const b=boxOf(p);x.fillStyle='rgba(165,180,210,.28)';rr(x,b.x+b.w*.12,b.y+b.h*.18,b.w*.76,b.h*.8,b.w*.3);x.fill();
    x.beginPath();x.arc(b.x+b.w/2,b.y+b.h*.09,b.w*.2,0,7);x.fill()}
  if(L.art)for(const a of DATA.arts){const p=proj(a.fx,a.fy),pw=w*.085*p.s,ph=h*.15*p.s;x.strokeStyle='#5B9BEA';x.lineWidth=2;
    x.strokeRect(p.x*w-pw/2,p.y*h-ph-pw*.3,pw,ph+pw*.3);x.fillStyle='#5B9BEA';x.font='600 11px Segoe UI';x.fillText('A'+a.id+' '+a.name,p.x*w-pw/2,p.y*h-ph-pw*.3-5)}
  for(const p of act){const b=boxOf(p),e=aev.find(e=>e.track===p.id&&e.decision!=='NORMAL');
    const col=e?evColor(e):'#4CC98C';
    if(L.skel){skeleton(x,b,p,e,t,col)}
    if(L.boxes){x.strokeStyle=col;x.lineWidth=e?3:2;x.strokeRect(b.x,b.y,b.w,b.h);
      if(L.labels){const txt='Visitor #'+p.id+(e?' · '+(e.decision==='CANDIDATE'?'?':BEH[e.behavior].t)+' '+(e.conf*100|0)+'%':'');
        x.font='600 11px Segoe UI';const tw=x.measureText(txt).width+10;x.fillStyle=col;x.fillRect(b.x,b.y-18,tw,18);x.fillStyle='#fff';x.fillText(txt,b.x+5,b.y-5)}}}
}
function skeleton(x,b,p,e,t,col){
  const sw=Math.sin(t*6+p.id)*.06,K={h:[.5,.09],n:[.5,.2],ls:[.28,.23],rs:[.72,.23],le:[.2,.42],re:[.8,.42],lw:[.22,.58],rw:[.78,.58],lh:[.38,.55],rh:[.62,.55],lk:[.4+sw,.77],rk:[.6-sw,.77],la:[.4-sw,.97],ra:[.6+sw,.97]};
  const P={};for(const k in K)P[k]=[b.x+K[k][0]*b.w,b.y+K[k][1]*b.h];
  if(e){const a=DATA.arts.find(a=>a.id===e.art),q=proj(a.fx,a.fy),W=x.canvas.width/(window.devicePixelRatio||1),H=x.canvas.height/(window.devicePixelRatio||1);
    const u=Math.min(1,(t-e.start)/1.2),tx=q.x*W,ty=q.y*H-H*.1*q.s;
    P.rw=[P.rw[0]+(tx-P.rw[0])*u,P.rw[1]+(ty-P.rw[1])*u];P.re=[(P.rs[0]+P.rw[0])/2+8,(P.rs[1]+P.rw[1])/2]}
  x.strokeStyle=col;x.lineWidth=2;x.lineCap='round';
  [['n','ls'],['n','rs'],['ls','le'],['le','lw'],['rs','re'],['re','rw'],['ls','lh'],['rs','rh'],['lh','rh'],['lh','lk'],['lk','la'],['rh','rk'],['rk','ra']].forEach(([a,c])=>{x.beginPath();x.moveTo(...P[a]);x.lineTo(...P[c]);x.stroke()});
  x.fillStyle='#fff';for(const k in P){x.beginPath();x.arc(P[k][0],P[k][1],2.2,0,7);x.fill()}
  if(e){x.fillStyle=col;x.beginPath();x.arc(P.rw[0],P.rw[1],6,0,7);x.fill()}
}

function floorRect(w,h){const pad=14,ar=4/3;let W=w-pad*2,H=W/ar;if(H>h-pad*2){H=h-pad*2;W=H*ar}return{x:(w-W)/2,y:(h-H)/2,w:W,h:H}}
function drawMap(c,o){
  const {x,w,h}=fit(c),R=floorRect(w,h),X=fx=>R.x+fx*R.w,Y=fy=>R.y+fy*R.h;
  x.clearRect(0,0,w,h);
  x.fillStyle='#19232B';rr(x,R.x,R.y,R.w,R.h,10);x.fill();x.strokeStyle='rgba(248,241,230,.5)';x.lineWidth=2;x.stroke();
  x.strokeStyle='rgba(248,241,230,.06)';x.lineWidth=1;for(let i=1;i<8;i++){x.beginPath();x.moveTo(X(i/8),R.y);x.lineTo(X(i/8),R.y+R.h);x.stroke()}
  for(let j=1;j<6;j++){x.beginPath();x.moveTo(R.x,Y(j/6));x.lineTo(R.x+R.w,Y(j/6));x.stroke()}
  x.strokeStyle='rgba(248,241,230,.5)';x.lineWidth=3;x.beginPath();x.moveTo(X(.5),R.y);x.lineTo(X(.5),Y(.14));x.moveTo(X(.5),Y(.26));x.lineTo(X(.5),Y(.5));x.stroke(); // vách ngăn + cửa
  if(o.heat){const cw=R.w/DER.W,ch=R.h/DER.H;for(let j=0;j<DER.H;j++)for(let i=0;i<DER.W;i++){const v=DER.heat[j*DER.W+i]/DER.mx;if(v<.04)continue;
    x.fillStyle=`hsla(${30-v*22},92%,${50+v*12}%,${.1+v*.5})`;x.fillRect(R.x+i*cw-cw/2,R.y+j*ch-ch/2,cw+1,ch+1)}}
  // camera + FOV
  const cx=X(.5),cy=R.y+R.h+2;x.fillStyle='rgba(242,138,75,.09)';x.beginPath();x.moveTo(cx,cy);x.lineTo(X(.02),Y(.0));x.lineTo(X(.98),Y(.0));x.closePath();x.fill();
  x.fillStyle='#F28A4B';x.beginPath();x.arc(cx,Math.min(cy,R.y+R.h),6,0,7);x.fill();
  for(const a of DATA.arts){if(o.zones){x.setLineDash([4,4]);x.strokeStyle='rgba(91,155,234,.45)';x.lineWidth=1.2;x.beginPath();x.arc(X(a.fx),Y(a.fy),R.w*.1,0,7);x.stroke();x.setLineDash([])}
    x.fillStyle='#5B9BEA';rr(x,X(a.fx)-7,Y(a.fy)-7,14,14,3);x.fill();x.font='11px Segoe UI';x.textAlign='center';const nw=x.measureText(a.name).width+10;x.fillStyle='rgba(17,25,32,.88)';x.fillRect(X(a.fx)-nw/2,Y(a.fy)-30,nw,16);x.fillStyle='#E9DFCF';x.fillText(a.name,X(a.fx),Y(a.fy)-18);x.textAlign='left'}
  if(o.path){const tr=DATA.tracks.find(t=>t.id===o.path);x.strokeStyle='#F28A4B';x.lineWidth=3;x.lineJoin='round';x.beginPath();
    tr.pts.forEach((p,i)=>i?x.lineTo(X(p[1]),Y(p[2])):x.moveTo(X(p[1]),Y(p[2])));x.stroke();
    const s=tr.pts[0];x.fillStyle='#4CC98C';x.beginPath();x.arc(X(s[1]),Y(s[2]),6,0,7);x.fill();
    DER.journey[o.path].forEach((s,i)=>{const a=DATA.arts.find(a=>a.id===s.art);x.fillStyle='#F28A4B';x.beginPath();x.arc(X(a.fx),Y(a.fy+.07),11,0,7);x.fill();x.fillStyle='#111920';x.font='700 12px Segoe UI';x.textAlign='center';x.fillText(i+1,X(a.fx),Y(a.fy+.07)+4);x.textAlign='left'})}
  if(o.live){const t=o.t,aev=activeEvents(t);
    for(const tr of DATA.tracks){const p=posAt(tr,t);if(!p)continue;
      if(o.trails){x.strokeStyle='rgba(76,201,140,.5)';x.lineWidth=2;x.beginPath();let st=false;
        for(let u=Math.max(tr.pts[0][0],t-6*D/60);u<=t;u+=.4*D/60){const q=posAt(tr,u);if(!q)continue;st?x.lineTo(X(q.fx),Y(q.fy)):(x.moveTo(X(q.fx),Y(q.fy)),st=true)}x.stroke()}
      const e=aev.find(e=>e.track===tr.id&&e.decision!=='NORMAL'),col=e?evColor(e):'#4CC98C';
      if(e){const r=10+(Math.sin(t*8)+1)*6;x.strokeStyle=col;x.lineWidth=2;x.globalAlpha=.6;x.beginPath();x.arc(X(p.fx),Y(p.fy),r,0,7);x.stroke();x.globalAlpha=1}
      x.fillStyle=col;x.beginPath();x.arc(X(p.fx),Y(p.fy),6,0,7);x.fill();x.fillStyle='#F8F1E6';x.font='600 11px Segoe UI';x.fillText('#'+tr.id,X(p.fx)+9,Y(p.fy)+4)}}
}

/* ===================== Dashboard ===================== */
const CFG=window.EF_CONFIG||{demo:false};
const S={t:0,playing:false,speed:1,videoURL:null,file:null,filter:'all',loaded:false,running:false,cancel:false,jobId:null,
  progress:0,srcName:'',layers:{boxes:true,skel:false,labels:true,art:false},mlayers:{trails:true,zones:true,heat:false}};
const vid=$('#vid');
let LIVE=null;
S.mode=(()=>{try{return localStorage.getItem('ef_mode')==='rt'?'rt':'batch'}catch{return 'batch'}})();S.liveSim=false;
const EMPTY={meta:{duration:60,camera_id:'',width:1280,height:720,nw:960},arts:[],tracks:[],events:[]};

/* ---------- Backend ---------- */
const NGROK_H={'ngrok-skip-browser-warning':'1'};
async function errText(r){try{const j=await r.json();return j.detail||JSON.stringify(j)}catch{return r.status+' '+r.statusText}}
const API={base:(()=>{try{return localStorage.getItem('ef_api')||''}catch{return ''}})(),ok:false,
  async f(p,o={}){const r=await fetch(this.base+p,{...o,headers:{...NGROK_H,...(o.headers||{})}});if(!r.ok)throw new Error(await errText(r));return r},
  async ping(){this.ok=false;
    if(this.base){try{const r=await fetch(this.base+'/api/health',{headers:NGROK_H,signal:AbortSignal.timeout(6000)});this.ok=r.ok}catch{this.ok=false}}
    const c=$('#beChip');c.classList.toggle('ok',this.ok);c.classList.toggle('bad',!this.ok);
    c.querySelector('span').textContent=this.ok?'Đã kết nối':'Mất kết nối';
    $('#beMsg').textContent=this.ok?'đã kết nối ✓':(this.base?'không kết nối được':'chưa kết nối');
    pollSystem()},
  async system(){return (await this.f('/api/system')).json()},
  async start(fd){return (await (await this.f('/api/jobs',{method:'POST',body:fd})).json()).job_id},
  async job(id){return (await this.f(`/api/jobs/${id}`)).json()},
  async result(id){return (await this.f(`/api/jobs/${id}/result`)).json()},
  async cancel(id){try{await this.f(`/api/jobs/${id}/cancel`,{method:'POST'})}catch{}},
  async blob(id,p){try{return URL.createObjectURL(await (await this.f(`/api/jobs/${id}/${p}`)).blob())}catch{return null}}
};

/* ---------- Nhật ký, trạng thái ---------- */
/* Bỏ dòng thanh tiến trình tải model (tqdm / huggingface: "Fetching 1 files: 100%|███| …") */
const noisyLog=l=>/\d+%\||it\/s|s\/it|^Fetching \d+ files|\[\d+:\d+<|[█▏▎▍▌▋▊▉]/.test(l);
/* Chỉ hiện tên bước đang chạy (không hiện log thô của pipeline như [DMA] ..., đường dẫn checkpoint) */
function stageLine(j,seen){const n=(j.stages||[]).length,i=j.stage;if(j.status==='running'&&n&&i>=0&&i<n&&!seen.has('s'+i)){seen.add('s'+i);logLine(`[${i+1}/${n}] ${j.stages[i]}`)}}
function logLine(m){const el=$('#srcmsg');el.textContent=m;el.className='srcmsg'+(/^(LỖI|Chưa|Camera chưa)/.test(m)?' err':'')}
function setStatus(t){$('#vstat').textContent=t}
function setBusy(b){S.running=b;$('#modeSel').classList.toggle('lock',b);$('#runBtn').disabled=b;$('#stopBtn').disabled=!b;$('#uploadBtn').disabled=b;$('#src').disabled=b;$('#camSel').disabled=b;$('#camUrl').disabled=b;if(b)$('#runBtn').textContent='Đang chạy…';else applySourceUI()}

/* ---------- System status ---------- */
let SYS=null;const sim={cpu:6,ram:2.4,gpu:0,vram:.3};
const ring=(pct,main,sub,label,col)=>`<div class="ring"><svg viewBox="0 0 80 80"><circle cx="40" cy="40" r="34" class="rbg"/><circle cx="40" cy="40" r="34" class="rfg" style="stroke:${col};stroke-dasharray:${(clamp(pct,0,100)/100*213.6).toFixed(1)} 213.6"/></svg><div class="rtxt"><b>${main}</b><small>${esc(sub)}</small></div><span>${label}</span></div>`;
function renderSystem(){
  const c=SYS&&SYS.cpu,r=SYS&&SYS.ram,g=SYS&&SYS.gpus&&SYS.gpus[0],na='—';
  $('#rings').innerHTML=[
    ring(c?c.pct:0,c?c.pct.toFixed(1)+'%':na,c?'load '+c.load:'','CPU','#5B9BEA'),
    ring(r?r.used_gb/r.total_gb*100:0,r?r.used_gb+' GB':na,r?'/ '+r.total_gb+' GB':'','RAM','#7B6CE0'),
    ring(g?g.util:0,g?Math.round(g.util)+'%':na,g?g.name:'','GPU','#2E9B6A'),
    ring(g?g.mem_used_gb/g.mem_total_gb*100:0,g?g.mem_used_gb+' GB':na,g?'/ '+g.mem_total_gb+' GB':'','VRAM','#E0601C')].join('');
  $('#sysTag').textContent=SYS?(API.ok?'':'mô phỏng'):'chưa có dữ liệu'}
async function pollSystem(){
  if(API.ok){try{SYS=await API.system()}catch{SYS=null}}
  else if(CFG.demo){const j=(v,a,b,s)=>clamp(v+(Math.random()-.5)*s,a,b),busy=S.running;
    sim.cpu=j(sim.cpu,busy?40:2,busy?85:12,6);sim.ram=j(sim.ram,2,busy?9:3.5,.4);sim.gpu=j(sim.gpu,busy?30:0,busy?95:3,10);sim.vram=j(sim.vram,busy?4:.3,busy?12:.6,.5);
    SYS={cpu:{pct:+sim.cpu.toFixed(1),load:+(sim.cpu/25).toFixed(2)},ram:{used_gb:+sim.ram.toFixed(1),total_gb:33.7},gpus:[{name:'Tesla T4',util:sim.gpu,mem_used_gb:+sim.vram.toFixed(1),mem_total_gb:15}]}}
  else SYS=null;
  renderSystem()}

/* ---------- Hiển thị dữ liệu ---------- */
const LEVEL={stealing:['Cao','hi'],holding:['Trung bình','mid'],touching:['Thấp','lo']};
function listEvents(){if(!S.liveSim)return DATA.events;
  return DATA.events.filter(e=>e.start<=S.t).map(e=>e.end<=S.t?e:{...e,decision:'CANDIDATE',behavior:'uncertain',conf:null,reason:'Đang chạm (chờ VLM kết luận)',provisional:true})}
function renderKpis(){const ev=listEvents(),an=ev.filter(e=>e.decision==='ANOMALY').length;
  const items=[['Video nguồn',S.srcName||'—','src'],['Tiến độ',Math.round(S.progress)+'%',''],['Alerts',an,'alert'],['Total detections',ev.length,'']];
  $('#kpis').innerHTML=items.map(([a,b,k])=>`<div class="kpi ${k==='alert'&&an?'red':''}"><small>${a}</small>${k==='alert'?`<i class="dotred${an?' on':''}"></i>`:''}<b class="${k==='src'?'sm':''}">${esc(b)}</b></div>`).join('')}
function thumb(e){const c=document.createElement('canvas');c.width=240;c.height=150;const x=c.getContext('2d'),col=evColor(e);
  x.fillStyle='#18232b';x.fillRect(0,0,240,150);x.fillStyle='#1d2740';x.fillRect(0,0,240,40);
  x.fillStyle='#33434f';x.fillRect(130,70,50,60);x.fillStyle='#cbd5ee';x.beginPath();x.ellipse(155,64,15,17,0,0,7);x.fill();
  x.fillStyle='rgba(165,180,210,.4)';rr(x,70,40,42,95,14);x.fill();x.beginPath();x.arc(91,36,11,0,7);x.fill();
  x.strokeStyle=col;x.lineWidth=3;x.strokeRect(62,22,58,118);x.strokeStyle='#5B9BEA';x.strokeRect(122,44,66,92);
  x.strokeStyle=col;x.lineWidth=4;x.beginPath();x.moveTo(108,78);x.lineTo(138,76);x.stroke();return c.toDataURL()}
function noImg(){const c=document.createElement('canvas');c.width=240;c.height=150;const x=c.getContext('2d');x.fillStyle='#18232b';x.fillRect(0,0,240,150);
  x.fillStyle='#A9B4BA';x.font='13px Segoe UI';x.textAlign='center';x.fillText('Không có ảnh bằng chứng',120,78);return c.toDataURL()}
function renderAlerts(){
  const list=listEvents().filter(e=>S.filter==='all'||e.decision===S.filter),box=$('#alerts');
  if(!list.length){box.innerHTML='<p class="note empty">Chưa có cảnh báo.</p>';return}
  box.innerHTML=list.map(e=>{const a=DATA.arts.find(a=>a.id===e.art)||{name:'Hiện vật '+e.art};
    const lab=e.provisional?'Đang chạm':evLabel(e),col=e.provisional?'#C98A00':evColor(e);
    if(!e.img)e.img=DATA.real?noImg():thumb(e);
    const btn=e.review?`<span class="st">✓ ${e.review==='ok'?'Đã xác nhận':'Đã bỏ qua'}</span>`
      :`<button class="btn sm" data-seek="${e.start}">Xem lại</button>${(e.decision==='NORMAL'||e.provisional)?'':`<button class="btn sm" data-rv="ok" data-id="${e.id}">Xác nhận</button><button class="btn sm" data-rv="no" data-id="${e.id}">Bỏ qua</button>`}`;
    return`<div class="al" data-id="${e.id}" style="--c:${col}"><img src="${e.img}" data-seek="${e.start}" alt="">
      <div><div class="hd"><span class="badge">${lab}</span><b>Visitor #${e.track}</b><span class="meta">→ ${esc(a.name)}</span></div>
      <div class="meta">${fmt(e.start)}–${fmt(e.end)} · VLM tin cậy ${e.conf==null?'—':((e.conf*100)|0)+'%'}</div>
      <div class="why">“${esc(e.reason)}”</div><div class="acts">${btn}</div></div></div>`}).join('')}
function renderTimeline(){const tl=$('#tl');tl.querySelectorAll('.seg').forEach(n=>n.remove());
  listEvents().forEach(e=>{const d=document.createElement('div');d.className='seg';d.style.cssText=`left:${e.start/D*100}%;width:${(e.end-e.start)/D*100}%;background:${evColor(e)}`;d.title=`#${e.track} ${evLabel(e)}`;d.dataset.seek=e.start;tl.append(d)});
  const o=DER.occ;if(DATA.real||!o.length||Math.max(...o)===0){$('#occ').innerHTML='';return}
  const m=Math.max(...o),pts=o.map((v,i)=>`${i/(o.length-1)*100},${100-v/m*70-4}`).join(' ');
  $('#occ').setAttribute('viewBox','0 0 100 100');$('#occ').innerHTML=`<polyline points="0,100 ${pts} 100,100" fill="rgba(25,35,43,.07)" stroke="rgba(25,35,43,.4)" stroke-width=".8" vector-effect="non-scaling-stroke"/>`}
function renderAll(){renderKpis();renderAlerts();renderTimeline()}

function hideLive(){LIVE=null;$('#liveimg').hidden=true;$('#rthud').hidden=true;const u=$('#liveimg').src;if(u&&u.startsWith('blob:'))URL.revokeObjectURL(u);$('#liveimg').removeAttribute('src')}
function loadData(data,videoURL,opts={}){
  hideLive();S.liveSim=false;
  DATA=data;D=data.meta.duration;derive();S.t=0;S.loaded=true;S.filter='all';
  if(videoURL&&videoURL!==S.videoURL){S.videoURL=videoURL;vid.src=videoURL}
  vid.hidden=!S.videoURL;$('#vmsg').hidden=true;
  document.body.classList.toggle('real',!!data.real);
  S.layers.art=data.real?!opts.liveVideo:false;S.layers.labels=true;
  $$('#layers [data-l="art"]').forEach(b=>b.classList.toggle('on',S.layers.art));
  $('#map').hidden=false;$('#mapNote').hidden=true;
  S.srcName=opts.name||(data.real?data.meta.camera_id:(S.file?S.file.name:'Dữ liệu mẫu'));
  $$('#filters .tg').forEach(b=>b.classList.toggle('on',b.dataset.f==='all'));
  setPlaying(false);renderAll()}
function resetData(){resetMap();hideLive();S.liveSim=false;DATA=EMPTY;D=60;derive();S.loaded=false;S.t=0;document.body.classList.remove('real');$('#map').hidden=false;$('#mapNote').hidden=true;renderAll()}

/* ---------- Map 2D thật (multicam-2d-floorplan-tracker qua backend): ảnh map + vị trí người / hiện vật ---------- */
const MAP0=()=>({ver:-1,img:null,people:[],arts:[],frames:null,keys:null,fps:25,shown:0,trail:{}});
let MAP=MAP0();
function resetMap(){MAP=MAP0();if(typeof SITEDATA!=='undefined')SITEDATA=null}
async function pollMap(id,alive,full){
  try{const d=await(await API.f(`/api/jobs/${id}/map/data${full?'?full=1':''}`)).json();
    if(!alive()&&!full)return;
    if(d.has_map&&d.version!==MAP.ver){const u=await API.blob(id,'map');if(u){const im=new Image();im.onload=()=>{MAP.img=im};im.src=u;MAP.ver=d.version}}
    MAP.people=d.people||[];MAP.arts=d.artifacts||[];MAP.fps=d.fps||MAP.fps;
    for(const p of MAP.people){const a=MAP.trail[p.id]||(MAP.trail[p.id]=[]);a.push([p.x,p.y]);if(a.length>8)a.shift()}
    if(full&&d.frames){MAP.frames=d.frames;MAP.keys=Object.keys(d.frames).map(Number).sort((a,b)=>a-b)}
  }catch{}}
async function mapLoop(id,alive){while(alive()){await pollMap(id,alive,false);await sleep(1000)}}
function mapPeopleAt(t){            // phát lại: vị trí theo khung gần nhất không vượt quá thời điểm t
  if(!MAP.keys||!MAP.keys.length)return null;const f=Math.round(t*MAP.fps);let lo=0,hi=MAP.keys.length-1;
  if(f<MAP.keys[0])return[];
  while(lo<hi){const m=(lo+hi+1)>>1;MAP.keys[m]<=f?lo=m:hi=m-1}
  return MAP.frames[MAP.keys[lo]].map(p=>({id:p[0],x:p[1],y:p[2]}))}
function trailAt(id,t){             // vệt đi 6 giây gần nhất khi phát lại
  const out=[],f1=Math.round(t*MAP.fps),f0=f1-6*MAP.fps;if(!MAP.keys)return out;
  for(let i=0;i<MAP.keys.length;i+=3){const k=MAP.keys[i];if(k<f0)continue;if(k>f1)break;const p=MAP.frames[k].find(q=>q[0]===id);if(p)out.push([p[1],p[2]])}
  return out}
function drawRealMap(c,t){
  const {x,w,h}=fit(c);x.clearRect(0,0,w,h);
  const im=MAP.img,pad=10,s=Math.min((w-pad*2)/im.naturalWidth,(h-pad*2)/im.naturalHeight),iw=im.naturalWidth*s,ih=im.naturalHeight*s,ox=(w-iw)/2,oy=(h-ih)/2;
  x.save();x.beginPath();x.roundRect(ox,oy,iw,ih,6);x.clip();x.drawImage(im,ox,oy,iw,ih);x.restore();
  x.strokeStyle='rgba(255,255,255,.25)';x.lineWidth=1;x.beginPath();x.roundRect(ox,oy,iw,ih,6);x.stroke();
  const X=v=>ox+v*iw,Y=v=>oy+v*ih;
  if(S.mlayers.zones)for(const a of MAP.arts){x.setLineDash([4,4]);x.strokeStyle='rgba(91,155,234,.6)';x.lineWidth=1.2;x.beginPath();x.arc(X(a.x),Y(a.y),iw*.06,0,7);x.stroke();x.setLineDash([])}
  x.textAlign='center';
  for(const a of MAP.arts){x.fillStyle='#5B9BEA';x.fillRect(X(a.x)-6,Y(a.y)-6,12,12);x.fillStyle='#fff';x.font='600 11px Segoe UI';x.fillText('Hiện vật '+a.id,X(a.x),Y(a.y)-10)}
  const replay=!LIVE&&!!MAP.keys&&MAP.keys.length>0&&!!S.videoURL,ppl=replay?(mapPeopleAt(t)||[]):MAP.people;
  const aev=replay?activeEvents(t).filter(e=>e.decision==='ANOMALY'):[];
  MAP.shown=ppl.length;x.textAlign='left';
  for(const p of ppl){
    if(S.mlayers.trails){const tr=replay?trailAt(p.id,t):(MAP.trail[p.id]||[]);
      if(tr.length>1){x.strokeStyle='rgba(61,220,151,.55)';x.lineWidth=2;x.beginPath();tr.forEach((q,i)=>i?x.lineTo(X(q[0]),Y(q[1])):x.moveTo(X(q[0]),Y(q[1])));x.stroke()}}
    const e=aev.find(e=>e.track===p.id),st=e?'alert':(p.state||'idle'),col=st==='alert'?'#FF4D4D':st==='contact'?'#FFB020':'#3DDC97';
    if(st==='alert'){x.strokeStyle=col;x.globalAlpha=.6;x.lineWidth=2;x.beginPath();x.arc(X(p.x),Y(p.y),10+(Math.sin(t*8)+1)*5,0,7);x.stroke();x.globalAlpha=1}
    x.fillStyle=col;x.beginPath();x.arc(X(p.x),Y(p.y),6,0,7);x.fill();x.strokeStyle='#0B0F13';x.lineWidth=1.5;x.stroke();
    x.fillStyle='#fff';x.font='600 11px Segoe UI';x.fillText('#'+p.id,X(p.x)+9,Y(p.y)+4)}}
function drawMapPanel(t){const c=$('#map');   // true nếu đã vẽ map thật; false thì vẽ khung giữ chỗ như cũ
  if(typeof SITEDATA!=='undefined'&&SITEDATA){drawSiteMap(c,t);return true}
  if(MAP.img&&MAP.img.complete&&MAP.img.naturalWidth){drawRealMap(c,t);return true}
  drawMap(c,DATA.real?{live:false,t:0,...S.mlayers}:{live:S.loaded,t,...S.mlayers});return false}

/* ---------- Phát lại ---------- */
function setPlaying(p){S.playing=p;$('#play').textContent=p?'⏸':'▶';if(S.videoURL)p?vid.play().catch(()=>{}):vid.pause()}
function seek(t){S.t=clamp(t,0,D);if(S.videoURL)vid.currentTime=S.t}
vid.addEventListener('loadedmetadata',()=>{if(!S.loaded&&vid.duration)D=vid.duration});
let last=performance.now();
function tick(now){const dt=(now-last)/1000;last=now;
  if(S.videoURL)S.t=vid.currentTime||0;else if(S.playing){S.t+=dt*S.speed;if(S.t>=D){S.t=D;setPlaying(false)}}
  paint();requestAnimationFrame(tick)}
function paint(){const t=S.t,L=$('#live');
  if(LIVE){const {x,w,h}=fit($('#scene'));x.clearRect(0,0,w,h);
    $('#ph').style.left=(D?t/D*100:0)+'%';$('#time').textContent=fmt(t)+' / '+fmt(D);L.hidden=false;
    if(LIVE.banner&&performance.now()<LIVE.banner.until){L.className='live '+LIVE.banner.cls;L.textContent=LIVE.banner.text}else{L.className='live grn';L.textContent='● LIVE'}
    {const m=drawMapPanel(t);$('#mapCount').textContent=(m?MAP.shown:0)+' người'}
    return}
  if(S.loaded)drawScene(t);else{const {x,w,h}=fit($('#scene'));x.clearRect(0,0,w,h)}
  const hasMap=drawMapPanel(t);
  $('#ph').style.left=(D?t/D*100:0)+'%';$('#time').textContent=fmt(t)+' / '+fmt(D);
  if(!S.loaded)L.hidden=true;else{L.hidden=false;
    const aev=activeEvents(t).filter(e=>e.decision!=='NORMAL'),an=aev.find(e=>e.decision==='ANOMALY'),ca=aev.find(e=>e.decision==='CANDIDATE');
    if(an){L.className='live red';L.textContent=`Visitor #${an.track}: ${evLabel(an)} — ${(DATA.arts.find(a=>a.id===an.art)||{name:'hiện vật '+an.art}).name}`}
    else if(ca){L.className='live yel';L.textContent=`Visitor #${ca.track} cần xác nhận`}
    else{L.className='live grn';L.textContent='● Bình thường'}}
  $('#mapCount').textContent=hasMap?MAP.shown+' người':DATA.real?'0 người':(S.loaded?activeAt(t).length:0)+' người';
  const act=S.loaded?activeEvents(t):[];
  $$('#alerts .al').forEach(el=>el.classList.toggle('act',act.some(e=>e.id==el.dataset.id&&e.decision!=='NORMAL')))}

/* ---------- Chạy phân tích ---------- */
const DEMO_STEPS=[['Khởi động & nạp model',['Nạp detector YOLOX','Nạp tracker ByteTrack-DMA-LTC']],['Tách hiện vật',['MM-Grounding-DINO: 4 hiện vật','SAM2.1 tạo mask xong']],
  ['Nạp VLM',['Qwen3-VL-8B Q4_K_M sẵn sàng']],['Phân tích video',['Tracking: 7 track','Phát hiện 5 sự kiện chạm','VLM: touching 0.91','VLM: holding 0.87','VLM: stealing 0.93','VLM: uncertain 0.52']],['Ghi kết quả',['Xuất result.json']]];
function stopped(){setStatus('Đã dừng');logLine('Đã dừng.')}
async function runDemo(){
  setStatus('Đang chạy (mô phỏng)');logLine('Chế độ demo: kết quả là dữ liệu mẫu.');
  const dur=S.videoURL&&vid.duration?Math.round(vid.duration)||60:60;
  for(let i=0;i<DEMO_STEPS.length;i++){const [name,subs]=DEMO_STEPS[i];logLine(`[${i+1}/${DEMO_STEPS.length}] ${name}`);
    for(let k=0;k<subs.length;k++){if(S.cancel)return stopped();logLine('  '+subs[k]);S.progress=(i+(k+1)/subs.length)/DEMO_STEPS.length*95;renderKpis();await sleep(450)}}
  if(S.cancel)return stopped();
  S.progress=100;loadData(makeMock(dur),S.videoURL,{name:S.file?S.file.name:'Dữ liệu mẫu'});setStatus('Hoàn tất (dữ liệu mẫu)');logLine('Hoàn tất.')}
async function runLive(src){
  const fd=new FormData();if(S.file)fd.append('video',S.file);else fd.append('source',src);
  fd.append('options',JSON.stringify({conf:+$('#conf').value}));
  setStatus('Đang chạy');logLine(S.file?'Tải video lên backend…':'Gửi đường dẫn / URL tới backend…');
  const id=await API.start(fd);S.jobId=id;const seen=new Set();let key=null;
  resetMap();mapLoop(id,()=>S.running&&S.jobId===id);
  for(;;){
    if(S.cancel){await API.cancel(id);return stopped()}
    const j=await API.job(id);
    S.progress=j.status==='done'?100:(Math.max(0,j.stage)+j.progress)/j.stages.length*100;
    stageLine(j,seen);renderKpis();
    if(j.status==='done'){key=j.key;break}
    if(j.status==='error')throw new Error(j.error);
    if(j.status==='cancelled')return stopped();
    await sleep(2000)}
  const res=await API.result(id),imgs={};
  await Promise.all(res.events.filter(e=>e.task).map(async e=>{imgs[e.event_id]=await API.blob(id,`files/${key}/vlm_inputs/${e.task}/sheet.jpg`)}));
  const videoURL=await API.blob(id,'video');
  await pollMap(id,()=>true,true);
  loadData(fromResult(res,e=>imgs[e.event_id]),videoURL,{liveVideo:true,name:S.file?S.file.name:(src.split(/[\/]/).pop()||undefined)});
  setStatus('Hoàn tất');logLine('Hoàn tất.');return true}
/* ---------- Realtime ---------- */
function prepareLiveView(name){
  S.videoURL=null;vid.removeAttribute('src');vid.hidden=true;$('#vmsg').hidden=true;
  $('#liveimg').hidden=false;$('#rthud').hidden=false;$('#rthud').textContent='LIVE · đang khởi động…';
  DATA={real:true,raw:null,meta:{duration:60,camera_id:name,width:960,height:540,nw:960},arts:[],tracks:[],events:[]};D=60;derive();
  S.loaded=true;S.t=0;S.filter='all';document.body.classList.add('real');S.srcName=name;
  $('#map').hidden=false;$('#mapNote').hidden=true;renderAll()}
function applyLive(le){
  const L=LIVE;if(!L)return;
  if(le.meta&&!L.meta){L.meta=le.meta;
    DATA.arts=(le.meta.artifacts||[]).map(a=>({id:a.id,name:a.label||('Hiện vật '+a.id),box:a.box,pedestal:!!a.pedestal}));
    D=le.meta.n_frames/le.meta.fps;DATA.meta.duration=D;derive()}
  for(const e of le.events||[]){L.after=Math.max(L.after,e.seq);
    if(e.kind==='contact'){
      DATA.events.push({id:'c'+e.seq,track:e.track,art:e.artifact,start:e.t_s,end:e.t_s+.5,behavior:'uncertain',decision:'CANDIDATE',conf:null,
        reason:'Đang chạm (chờ VLM kết luận)',provisional:true,review:null,img:null});
      logLine(`Chạm: Visitor #${e.track} → hiện vật ${e.artifact}`)}
    else if(e.kind==='vlm'){
      const art=(e.artifacts||[])[0],i=DATA.events.findIndex(x=>x.provisional&&x.track===e.track&&x.art===art);
      const row={id:'v'+e.seq,track:e.track,art,start:e.start_s!=null?e.start_s:(i>=0?DATA.events[i].start:S.t),end:e.end_s!=null?e.end_s:S.t,
        behavior:e.decision==='ANOMALY'?(e.behavior||'touching'):e.decision==='CANDIDATE'?'uncertain':'normal',decision:e.decision,conf:e.confidence,
        reason:e.evidence||(e.decision==='NORMAL'?'VLM đánh giá hành vi bình thường.':e.decision==='ANOMALY'?'VLM kết luận có hành vi bất thường.':'VLM chưa đủ chắc chắn.'),review:null,img:null};
      if(i>=0)DATA.events.splice(i,1,row);else DATA.events.push(row);
      logLine(`VLM: ${e.label||e.decision}${e.confidence==null?'':' '+e.confidence} (Visitor #${e.track})`);
      if(e.decision==='ANOMALY')L.banner={text:`Visitor #${e.track}: ${evLabel(row)} — ${(DATA.arts.find(a=>a.id===art)||{name:'hiện vật '+art}).name}`,cls:'red',until:performance.now()+8000}}}
  const st=le.status;
  if(st){S.t=st.t_s||0;S.progress=st.n_frames?Math.max(0,st.frame)/st.n_frames*100:0;
    $('#rthud').textContent=`LIVE · khung ${Math.max(st.frame,0)}/${st.n_frames} · ${st.live_fps} FPS · bỏ ${st.dropped} khung · VLM chờ ${st.vlm_pending==null?0:st.vlm_pending}`}
  renderAll()}
async function runRealtime(src,cam){
  const fd=new FormData();if(!cam&&S.file)fd.append('video',S.file);else fd.append('source',src);
  fd.append('options',JSON.stringify({conf:+$('#conf').value,realtime:true,camera:!!cam}));
  setStatus('Realtime: đang khởi động');logLine('Realtime: gửi yêu cầu tới backend…');
  const id=await API.start(fd);S.jobId=id;
  prepareLiveView(cam?camChoice().name:(S.file?S.file.name:src));LIVE={id,after:0,meta:null,banner:null};const mine=LIVE;resetMap();mapLoop(id,()=>LIVE===mine);
  let stop=false;const img=$('#liveimg');
  const frames=(async()=>{while(!stop&&LIVE===mine){
    try{const u=await API.blob(id,'live');if(u&&LIVE===mine){const old=img.src;img.src=u;if(old&&old.startsWith('blob:'))URL.revokeObjectURL(old)}}catch{}
    await sleep(120)}})();
  const seen=new Set();let key=null;
  try{for(;;){
    if(S.cancel){await API.cancel(id);stop=true;LIVE=null;return stopped()}
    const j=await API.job(id);
    stageLine(j,seen);
    if(j.stage>=3||j.status==='done')setStatus('Realtime · LIVE');
    let le=null;try{le=await(await API.f(`/api/jobs/${id}/live/events?after=${mine.after}`)).json()}catch{}
    if(le&&LIVE===mine)applyLive(le);
    if(j.status==='done'){key=j.key;break}
    if(j.status==='error')throw new Error(j.error);
    if(j.status==='cancelled'){stop=true;LIVE=null;return stopped()}
    await sleep(1000)}
  }catch(e){stop=true;LIVE=null;throw e}
  stop=true;await frames;
  const res=await API.result(id),imgs={};
  await Promise.all(res.events.filter(e=>e.task).map(async e=>{imgs[e.event_id]=await API.blob(id,`files/${key}/vlm_inputs/${e.task}/sheet.jpg`)}));
  const videoURL=await API.blob(id,'video');S.progress=100;
  await pollMap(id,()=>true,true);
  loadData(fromResult(res,e=>imgs[e.event_id]),videoURL,{liveVideo:false,name:S.file?S.file.name:(src.split(/[\/]/).pop()||undefined)});
  setStatus('Hoàn tất');logLine('Hoàn tất realtime: đã thay danh sách tạm bằng kết quả cuối.')}
async function runDemoRealtime(cam){
  if(cam){S.videoURL=null;vid.removeAttribute('src')}
  setStatus(cam?'Camera (mô phỏng)':'Realtime (mô phỏng)');logLine('Chế độ demo: mô phỏng realtime, cảnh báo hiện dần theo thời gian video.');
  const dur=S.videoURL&&vid.duration?Math.round(vid.duration)||60:60;
  loadData(makeMock(dur),S.videoURL,{name:cam?camChoice().name:(S.file?S.file.name:'Dữ liệu mẫu')});
  S.liveSim=true;S.progress=0;S.t=0;if(S.videoURL)vid.currentTime=0;setPlaying(true);renderAll();
  while(S.t<D-.05&&!S.cancel){S.progress=S.t/D*100;renderAll();await sleep(500)}
  setPlaying(false);
  if(S.cancel){S.progress=S.t/D*100;renderKpis();return stopped()}
  S.liveSim=false;S.progress=100;renderAll();setStatus('Hoàn tất (mô phỏng)');logLine('Hoàn tất.')}
function setMode(m){S.mode=m;try{localStorage.setItem('ef_mode',m)}catch{}
  $$('#modeSel .tg').forEach(b=>b.classList.toggle('on',b.dataset.mode===m));
  applySourceUI()}
$('#modeSel').onclick=e=>{const b=e.target.closest('[data-mode]');if(b&&!S.running)setMode(b.dataset.mode)};

async function startAnalysis(){
  if(S.running)return;
  const cam=camMode(),src=cam?camChoice().url:$('#src').value.trim();
  if(cam&&!src&&!CFG.demo){setStatus('Chưa có URL camera');logLine('Camera chưa có URL: vào Settings → Camera để nhập (vd. rtsp://...).');return}
  if(!cam&&!S.file&&!src){setStatus('Chưa có video');logLine('Chưa có video: bấm Upload hoặc nhập đường dẫn / URL.');return}
  S.cancel=false;S.progress=0;setBusy(true);renderKpis();
  try{
    let site=null;
    if(API.ok&&!cam&&S.mode==='batch'&&S.files.length>=1){logLine('Đang nhận diện camera…');setStatus('Nhận diện camera');site=await matchSite(S.files)}
    if(site)await runSite(S.files,site);
    else if(API.ok&&!cam&&S.mode==='batch'&&S.files.length>1)await runMulti(S.files);
    else if(API.ok)await(S.mode==='rt'?runRealtime(src,cam):runLive(src));
    else if(CFG.demo)await(S.mode==='rt'?runDemoRealtime(cam):runDemo());
    else{setStatus('Mất kết nối');logLine('Chưa kết nối backend. Vào Settings để nhập URL.')}
  }catch(err){logLine('LỖI: '+err.message);setStatus('Lỗi')}
  finally{setBusy(false);pollSystem()}}

/* ---------- Nhiều camera: mỗi video chạy riêng, mỗi camera một tab (video, cảnh báo, map riêng) ---------- */
S.files=[];let CAMRES=[],CAMIDX=-1;
function camSnap(name){return{name,data:DATA,videoURL:S.videoURL,map:MAP}}
function renderCamTabs(){const el=$('#camTabs');el.hidden=CAMRES.length<2;
  el.innerHTML=CAMRES.map((c,i)=>{const n=c.data.events.filter(e=>e.decision==='ANOMALY').length;
    return`<button class="tg${i===CAMIDX?' on':''}" data-cam="${i}" title="${esc(c.name)}">${esc(c.label||('CAM '+(i+1)))}${n?` · ${n}`:''}</button>`}).join('')}
function showCam(i){const c=CAMRES[i];if(!c)return;CAMIDX=i;MAP=c.map;
  loadData(c.data,c.videoURL,{liveVideo:true,name:c.name});S.srcName=c.name;renderKpis();renderCamTabs()}
$('#camTabs').onclick=e=>{const b=e.target.closest('[data-cam]');if(b&&!S.running)showCam(+b.dataset.cam)};
async function runMulti(files){
  CAMRES=[];CAMIDX=-1;renderCamTabs();
  for(let i=0;i<files.length;i++){
    if(S.cancel)break;
    S.file=files[i];S.srcName=files[i].name;S.progress=0;renderKpis();
    logLine(`[Camera ${i+1}/${files.length}] ${files[i].name}`);setStatus(`Camera ${i+1}/${files.length}`);
    const ok=await runLive('');
    if(!ok||S.cancel)break;
    CAMRES.push(camSnap(files[i].name));CAMIDX=CAMRES.length-1;renderCamTabs()}
  if(CAMRES.length){showCam(0);setStatus(CAMRES.length>1?`Hoàn tất ${CAMRES.length} camera`:'Hoàn tất');logLine(CAMRES.length>1?`Hoàn tất ${CAMRES.length} camera: bấm CAM 1, CAM 2... để xem từng camera.`:'Hoàn tất.')}}

/* ---------- Tự nhận diện camera đã hiệu chuẩn và ghép nhiều camera thành MỘT map (multicam-2d-floorplan-tracker) ---------- */
const SITE_COLORS=['#5B9BEA','#3DDC97','#FFA040','#B388FF','#FF6B8A','#4DD0E1','#FFD54F'];
let SITEDATA=null;
async function blobUrl(path){try{return URL.createObjectURL(await(await API.f(path)).blob())}catch{return null}}
async function grabFrames(file){          // 3 khung hình nhỏ (15%, 50%, 85% thời lượng) để server nhận diện camera
  const url=URL.createObjectURL(file),v=document.createElement('video');v.muted=true;v.preload='auto';v.src=url;
  try{await new Promise((ok,no)=>{v.onloadedmetadata=ok;v.onerror=()=>no(0);setTimeout(()=>no(0),15000)});
    const out=[],c=document.createElement('canvas');c.width=640;c.height=360;const x=c.getContext('2d');
    for(const p of [.15,.5,.85]){v.currentTime=Math.max(0,(v.duration||0)*p);await new Promise((ok,no)=>{v.onseeked=ok;setTimeout(()=>no(0),10000)});
      x.drawImage(v,0,0,640,360);const b=await new Promise(r=>c.toBlob(r,'image/jpeg',.85));if(b)out.push(b)}
    return out}catch{return[]}finally{URL.revokeObjectURL(url)}}
async function matchSite(files){          // trả về {site_id,name,assign:{chỉ số video:id camera}} hoặc null nếu không nhận ra
  try{const fd=new FormData();let n=0;
    for(let i=0;i<files.length;i++)(await grabFrames(files[i])).forEach((b,j)=>{fd.append(`v${i}_${j}`,b,`f${i}_${j}.jpg`);n++});
    if(!n)return null;
    const r=await(await API.f('/api/site-match',{method:'POST',body:fd})).json();return r&&r.matched?r:null}catch{return null}}
const camNum=id=>(String(id).match(/\d+/)||[0])[0]*1;
async function runSite(files,m){
  const map=Object.keys(m.assign).map(Number).map(i=>({cam:m.assign[i],file:files[i]})).sort((a,b)=>camNum(a.cam)-camNum(b.cam)),skipped=files.length-map.length;
  const label=id=>'CAM '+(camNum(id)||id);
  CAMRES=[];CAMIDX=-1;renderCamTabs();resetMap();
  const fd=new FormData();fd.append('site',m.site_id);map.forEach(x=>fd.append(x.cam,x.file));
  logLine(`Nhận ra ${m.name}: `+map.map(x=>`${x.file.name} → ${label(x.cam)}`).join(' · ')+(skipped?` (bỏ qua ${skipped} video không khớp)`:''));setStatus('Ghép camera');
  const id=(await(await API.f('/api/site-jobs',{method:'POST',body:fd})).json()).job_id;S.jobId=id;
  for(;;){
    if(S.cancel){logLine('Đã dừng. Server vẫn xử lý xong lượt ghép hiện tại.');return stopped()}
    const j=await(await API.f('/api/site-jobs/'+id)).json();
    S.progress=j.status==='done'?100:(j.stage+.5)/j.stages.length*100;renderKpis();
    logLine(`[${j.stage+1}/${j.stages.length}] ${j.stages[j.stage]}`);
    if(j.status==='error')throw new Error(j.error);
    if(j.status==='done')break;
    await sleep(1500)}
  const res=await(await API.f(`/api/site-jobs/${id}/result`)).json();
  SITEDATA={bounds:res.bounds,cams:res.cameras,frames:res.frames,fps:res.fps};
  const dur=res.frames.length/(res.fps||1);
  for(const x of map){
    const data={real:true,raw:null,meta:{duration:dur,camera_id:x.file.name,width:1920,height:1080,nw:960},arts:[],tracks:[],events:[]};
    CAMRES.push({name:x.file.name,label:label(x.cam),cam:x.cam,data,videoURL:await blobUrl(`/api/site-jobs/${id}/video/${x.cam}`),map:MAP})}
  showCam(0);setStatus(map.length>1?`Hoàn tất ${map.length} camera`:'Hoàn tất');
  const ids=new Set(res.frames.flatMap(f=>f.people.map(p=>p.id)));
  logLine(map.length>1?`Hoàn tất: ${map.length} camera ghép thành một map chung, ${ids.size} người (ID chung).`:`Hoàn tất: map từ 1 camera, ${ids.size} người.`);return true}
function siteFrameAt(t){const F=SITEDATA.frames;let lo=0,hi=F.length-1;if(!F.length||t<F[0].t)return -1;
  while(lo<hi){const m=(lo+hi+1)>>1;F[m].t<=t?lo=m:hi=m-1}return lo}
function drawSiteMap(c,t){
  const {x,w,h}=fit(c);x.clearRect(0,0,w,h);const D_=SITEDATA,[x0,y0,x1,y1]=D_.bounds,pad=12;
  const sc=Math.min((w-pad*2)/(x1-x0),(h-pad*2)/(y1-y0)),ox=(w-(x1-x0)*sc)/2,oy=(h-(y1-y0)*sc)/2,X=v=>ox+(v-x0)*sc,Y=v=>oy+(y1-v)*sc;
  x.fillStyle='#0D1217';x.beginPath();x.roundRect(ox,oy,(x1-x0)*sc,(y1-y0)*sc,6);x.fill();
  x.strokeStyle='rgba(255,255,255,.07)';x.lineWidth=1;
  for(let g=Math.ceil(x0/5)*5;g<x1;g+=5){x.beginPath();x.moveTo(X(g),oy);x.lineTo(X(g),oy+(y1-y0)*sc);x.stroke()}
  for(let g=Math.ceil(y0/5)*5;g<y1;g+=5){x.beginPath();x.moveTo(ox,Y(g));x.lineTo(ox+(x1-x0)*sc,Y(g));x.stroke()}
  const active=CAMRES[CAMIDX]&&CAMRES[CAMIDX].cam;
  D_.cams.forEach((cm,i)=>{const col=SITE_COLORS[i%SITE_COLORS.length],on=cm.id===active;
    x.beginPath();cm.fov.forEach((p,k)=>k?x.lineTo(X(p[0]),Y(p[1])):x.moveTo(X(p[0]),Y(p[1])));x.closePath();
    x.globalAlpha=on?.28:.13;x.fillStyle=col;x.fill();x.globalAlpha=on?.9:.45;x.strokeStyle=col;x.lineWidth=on?2:1;x.stroke();x.globalAlpha=1;
    x.fillStyle=col;x.beginPath();x.arc(X(cm.center[0]),Y(cm.center[1]),6,0,7);x.fill();
    x.fillStyle='#fff';x.font='600 11px Segoe UI';x.textAlign='left';x.fillText(cm.id,X(cm.center[0])+9,Y(cm.center[1])+4)});
  const k=siteFrameAt(t);let n=0;
  if(k>=0){const cur=D_.frames[k].people;n=cur.length;
    if(S.mlayers.trails){const t0=t-6;for(const p of cur){x.strokeStyle='rgba(61,220,151,.55)';x.lineWidth=2;x.beginPath();let st=false;
      for(let j=k;j>=0&&D_.frames[j].t>=t0;j--){const q=D_.frames[j].people.find(r=>r.id===p.id);if(!q)continue;st?x.lineTo(X(q.x),Y(q.y)):(x.moveTo(X(q.x),Y(q.y)),st=true)}x.stroke()}}
    for(const p of cur){x.fillStyle='#3DDC97';x.beginPath();x.arc(X(p.x),Y(p.y),5.5,0,7);x.fill();x.strokeStyle='#0B0F13';x.lineWidth=1.5;x.stroke();
      x.fillStyle='#fff';x.font='600 11px Segoe UI';x.fillText('#'+p.id,X(p.x)+8,Y(p.y)+4)}}
  MAP.shown=n}

/* ---------- Upload ---------- */
function setFile(f){if(!f)return;S.file=f;$('#src').value=f.name;
  S.videoURL=URL.createObjectURL(f);vid.src=S.videoURL;vid.hidden=false;$('#vmsg').hidden=true;
  resetData();S.srcName=f.name;setStatus('Đã chọn video');renderKpis();logLine('Đã chọn video: '+f.name)}
$('#uploadBtn').onclick=()=>$('#file').click();
$('#file').onchange=e=>{const fs=[...e.target.files];S.files=fs;CAMRES=[];CAMIDX=-1;renderCamTabs();
  if(fs.length>1){S.file=fs[0];$('#src').value=`${fs.length} video: ${fs.map(f=>f.name).join(', ')}`;S.videoURL=URL.createObjectURL(fs[0]);vid.src=S.videoURL;vid.hidden=false;$('#vmsg').hidden=true;
    resetData();S.srcName=`${fs.length} camera`;setStatus(`Đã chọn ${fs.length} video`);renderKpis();logLine(`Đã chọn ${fs.length} video: mỗi video chạy như một camera, kết quả hiện ở các tab CAM 1, CAM 2...`)}
  else setFile(fs[0])};
$('#src').oninput=()=>{S.file=null;S.files=[]};
$('#runBtn').onclick=startAnalysis;
$('#stopBtn').onclick=()=>{S.cancel=true;logLine('Đang dừng…')};

/* ---------- Điều khiển giao diện ---------- */
function setView(v){['dash','set'].forEach(k=>$('#v-'+k).hidden=k!==v);$$('#tabs .tab').forEach(b=>b.classList.toggle('on',b.dataset.view===v))}
$('#tabs').onclick=e=>{const b=e.target.closest('[data-view]');if(b)setView(b.dataset.view)};
$('#beChip').onclick=()=>setView('set');
$('#refreshBtn').onclick=()=>API.ping();
$('#play').onclick=()=>{if(S.t>=D-.05)seek(0);setPlaying(!S.playing)};
$('#speed').onchange=e=>{S.speed=+e.target.value;vid.playbackRate=S.speed};
$('#layers').onclick=e=>{const b=e.target.closest('[data-l]');if(!b)return;S.layers[b.dataset.l]=!S.layers[b.dataset.l];b.classList.toggle('on')};
$('#mlayers').onclick=e=>{const b=e.target.closest('[data-m]');if(!b)return;S.mlayers[b.dataset.m]=!S.mlayers[b.dataset.m];b.classList.toggle('on')};
$('#tl').onclick=e=>{if(e.target.dataset.seek)return seek(+e.target.dataset.seek);const r=$('#tl').getBoundingClientRect();seek((e.clientX-r.left)/r.width*D)};
$('#filters').onclick=e=>{const b=e.target.closest('[data-f]');if(!b)return;S.filter=b.dataset.f;$$('#filters .tg').forEach(x=>x.classList.toggle('on',x===b));renderAlerts()};
$('#alerts').onclick=e=>{const sk=e.target.closest('[data-seek]');if(sk){seek(+sk.dataset.seek);return}
  const rv=e.target.closest('[data-rv]');if(rv){const ev=DATA.events.find(x=>x.id==rv.dataset.id);if(ev){ev.review=rv.dataset.rv;renderAlerts()}}};
$('#csvBtn').onclick=()=>{
  const rows=[['id','bat_dau_s','ket_thuc_s','khach','hien_vat','hanh_vi','quyet_dinh','tin_cay','ghi_chu_vlm']];
  DATA.events.forEach(e=>{const a=DATA.arts.find(a=>a.id===e.art);rows.push([e.id,e.start.toFixed(2),e.end.toFixed(2),e.track,a?a.name:e.art,e.behavior,e.decision,e.conf==null?'':e.conf,e.reason])});
  const csv='﻿'+rows.map(r=>r.map(v=>`"${String(v).replace(/"/g,'""')}"`).join(',')).join('\r\n');
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));a.download='exhibitflow_alerts.csv';a.click()};
document.addEventListener('keydown',e=>{if(e.code==='Space'&&!$('#v-dash').hidden&&!/SELECT|INPUT|BUTTON|TEXTAREA/.test(e.target.tagName)){e.preventDefault();$('#play').click()}});

/* ---------- Settings: kết nối, nhập kết quả ---------- */
$('#beUrl').value=API.base;
$('#beBtn').onclick=async()=>{API.base=$('#beUrl').value.trim().replace(/\/+$/,'');try{localStorage.setItem('ef_api',API.base)}catch{}await API.ping()};
$('#beOff').onclick=async()=>{API.base='';$('#beUrl').value='';try{localStorage.removeItem('ef_api')}catch{}await API.ping()};
$('#conf').oninput=e=>$('#confV').textContent=(+e.target.value).toFixed(2);

/* ---------- Camera (Realtime) ---------- */
let CAMS=(()=>{try{const v=JSON.parse(localStorage.getItem('ef_cams')||'null');if(Array.isArray(v)&&v.length)return v}catch{}return[{name:'CAM 1 · Sảnh chính',url:''},{name:'CAM 2 · Phòng gốm',url:''}]})();
const saveCams=()=>{try{localStorage.setItem('ef_cams',JSON.stringify(CAMS))}catch{}};
const camMode=()=>S.mode==='rt';          // Realtime = camera
function camChoice(){const v=$('#camSel').value;if(v==='__url')return{name:'Camera (URL)',url:$('#camUrl').value.trim()};return CAMS[+v.slice(1)]||{name:'Camera',url:''}}
function renderCams(){
  const cur=$('#camSel').value;
  $('#camSel').innerHTML=CAMS.map((c,i)=>`<option value="c${i}">${esc(c.name)}</option>`).join('')+'<option value="__url">Nhập URL khác…</option>';
  if([...$('#camSel').options].some(o=>o.value===cur))$('#camSel').value=cur;
  $('#camList').innerHTML=CAMS.map((c,i)=>`<div class="camrow"><b>${esc(c.name)}</b><small>${esc(c.url||'chưa có URL')}</small><button class="btn sm" data-del="${i}">Xóa</button></div>`).join('')||'<p class="note">Chưa có camera.</p>';
  applySourceUI()}
function applySourceUI(){
  const rt=S.mode==='rt',cam=camMode();
  $('#srcFile').hidden=cam;$('#srcCam').hidden=!cam;$('#camUrl').hidden=!(cam&&$('#camSel').value==='__url');
  $('#runBtn').textContent=rt?'Bắt đầu':'Phân tích';
  $('#vtitle').textContent=cam?`Camera realtime — ${camChoice().name}`:'Video phân tích — CAM 1'}
$('#camSel').onchange=applySourceUI;
$('#camAdd').onclick=()=>{const name=$('#camName').value.trim(),url=$('#camAddUrl').value.trim();if(!name)return;CAMS.push({name,url});saveCams();$('#camName').value='';$('#camAddUrl').value='';renderCams()};
$('#camList').onclick=e=>{const b=e.target.closest('[data-del]');if(!b)return;CAMS.splice(+b.dataset.del,1);saveCams();renderCams()};

/* ---------- Khởi động ---------- */
resetData();renderSystem();setStatus('Chưa chạy');renderCams();setMode(S.mode);
logLine('Sẵn sàng.'+(CFG.demo?' Chế độ demo: chưa kết nối backend thì kết quả là dữ liệu mẫu.':''));
setInterval(()=>{$('#clock').textContent=new Date().toLocaleString('vi-VN')},1000);$('#clock').textContent=new Date().toLocaleString('vi-VN');
setInterval(pollSystem,3000);
API.ping();requestAnimationFrame(tick);

/* ---------- Ngăn kéo menu ---------- */
(()=>{const app=$('.app'),btn=$('#menuBtn'),scrim=$('#scrim');
  const set=o=>{app.classList.toggle('open',o);scrim.hidden=!o;btn.setAttribute('aria-expanded',o);btn.setAttribute('aria-label',o?'Đóng menu':'Mở menu');
    if(o){const t=$('#tabs .tab.on')||$('#tabs .tab');t&&t.focus()}else if(document.activeElement&&document.activeElement.closest('.side'))btn.focus()};
  btn.onclick=()=>set(!app.classList.contains('open'));
  scrim.onclick=()=>set(false);
  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&app.classList.contains('open'))set(false)});
  $('#tabs').addEventListener('click',e=>{if(e.target.closest('[data-view]'))set(false)});
})();
