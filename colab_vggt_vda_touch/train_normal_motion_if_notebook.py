# %% [markdown]
# # ExhibitFlow — Train/evaluate normal pose-motion Isolation Forest
# Notebook độc lập để kiểm tra baseline normal-only trước khi ghép vào pipeline touch.
# YOLOv8n-Pose ước lượng pose trên bbox người trong MuseumVisitors; Isolation Forest
# học cửa sổ pose/quỹ đạo bình thường. Segment cuối mỗi camera được giữ riêng để đánh giá.
# Không cần nhãn touch/anomaly; vì vậy kết quả là kiểm tra độ ổn định + audit thủ công,
# không phải accuracy/precision/recall và không tự nhận diện tên hành vi.

# %%
import os, sys, subprocess
subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics>=8.3,<9',
                'scikit-learn','pandas','matplotlib'],check=True)
print('Cài dependency xong. Khuyến nghị Colab GPU T4/L4; không cài lại PyTorch/CUDA.')

# %% [markdown]
# ## Đưa dữ liệu vào Colab
# Đặt thư mục `MuseumVisitors` trong Google Drive với cấu trúc `MuseumVisitors/frames/1/...`,
# `frames/2/...`, `frames/3/...`. Dataset ảnh khoảng 1.4 GB, không upload qua files.upload.
# Upload hai file metadata `manifest.csv` và `annotations.jsonl` từ `Museum/pose_rule_anomaly/outputs/`.

# %%
from google.colab import drive, files
from pathlib import Path
drive.mount('/content/drive')
ROOT=Path('/content/exhibitflow_iforest_train'); ROOT.mkdir(exist_ok=True)
INPUT=ROOT/'input'; INPUT.mkdir(exist_ok=True)
REQUIRED=('manifest.csv','annotations.jsonl')
if any(not (INPUT/n).exists() for n in REQUIRED):
    uploaded=files.upload()
    for name,data in uploaded.items():
        if Path(name).name in REQUIRED: (INPUT/Path(name).name).write_bytes(data)
    del uploaded
missing=[n for n in REQUIRED if not (INPUT/n).exists()]
assert not missing, f'Thiếu metadata: {missing}'
NORMAL_ROOT=Path('/content/drive/MyDrive/MuseumVisitors')
assert (NORMAL_ROOT/'frames').is_dir(), 'Sửa NORMAL_ROOT tới thư mục MuseumVisitors có thư mục frames/.'
print('Dataset:',NORMAL_ROOT)

# %% [markdown]
# ## Trích pose normal bằng đúng YOLOv8n-Pose
# Mỗi bbox/visitor ID lấy từ annotation gốc; pose được chạy trên crop người giống bước pose
# trong notebook chính. Thời gian được dựng từ frame quan sát trong từng giây nguồn; không nối segment.

# %%
import csv, json, math, warnings
from collections import defaultdict
import cv2, numpy as np, pandas as pd, torch
from PIL import Image, ImageDraw
from IPython.display import display
from ultralytics import YOLO

WINDOW_SECONDS=2.5; N_STEPS=16; STRIDE_SECONDS=1.0
MAX_GAP_S=.75; MIN_VALID_RATIO=.80; MIN_VISIBILITY=.60; MIN_KP_CONF=.50
KEYPOINT_NAMES=['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder',
    'left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee',
    'left_ankle','right_ankle']
FEATURE_NAMES=([f'pose_{n}_{a}' for n in KEYPOINT_NAMES for a in ('x','y')]
    +[f'keypoint_confidence_{n}' for n in KEYPOINT_NAMES]
    +[f'pose_velocity_{n}_{a}' for n in KEYPOINT_NAMES for a in ('x','y')]
    +['bbox_center_x','bbox_center_y','bbox_width','bbox_height','bbox_aspect','hip_height','torso_angle',
      'bbox_velocity_x','bbox_velocity_y'])
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)

def make_frame_times(path):
    frame_times={}; mf=pd.read_csv(path)
    for (cam,seg),g in mf.groupby(['camera_id','segment_id']):
        for sec,sg in g.groupby('timestamp_source'):
            sg=sg.sort_values('frame_id'); n=len(sg)
            for rank,row in enumerate(sg.itertuples(index=False)):
                frame_times[(int(cam),str(seg),int(row.frame_id))]=float(sec)+(rank+.5)/n
    return frame_times

def pose_row_features(kp,kc,box,frame_w,frame_h):
    if kp is None or kc is None: return None
    xy=np.asarray(kp,np.float32); conf=np.asarray(kc,np.float32); box=np.asarray(box,np.float32)
    if xy.shape!=(17,2) or conf.shape!=(17,) or not np.isfinite(box).all(): return None
    x1,y1,x2,y2=box; bw=max(float(x2-x1),1.); bh=max(float(y2-y1),1.)
    visible=conf>=MIN_KP_CONF; hips=[11,12]; shoulders=[5,6]
    origin=xy[hips].mean(axis=0) if visible[hips].all() else np.array([(x1+x2)/2,(y1+y2)/2],np.float32)
    scale=max(bh,math.sqrt(bw*bh),1.); norm=(xy-origin)/scale; norm[~visible]=np.nan
    hip_height=np.nan; torso_angle=np.nan
    if visible[hips].all():
        hc=xy[hips].mean(axis=0); hip_height=(hc[1]-y1)/bh
        if visible[shoulders].all():
            vec=xy[shoulders].mean(axis=0)-hc
            torso_angle=math.atan2(float(vec[0]),float(-vec[1]))
    bbox=np.asarray([((x1+x2)/2)/frame_w,((y1+y2)/2)/frame_h,bw/frame_w,bh/frame_h,
                     bw/bh,hip_height,torso_angle],np.float32)
    return norm.reshape(-1),conf,bbox

def box_iou(a,b):
    lo=np.maximum(a[:2],b[:2]); hi=np.minimum(a[2:4],b[2:4])
    inter=np.maximum(hi-lo,0).prod()
    area=lambda q: np.maximum(q[2:4]-q[:2],0).prod()
    return float(inter/max(area(a)+area(b)-inter,1e-6))

def extract_normal_rows():
    frame_times=make_frame_times(INPUT/'manifest.csv'); grouped=defaultdict(list)
    annotation_lookup={}; frame_path_lookup={}
    with (INPUT/'annotations.jsonl').open() as f:
        for line in f:
            ann=json.loads(line); key=(int(ann['camera_id']),str(ann['segment_id']),int(ann['frame_id']))
            if key in frame_times:
                grouped[key].append(ann)
                annotation_lookup[(key[0],key[1],key[2],int(ann['visitor_id']))]=ann
    model=YOLO('yolov8n-pose.pt'); device=0 if torch.cuda.is_available() else 'cpu'
    by_track=defaultdict(list); processed=missing=0
    for key,people in sorted(grouped.items()):
        cam,seg,fid=key; rel=Path(*Path(people[0]['filename_raw']).parts[-2:])
        image_path=NORMAL_ROOT/'frames'/str(cam)/rel; frame_path_lookup[key]=image_path
        frame=cv2.imread(str(image_path))
        if frame is None: missing+=1; continue
        h,w=frame.shape[:2]; jobs=[]
        for ann in people:
            b=np.asarray([ann['bb_x'],ann['bb_y'],ann['bb_x']+ann['bb_w'],ann['bb_y']+ann['bb_h']],np.float32)
            x1,y1,x2,y2=b; dx=(x2-x1)*.20; dy=(y2-y1)*.10
            xa,ya=max(0,int(x1-dx)),max(0,int(y1-dy)); xb,yb=min(w,int(x2+dx)),min(h,int(y2+dy))
            if xb-xa<8 or yb-ya<8: continue
            jobs.append((ann,frame[ya:yb,xa:xb],np.asarray([xa,ya]),b-np.asarray([xa,ya,xa,ya]),b))
        results=model.predict([j[1] for j in jobs],device=device,imgsz=640,conf=.15,verbose=False) if jobs else []
        for (ann,crop,origin,target,b),result in zip(jobs,results):
            kp=kc=None
            if result.keypoints is not None and result.keypoints.conf is not None and len(result.boxes):
                overlaps=np.asarray([box_iou(q,target) for q in result.boxes.xyxy.cpu().numpy()])
                order=np.argsort(overlaps)[::-1]
                if overlaps[order[0]]>=.30 and (len(order)==1 or overlaps[order[0]]-overlaps[order[1]]>=.15):
                    j=order[0]; kp=result.keypoints.xy[j].cpu().numpy()+origin; kc=result.keypoints.conf[j].cpu().numpy()
            features=pose_row_features(kp,kc,b,w,h)
            if features is not None:
                tid=int(ann['visitor_id']); by_track[(cam,seg,tid)].append((frame_times[key],*features,fid))
        processed+=1
        if processed%100==0: print(f'Frames with YOLO pose: {processed}/{len(grouped)}')
    del model
    if missing: print(f'Warning: {missing} source frames could not be read from Drive.')
    return by_track,annotation_lookup,frame_path_lookup,processed

def make_windows(by_track):
    windows=[]
    for (cam,seg,tid),rows in by_track.items():
        rows.sort(key=lambda r:r[0]); times=np.asarray([r[0] for r in rows])
        if len(rows)<2: continue
        start=float(math.ceil(times[0])); last=times[-1]-WINDOW_SECONDS
        while start<=last+1e-8:
            grid=np.linspace(start,start+WINDOW_SECONDS,N_STEPS)
            pose=np.full((N_STEPS,34),np.nan,np.float32); conf=np.full((N_STEPS,17),np.nan,np.float32)
            bbox=np.full((N_STEPS,7),np.nan,np.float32); valid=np.zeros(N_STEPS,bool)
            for si,target in enumerate(grid):
                right=int(np.searchsorted(times,target,side='left'))
                if right==0 or right==len(times): continue
                left=right-1; gap=times[right]-times[left]
                if gap<=0 or gap>MAX_GAP_S: continue
                frac=(target-times[left])/gap
                for dest,a,b in ((pose,rows[left][1],rows[right][1]),
                                 (conf,rows[left][2],rows[right][2]),
                                 (bbox,rows[left][3],rows[right][3])):
                    ok=np.isfinite(a)&np.isfinite(b); dest[si,ok]=a[ok]+(b[ok]-a[ok])*frac
                valid[si]=np.isfinite(bbox[si,:5]).all()
            ratio=float(valid.mean()); kp_mask=np.isfinite(pose.reshape(N_STEPS,17,2)).all(axis=2)
            visibility=float(kp_mask[valid].mean()) if valid.any() else 0.
            if ratio>=MIN_VALID_RATIO and visibility>=MIN_VISIBILITY:
                dt=WINDOW_SECONDS/(N_STEPS-1); pv=np.full_like(pose,np.nan); pv[1:]=(pose[1:]-pose[:-1])/dt
                bv=np.full((N_STEPS,2),np.nan,np.float32); bv[1:]=(bbox[1:,:2]-bbox[:-1,:2])/dt
                feats=np.concatenate([pose,conf,pv,bbox,bv],axis=1).astype(np.float32)
                frame_ids=[r[4] for r in rows if len(r)>4 and start<=r[0]<=start+WINDOW_SECONDS]
                windows.append({'features':feats,'camera_id':int(cam),'segment_id':str(seg),'track_id':int(tid),
                    'start_s':start,'end_s':start+WINDOW_SECONDS,
                    'start_frame':int(min(frame_ids)) if frame_ids else None,
                    'end_frame':int(max(frame_ids)) if frame_ids else None,
                    'valid_step_ratio':ratio,'keypoint_visibility':visibility})
            start+=STRIDE_SECONDS
    return windows

by_track,ANNOTATIONS,FRAME_PATHS,PROCESSED_FRAMES=extract_normal_rows()
WINDOWS=make_windows(by_track)
assert WINDOWS, 'Không tạo được windows: kiểm tra độ dài tracks/pose và cấu trúc dataset.'
X=np.stack([w['features'] for w in WINDOWS])
META=[{k:v for k,v in w.items() if k!='features'} for w in WINDOWS]
np.savez_compressed(OUT/'normal_yolov8_pose_windows.npz',features=X,
    metadata_json=np.asarray([json.dumps(m) for m in META]),feature_names=np.asarray(FEATURE_NAMES))
print('Windows:',X.shape,'| cameras:',sorted({m['camera_id'] for m in META}),
      '| tracks:',len(by_track),'| pose-processed frames:',PROCESSED_FRAMES)

# %% [markdown]
# ## Train Isolation Forest + quantitative holdout checks
# Segment cuối theo thứ tự tên ở từng camera là validation; không có cửa sổ từ segment đó trong fit.
# Ngưỡng percentile được dựng từ train pool của chính camera. Tỷ lệ vượt ngưỡng trong normal holdout
# là chỉ báo drift, không phải false-positive rate đã xác nhận bằng nhãn.

# %%
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from IPython.display import display

def summarize(X):
    keep=np.asarray([not n.startswith('keypoint_confidence_') for n in FEATURE_NAMES]); q=X[:,:,keep]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.concatenate([np.nanmedian(q,axis=1),np.nanpercentile(q,75,axis=1)-np.nanpercentile(q,25,axis=1),
                              np.nanmin(q,axis=1),np.nanmax(q,axis=1)],axis=1).astype(np.float32)

eligible=np.asarray([m['valid_step_ratio']>=MIN_VALID_RATIO and m['keypoint_visibility']>=MIN_VISIBILITY for m in META])
cameras=sorted({int(m['camera_id']) for m in META})
held={c:sorted({m['segment_id'] for m in META if int(m['camera_id'])==c})[-1] for c in cameras}
held_mask=np.asarray([m['segment_id']==held[int(m['camera_id'])] for m in META])
validation=held_mask&eligible; train=(~held_mask)&eligible
assert validation.any() and train.any(), 'Cần normal windows ở train và segment holdout.'
embedding=summarize(X); rng=np.random.default_rng(42); fit_idx=[]
for c in cameras:
    ids=np.flatnonzero(train&np.asarray([int(m['camera_id'])==c for m in META]))
    assert len(ids)>0, f'Camera {c} không có train windows.'
    cap=min(2500,min(int(np.sum(train&np.asarray([int(m['camera_id'])==q for m in META]))) for q in cameras))
    fit_idx.extend(rng.choice(ids,cap,replace=False).tolist() if len(ids)>cap else ids.tolist())
model=Pipeline([('imputer',SimpleImputer(strategy='median')),
    ('scaler',RobustScaler(quantile_range=(10,90))),
    ('iforest',IsolationForest(n_estimators=300,contamination='auto',random_state=42,n_jobs=-1))])
model.fit(embedding[fit_idx])
raw=np.full(len(X),np.nan); raw[eligible]=-model.decision_function(embedding[eligible])
percentile=np.full(len(X),np.nan); reference={}; metrics=[]
for c in cameras:
    cam=np.asarray([int(m['camera_id'])==c for m in META]); ref=raw[train&cam]; reference[c]=ref.astype(np.float32)
    percentile[eligible&cam]=np.searchsorted(np.sort(ref),raw[eligible&cam],side='right')/max(len(ref),1)
    val=validation&cam; p=percentile[val]
    metrics.append({'camera_id':c,'held_out_segment':held[c],'validation_windows':int(val.sum()),
        'median_percentile':float(np.median(p)) if len(p) else None,
        'over_p95_rate':float(np.mean(p>=.95)) if len(p) else None,
        'over_p99_rate':float(np.mean(p>=.99)) if len(p) else None})
EVALUATION=pd.DataFrame(metrics)
display(EVALUATION)
fig,ax=plt.subplots(1,2,figsize=(13,4))
for c in cameras:
    cam=eligible&np.asarray([int(m['camera_id'])==c for m in META]); val=validation&cam
    ax[0].hist(raw[cam],bins=35,alpha=.5,label=f'Cam {c}')
    ax[1].hist(percentile[val],bins=np.linspace(0,1,21),alpha=.5,label=f'Cam {c} holdout')
ax[0].set(title='Raw anomaly scores (higher = stranger)',xlabel='score',ylabel='windows')
ax[1].axvline(.95,color='orange',linestyle='--'); ax[1].axvline(.99,color='red',linestyle='--')
ax[1].set(title='Held-out normal percentiles',xlabel='percentile vs same-camera train pool',ylabel='windows')
for a in ax:a.legend()
fig.tight_layout(); fig.savefig(OUT/'iforest_evaluation.png',dpi=150); plt.show()

SCORES=pd.DataFrame([{**m,'anomaly_score':float(raw[i]),'camera_percentile':float(percentile[i]),
    'split':'validation' if validation[i] else 'train_pool'} for i,m in enumerate(META) if eligible[i]])
SCORES.to_csv(OUT/'normal_window_scores.csv',index=False)
top=SCORES[SCORES.split=='validation'].sort_values('camera_percentile',ascending=False)
for c in cameras:
    print(f'Camera {c}: các cửa sổ normal holdout score cao nhất — mở ảnh để audit lỗi pose/ID/occlusion/domain drift')
    display(top[top.camera_id==c].head(10)[['segment_id','track_id','start_frame','end_frame','camera_percentile']])

# %% [markdown]
# ## Audit trực quan các outlier normal
# Hiện frame giữa cửa sổ cho tối đa ba outlier cao nhất mỗi camera. Đây là bước quan trọng:
# nếu score cao do pose sai, che khuất hoặc tracking/box lỗi thì model chưa sẵn sàng ghép vào touch.

# %%
def show_audit_frame(row):
    key=(int(row.camera_id),str(row.segment_id),int(row.start_frame))
    path=FRAME_PATHS.get(key)
    ann=ANNOTATIONS.get((*key,int(row.track_id)))
    if path is None or ann is None: return
    frame=cv2.imread(str(path))
    if frame is None:return
    x,y,w,h=[int(ann[k]) for k in ('bb_x','bb_y','bb_w','bb_h')]
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.putText(frame,f"ID {row.track_id} | p={row.camera_percentile:.3f} | normal holdout",
        (max(0,x),max(18,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),2)
    display(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
for c in cameras:
    print('Audit camera',c)
    for _,row in top[top.camera_id==c].head(3).iterrows(): show_audit_frame(row)

# %% [markdown]
# ## Lưu model, evaluation và bảng audit
# Điền `audit_label`/`audit_notes` trong CSV sau khi xem; nhãn này chỉ phục vụ audit thủ công,
# không biến kết quả hiện tại thành benchmark đầy đủ. Model chỉ tạo unusual-motion candidates.

# %%
import joblib, zipfile
SCORES['audit_label']=''; SCORES['audit_notes']=''
SCORES.to_csv(OUT/'normal_window_scores_for_audit.csv',index=False)
bundle={'pipeline':model,'feature_names':FEATURE_NAMES,'camera_reference_scores':reference,
        'held_out_segments':held,'window_seconds':WINDOW_SECONDS,'n_steps':N_STEPS}
joblib.dump(bundle,OUT/'isolation_forest_normal_yolov8_pose.joblib')
summary={'normal_windows':len(X),'scored_windows':int(eligible.sum()),'excluded_windows':int((~eligible).sum()),
    'fit_windows':len(fit_idx),'held_out_segments':held,'per_camera':metrics,
    'claim':'unlabeled normal holdout stability only; no abnormal-behavior accuracy'}
(OUT/'evaluation.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
archive=ROOT/'isolation_forest_normal_pose_training_results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if p.is_file():z.write(p,p.relative_to(OUT))
files.download(str(archive))
print('Audit: mở normal_window_scores_for_audit.csv; model:',OUT/'isolation_forest_normal_yolov8_pose.joblib')
