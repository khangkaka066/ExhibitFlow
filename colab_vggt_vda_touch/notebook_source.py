# %% [markdown]
# # ExhibitFlow — VGGT + Depth Anything V2 Metric + ByteTrack DMA/LTC
# Notebook thử nghiệm camera cố định, chạy từng clip độc lập. **Không phải phép đo touch đã hiệu chuẩn.**
#
# Detector YOLOX nano trong repo tracker + checkpoint `best_ckpt` → ByteTrack DMA/LTC → pose trên crop có ID.
# MM-Grounding-DINO large + SAM2.1 → mask hiện vật. VGGT → camera và hình học tham chiếu.
# DA-V2 Metric Indoor Small → per-frame depth → căn chỉnh trên nền tĩnh → khoảng cách cổ tay–hiện vật.
# Kết quả: `TOUCH_CANDIDATE`, `NO_TOUCH`, `UNKNOWN`, video H.264, ảnh sự kiện, CSV và báo cáo chất lượng.
#
# **Giới hạn:** pose 17 điểm chỉ có cổ tay, không có ngón tay. Candidate không xác nhận tiếp xúc.
# Depth DA-V2 được huấn luyện trên Hypersim trong nhà; mét là ước lượng chưa hiệu chuẩn tại bảo tàng.
# DA-V2 dự đoán độc lập từng frame, kiểm tra tính ổn định ở phase alignment trước khi xem touch.
# Isolation Forest normal-only đã được train/đánh giá riêng; notebook này nạp model để chấm pose.
# Camera 3 có drift trên holdout: chỉ hiện điểm review, không tự sinh unusual-motion candidate.
#
# Kaggle: bật GPU + Internet, Add Input dataset chứa 9 file:
# `1.mp4`, `2.mp4`, `3.mp4`, `best_ckpt`, `dma_gbm_mot17.gbm`, `ltc_motion_mot17.pth`,
# `isolation_forest_normal_yolov8_pose.joblib`, `evaluation.json`, `sam2.1_b.pt`.
# `best_ckpt` là checkpoint torch.save, được nạp trực tiếp bằng `torch.load`.
# Các checkpoint phải là file tin cậy của bạn. Không hạ NumPy hoặc thay PyTorch/CUDA.
# **Chế độ từng phase:** chọn một video ở phase 0; mỗi phase hiện preview/báo cáo rồi tự chạy tiếp.
# Có thể dùng Run all để chạy hết video hiện tại. Đổi video tại phase 0 sau khi lượt hiện tại hoàn tất.

# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/Kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')

# %%
import os, sys, subprocess
from pathlib import Path

ROOT = Path('/kaggle/working/exhibit_vggt_depth_anything_v2')
ROOT.mkdir(parents=True,exist_ok=True)
REPOS = {
    'tracker': ('https://github.com/khangkaka066/ByteTrack-DMA-LTC-Motion-Tracker.git', '5c10c793565c77c7e4d238ebbec81397c1983309'),
    'vggt': ('https://github.com/facebookresearch/vggt.git', 'a288dd0f14786c93483e45524328726ab7b1b4ce'),
    'depth_anything_v2': ('https://github.com/DepthAnything/Depth-Anything-V2.git', 'a561b849ebae10a6f5ef49e26c83cbbcd36c71bf'),
}
for name, (url, revision) in REPOS.items():
    dest = ROOT / name
    if not (dest / '.git').exists():
        subprocess.run(['git', 'clone', '--filter=blob:none', url, str(dest)], check=True)
    subprocess.run(['git', '-C', str(dest), 'checkout', revision], check=True)

# Không install requirements cũ của YOLOX: có thể kéo NumPy/PyTorch không phù hợp Kaggle.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'ultralytics>=8.3,<9', 'transformers>=4.46,<5', 'accelerate', 'huggingface_hub', 'safetensors',
    'timm', 'einops', 'scipy', 'lap', 'cython_bbox', 'lightgbm', 'gdown', 'loguru',
    'pycocotools', 'thop', 'tabulate', 'tensorboard', 'yacs', 'ftfy', 'regex',
    'xgboost', 'scikit-learn', 'matplotlib', 'pandas', 'Pillow', 'six',
    'imageio-ffmpeg', 'hydra-core', 'iopath', 'easydict'], check=True)
os.environ['XFORMERS_DISABLED'] = '1'
print('Cài đặt xong. Nếu runtime trước đó từng đổi NumPy, hãy dùng runtime mới.')

# %% [markdown]
# ## Đọc Kaggle Input và cấu hình
# Ngưỡng dưới đây chỉ là điểm bắt đầu để review, không phải ngưỡng tiếp xúc vật lý.
# `MAX_WRIST_GAP_M` áp dụng cho cổ tay nên không được gọi là khoảng cách ngón tay.
# Notebook giới hạn clip ngắn để không hết RAM. Không âm thầm cắt video.

# %%
REQUIRED = ('1.mp4', '2.mp4', '3.mp4', 'best_ckpt', 'dma_gbm_mot17.gbm', 'ltc_motion_mot17.pth',
            'isolation_forest_normal_yolov8_pose.joblib', 'evaluation.json', 'sam2.1_b.pt')
# Kaggle: Notebook → Add Input → gắn Dataset có kaggle_upload_bundle.zip (hoặc 9 file đã giải nén).
# Nếu dataset có tên/slug khác, có thể điền chính xác thư mục ở đây.
DATASET_DIR = Path('/kaggle/input/datasets/nguyenvohoangkhang/anything1')
# Kaggle dataset standalone override. This path may point to a torch.save file
# without a .zip extension; the loader uses file contents, not the filename suffix.
INPUT_OVERRIDES = {
    'best_ckpt': Path('/kaggle/input/datasets/nguyenvohoangkhang/anything1/best_ckpt'),
}
def repack_extracted_torch_checkpoint(folder):
    """Rebuild a torch.save ZIP checkpoint that Kaggle expanded as a directory."""
    import zipfile
    data_pkls=[p for p in folder.rglob('data.pkl') if p.is_file()]
    if len(data_pkls)!=1:
        raise RuntimeError(f'Checkpoint đã bung phải có đúng một data.pkl; thấy {len(data_pkls)} trong {folder}')
    destination=ROOT/'repacked_best_ckpt.zip'
    files=sorted(p for p in folder.rglob('*') if p.is_file())
    if not files:
        raise RuntimeError(f'Thư mục checkpoint rỗng: {folder}')
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_STORED) as archive:
        for path in files:
            archive.write(path,arcname=path.relative_to(folder).as_posix())
    print(f'Đóng gói lại checkpoint đã bung: {len(files)} files → {destination}')
    return destination
KAGGLE_INPUT_ROOT=Path('/kaggle/input')
BUNDLE_PATHS=sorted(KAGGLE_INPUT_ROOT.rglob('kaggle_upload_bundle.zip')) if KAGGLE_INPUT_ROOT.exists() else []
if len(BUNDLE_PATHS)>1:
    raise RuntimeError(f'Tìm thấy nhiều kaggle_upload_bundle.zip: {BUNDLE_PATHS}. Chỉ gắn một dataset bundle.')
if BUNDLE_PATHS:
    import zipfile
    INPUT_ROOT=ROOT/'kaggle_input'; INPUT_ROOT.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(BUNDLE_PATHS[0]) as bundle:
        extracted=set()
        for member in bundle.infolist():
            member_path=Path(member.filename)
            if member_path.is_absolute() or '..' in member_path.parts:
                raise RuntimeError(f'Đường dẫn không an toàn trong input ZIP: {member.filename}')
            logical_name='best_ckpt' if member_path.name=='best_ckpt.zip' else member_path.name
            if member.is_dir() or logical_name not in REQUIRED: continue
            extracted_path=Path(bundle.extract(member,INPUT_ROOT))
            if logical_name=='best_ckpt' and extracted_path.name=='best_ckpt.zip':
                extracted_path.rename(INPUT_ROOT/'best_ckpt')
            extracted.add(logical_name)
    missing_from_bundle=sorted(set(REQUIRED)-extracted)
    if missing_from_bundle: raise FileNotFoundError(f'ZIP input thiếu file: {missing_from_bundle}')
else:
    INPUT_ROOT=DATASET_DIR if DATASET_DIR.exists() else KAGGLE_INPUT_ROOT
def locate_input_file(name, required=True):
    override=INPUT_OVERRIDES.get(name)
    if override is not None:
        if override.is_file():
            print(f'  Dùng override {name}: {override}')
            return override
        if override.is_dir() and name=='best_ckpt':
            if any(p.is_file() for p in override.rglob('data.pkl')):
                return repack_extracted_torch_checkpoint(override)
    # Kaggle's mounted path can differ from the URL/path shown in the UI.
    # Search the configured dataset, extracted bundle, and all mounted inputs.
    roots=[]
    for root in (INPUT_ROOT, DATASET_DIR, KAGGLE_INPUT_ROOT):
        if root.exists() and root not in roots: roots.append(root)
    matches=sorted({p for root in roots for p in root.rglob(name) if p.is_file()})
    # Kaggle may mount the uploaded checkpoint as a directory; accept it only if
    # it contains exactly one file, and pass that file to torch.load.
    if not matches and name=='best_ckpt':
        candidates=[]
        for root in roots:
            for folder in root.rglob('best_ckpt'):
                if folder.is_dir(): candidates.extend(p for p in folder.rglob('*') if p.is_file())
        matches=sorted(set(candidates))
    if len(matches)>1:
        raise RuntimeError(f'Tìm thấy nhiều file {name}: {matches}. Đặt DATASET_DIR đúng dataset ở cell input.')
    if not matches:
        if required:
            seen=sorted(str(p) for root in roots for p in root.rglob('*best_ckpt*') if p.is_file() or p.is_dir()) if name=='best_ckpt' else []
            raise FileNotFoundError(f'Thiếu {name} (checkpoint không cần đuôi .zip). Đã tìm trong {roots}. Các đường dẫn best_ckpt nhìn thấy: {seen}')
        return None
    if name=='best_ckpt': print(f'  Tự tìm thấy detector checkpoint: {matches[0]}')
    return matches[0]
INPUT_FILES={name:locate_input_file(name) for name in REQUIRED}
# Backward-compatible alias: lets an already-defined tracking cell from an
# earlier notebook revision keep working during an interactive Kaggle session.
INPUT_FILES['best_ckpt.zip']=INPUT_FILES['best_ckpt']
print('Đã tìm thấy input files:')
for name,path in INPUT_FILES.items(): print(f'  {name}: {path}')

CFG = dict(
    width=640, max_frames=900, max_seconds=40, reference_frames=4,
    da_v2_input_size=392, detector_size=[608,1088], detector_preprocess='bgr255',
    wrist_conf=0.55, max_image_gap_px=12, max_wrist_gap_m=0.15,
    hold_seconds=0.12, history_seconds=0.25, max_hand_speed_m_s=0.8,
    min_alignment_pixels=800, max_alignment_median_rel=0.08,
    max_alignment_p90_rel=0.20, max_scale_drift=0.25,
    max_surface_gap_px=12, min_surface_pixels=40,
)
print('Input sẵn sàng. Isolation Forest chỉ tạo unusual-motion candidate, không tự xác nhận touch.')

# %%
import gc, json, csv, math, hashlib, platform, importlib.metadata, warnings
from collections import defaultdict, deque
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from IPython.display import display, Video

assert torch.cuda.is_available(), 'Kaggle: Notebook Settings → Accelerator → GPU.'
DEVICE = 'cuda'
DTYPE = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
torch.set_autocast_dtype('cuda', DTYPE)
sys.path[:0] = [str(ROOT/'tracker'), str(ROOT/'vggt'), str(ROOT/'depth_anything_v2'/'metric_depth')]
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)

# Checkpoint cũ chứa scalar/array NumPy: chỉ allowlist kiểu dữ liệu, không tắt weights_only toàn cục.
multiarray = np._core.multiarray if hasattr(np, '_core') else np.core.multiarray
safe_numpy = [np.dtype, np.ndarray,
    (multiarray.scalar, 'numpy.core.multiarray.scalar'),
    (multiarray._reconstruct, 'numpy.core.multiarray._reconstruct')]
safe_numpy += [type(np.dtype(t)) for t in (np.float32,np.float64,np.int32,np.int64)]
torch.serialization.add_safe_globals(safe_numpy)

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))

def write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read_clip(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'OpenCV không đọc được {path}; kiểm tra file upload/codec.')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError('Không đọc được FPS, không tự giả định để tránh sai thời gian.')
    frames = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            w = CFG['width']; h = round(bgr.shape[0] * w / bgr.shape[1] / 2) * 2
            frames.append(cv2.resize(bgr, (w,h)))
            if len(frames) > min(CFG['max_frames'], int(CFG['max_seconds'] * fps)):
                raise RuntimeError('Clip vượt giới hạn RAM; chia clip ngắn rồi chạy, không tự cắt.')
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f'{path}: không có frame giải mã được.')
    return np.stack(frames), fps

def box_iou(a, b):
    lo = np.maximum(a[:2], b[:2]); hi = np.minimum(a[2:4], b[2:4])
    area = lambda q: np.maximum(np.asarray(q[2:4])-q[:2],0).prod()
    intersection = np.maximum(hi-lo,0).prod()
    return float(intersection / max(area(a)+area(b)-intersection,1e-6))

def foreground_mask(boxes, shape, pad=0.12):
    h,w = shape; mask = np.zeros((h,w), bool)
    for b in boxes:
        x1,y1,x2,y2 = b[:4]; dx=(x2-x1)*pad; dy=(y2-y1)*pad
        xa,ya = max(0,int(x1-dx)),max(0,int(y1-dy))
        xb,yb = min(w,int(math.ceil(x2+dx))),min(h,int(math.ceil(y2+dy)))
        mask[ya:yb,xa:xb] = True
    return mask

manifest = dict(config=CFG, repos=REPOS, gpu=torch.cuda.get_device_name(0),
    python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
    input_sha256={n:sha256(INPUT_FILES[n]) for n in REQUIRED},
    scale='Depth Anything V2 Metric estimate; NOT physically calibrated', pose='COCO17 wrist proxy',
    pipeline='YOLOX nano checkpoint → ByteTrack DMA/LTC → cropped pose; VGGT + DA-V2 Metric Indoor Small')
manifest['packages']={n:importlib.metadata.version(n) for n in
    ('ultralytics','transformers','huggingface_hub','timm','scipy','lightgbm')}
write_json(OUT/'manifest.json',manifest)
print(manifest['gpu'], DTYPE)

# %% [markdown]
# ## Nạp Isolation Forest đã train và đánh giá
# Tải checkpoint và `evaluation.json` từ thư mục `colab_iforest_normal_motion`.
# Camera 3 vượt ngưỡng drift trên normal holdout; camera đó chỉ hiện điểm tham khảo, không sinh candidate.

# %%
import warnings
from collections import Counter, defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

WINDOW_SECONDS=2.5
N_STEPS=16
MAX_POSE_GAP_S=.75
MIN_VALID_RATIO=.80
MIN_VISIBILITY=.60
MIN_KP_CONF=.50
IF_DIR=OUT/'isolation_forest'
IF_DIR.mkdir(parents=True,exist_ok=True)

def summarize_if_windows(features, feature_names):
    keep=np.asarray([not n.startswith('keypoint_confidence_') for n in feature_names])
    x=features[:,:,keep]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        med=np.nanmedian(x,axis=1)
        spread=np.nanpercentile(x,75,axis=1)-np.nanpercentile(x,25,axis=1)
        low=np.nanmin(x,axis=1); high=np.nanmax(x,axis=1)
    emb=np.concatenate([med,spread,low,high],axis=1).astype(np.float32)
    return emb

def if_percentile(reference, values):
    reference=np.sort(np.asarray(reference))
    return np.searchsorted(reference,values,side='right')/max(len(reference),1)

def train_and_evaluate_if(X,meta,names):
    X=np.asarray(X,dtype=np.float32)
    assert X.ndim==3 and X.shape[1:]==(16,94), f'Không đúng schema pose windows: {X.shape}'
    assert X.shape[0]==len(meta) and X.shape[2]==len(names)
    if np.isinf(X).any(): raise ValueError('Feature normal có infinity; dừng trước khi fit.')
    eligible=np.asarray([m.get('valid_step_ratio',0)>=MIN_VALID_RATIO and
                         m.get('keypoint_visibility',0)>=MIN_VISIBILITY for m in meta])
    cameras=sorted({int(m['camera_id']) for m in meta})
    held={cam:sorted({m['segment_id'] for m in meta if int(m['camera_id'])==cam})[-1]
          for cam in cameras}
    val=np.asarray([m['segment_id']==held[int(m['camera_id'])] for m in meta])&eligible
    train=(~np.asarray([m['segment_id']==held[int(m['camera_id'])] for m in meta]))&eligible
    if not val.any() or not train.any():
        raise ValueError('Cần nhiều hơn một segment/camera để đánh giá holdout theo segment.')
    emb=summarize_if_windows(X,names)
    rng=np.random.default_rng(42); chosen=[]
    bycam={c:np.flatnonzero(train&np.asarray([int(m['camera_id'])==c for m in meta])) for c in cameras}
    if any(len(v)==0 for v in bycam.values()): raise ValueError('Có camera không có train windows.')
    cap=min(2500,min(map(len,bycam.values())))
    for c,idx in bycam.items(): chosen.extend(rng.choice(idx,cap,replace=False).tolist() if len(idx)>cap else idx.tolist())
    model=Pipeline([('imputer',SimpleImputer(strategy='median')),
                    ('scaler',RobustScaler(quantile_range=(10,90))),
                    ('iforest',IsolationForest(n_estimators=300,contamination='auto',
                                               random_state=42,n_jobs=-1))])
    model.fit(emb[chosen])
    raw=-model.decision_function(emb[eligible])
    all_scores=np.full(len(X),np.nan); all_scores[eligible]=raw
    pct=np.full(len(X),np.nan)
    for c in cameras:
        cam=np.asarray([int(m['camera_id'])==c for m in meta])
        ref=all_scores[cam&train]
        pct[cam&eligible]=if_percentile(ref,all_scores[cam&eligible])
    rows=[]; metrics=[]
    for i,m in enumerate(meta):
        if eligible[i]:
            rows.append({**m,'anomaly_score':float(all_scores[i]),'camera_percentile':float(pct[i]),
                         'split':'validation' if val[i] else ('train_pool' if train[i] else 'excluded')})
    for c in cameras:
        idx=val&np.asarray([int(m['camera_id'])==c for m in meta])
        v=pct[idx]
        metrics.append({'camera_id':c,'validation_windows':int(idx.sum()),
                        'validation_median_percentile':float(np.nanmedian(v)) if len(v) else None,
                        'validation_over_p95_rate':float(np.mean(v>=.95)) if len(v) else None,
                        'validation_over_p99_rate':float(np.mean(v>=.99)) if len(v) else None,
                        'held_out_segment':held[c]})
    bundle={'pipeline':model,'source_feature_names':names,'camera_reference_scores':{
        c:all_scores[train&np.asarray([int(m['camera_id'])==c for m in meta])].astype(np.float32)
        for c in cameras},'held_out_segments':held,'window_seconds':WINDOW_SECONDS,
        'n_steps':N_STEPS,'min_valid_ratio':MIN_VALID_RATIO,'min_visibility':MIN_VISIBILITY}
    joblib.dump(bundle,IF_DIR/'isolation_forest_colab.joblib')
    pd.DataFrame(rows).to_csv(IF_DIR/'normal_window_scores.csv',index=False)
    write_json(IF_DIR/'evaluation.json',{'windows':len(X),'eligible':int(eligible.sum()),
        'excluded':int((~eligible).sum()),'fit_windows':len(chosen),'held_out_segments':held,
        'per_camera':metrics,'evaluation_type':'unlabeled held-out normal stability; not accuracy'})
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for c in cameras:
        ci=eligible&np.asarray([int(m['camera_id'])==c for m in meta])
        axes[0].hist(all_scores[ci],bins=35,alpha=.5,label=f'Cam {c} all scored')
        vi=val&np.asarray([int(m['camera_id'])==c for m in meta])
        axes[1].hist(pct[vi],bins=np.linspace(0,1,21),alpha=.55,label=f'Cam {c} held-out')
    axes[0].set(title='Isolation Forest raw scores',xlabel='higher = more unusual',ylabel='windows')
    axes[1].axvline(.95,color='orange',linestyle='--',label='p95 review threshold')
    axes[1].axvline(.99,color='red',linestyle='--',label='p99 review threshold')
    axes[1].set(title='Held-out normal percentile by camera',xlabel='percentile vs camera train pool',ylabel='windows')
    for ax in axes: ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(IF_DIR/'iforest_evaluation.png',dpi=150); plt.show(); plt.close(fig)
    display(pd.DataFrame(metrics))
    top=pd.DataFrame(rows).query("split == 'validation'").sort_values('camera_percentile',ascending=False)
    for c in cameras:
        print(f'Camera {c}: top held-out windows to audit')
        display(top[top.camera_id==c].head(10)[['segment_id','track_id','start_s','end_s','camera_percentile']])
    drift=[m for m in metrics if (m['validation_over_p95_rate'] or 0)>.10 or (m['validation_over_p99_rate'] or 0)>.04]
    if drift:
        print('CẢNH BÁO: held-out normal có tỷ lệ outlier cao; model/segment có drift, cần audit trước khi tin score:',drift)
    else:
        print('Holdout score ổn định sơ bộ theo các ngưỡng kiểm tra; vẫn cần audit hình ảnh top windows.')
    print('Không có ground truth bất thường: không báo accuracy, precision hay recall.')
    return bundle,metrics

IF_BUNDLE=joblib.load(INPUT_FILES['isolation_forest_normal_yolov8_pose.joblib'])
IF_EVALUATION=json.loads(INPUT_FILES['evaluation.json'].read_text())
IF_FEATURE_NAMES=IF_BUNDLE.get('source_feature_names',IF_BUNDLE.get('feature_names'))
assert IF_FEATURE_NAMES and len(IF_FEATURE_NAMES)==94, 'Checkpoint Isolation Forest không đúng schema pose.'
IF_UNSTABLE_CAMERAS={int(m['camera_id']) for m in IF_EVALUATION.get('per_camera',[])
    if (m.get('over_p95_rate',m.get('validation_over_p95_rate',0)) or 0)>.10
    or (m.get('over_p99_rate',m.get('validation_over_p99_rate',0)) or 0)>.04}
manifest['isolation_forest']={'model': 'normal_yolov8_pose',
    'model_sha256':sha256(INPUT_FILES['isolation_forest_normal_yolov8_pose.joblib']),
    'evaluation':IF_EVALUATION,'unstable_cameras':sorted(IF_UNSTABLE_CAMERAS)}
write_json(OUT/'manifest.json',manifest)
print('Isolation Forest loaded. Holdout drift cameras:',sorted(IF_UNSTABLE_CAMERAS),
      '| those cameras remain review-only.')

# %% [markdown]
# ## 1. Detector nano của repo → DMA/LTC → pose theo ID
# Nạp checkpoint `best_ckpt` strict=True; sai kiến trúc sẽ dừng, không fallback sang detector khác.
# Confidence detector 0.10 giữ cả detections thấp cho bước association thứ hai của ByteTrack.
# Pose YOLOv8n-pose chỉ chạy **sau tracking** trên crop. Crop lẫn nhiều người mơ hồ → bỏ pose.
# Checkpoint hiện có đã smoke-test: BGR 0–255 hoạt động; tiền xử lý ImageNet legacy trong demo repo cho 0 detections.
# Chọn rõ `detector_preprocess='bgr255'`; không tự dò/thay đổi chế độ theo số detection lúc chạy.

# %%
def detector_preprocess(frame, size, mode):
    ih,iw=size; h,w=frame.shape[:2]; ratio=min(ih/h,iw/w)
    image=np.full((ih,iw,3),114,np.float32)
    image[:int(h*ratio),:int(w*ratio)]=cv2.resize(frame,(int(w*ratio),int(h*ratio)))
    if mode=='legacy_rgb_imagenet':
        image=(image[:,:,::-1]/255.-np.array([.485,.456,.406]))/np.array([.229,.224,.225])
    elif mode!='bgr255':
        raise ValueError('detector_preprocess phải là bgr255 hoặc legacy_rgb_imagenet.')
    return np.ascontiguousarray(image.transpose(2,0,1),dtype=np.float32),ratio

def track_people(frames, fps, folder):
    from yolox.exp import get_exp
    from yolox.utils import postprocess
    from yolox.tracker.byte_tracker import BYTETracker
    from yolox.tracker.basetrack import BaseTrack
    from yolox.tracker import reid as reid_module
    import gdown

    exp = get_exp(str(ROOT/'tracker/exps/example/mot/yolox_nano_mix_det.py'), None)
    exp.test_size = tuple(CFG['detector_size'])
    detector = exp.get_model()
    payload = torch.load(INPUT_FILES['best_ckpt'],map_location='cpu',weights_only=True)
    state = payload.get('model',payload)
    state = {k.removeprefix('module.'):v for k,v in state.items()}
    detector.load_state_dict(state,strict=True)
    del state,payload
    detector = detector.to(DEVICE).eval()
    reid_path = ROOT/'osnet_x1_0_imagenet.pth'
    if not reid_path.exists():
        gdown.download(id='1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY',output=str(reid_path),quiet=False)
    assert reid_path.exists() and reid_path.stat().st_size > 1_000_000, 'Tải ReID thất bại.'
    args = SimpleNamespace(track_thresh=.25,track_buffer=45,match_thresh=.8,mot20=False,
        ltc_motion_ckpt=str(INPUT_FILES['ltc_motion_mot17.pth']),ltc_device=DEVICE,
        ltc_history_len=16,ltc_input_dim=12,ltc_min_history=8,ltc_hidden_size=128,
        ltc_num_layers=2,ltc_covariance_scale=1.,ltc_max_abs_residual=256.,
        with_reid=True,reid_device=DEVICE,reid_weight=.35,reid_thresh=.7,reid_alpha=.9,
        reid_backend='deep',reid_model='osnet_x1_0',reid_model_path=str(reid_path),
        ml='gbm',ml_weights=str(INPUT_FILES['dma_gbm_mot17.gbm']),dma_device=DEVICE)
    BaseTrack._count = 0
    tracker = BYTETracker(args,frame_rate=fps)
    assert tracker.dma is not None and tracker.motion_predictor.enabled and tracker.reid_extractor is not None
    all_tracks, all_dets = [], []
    h,w = frames.shape[1:3]
    for fi, frame in enumerate(frames):
        inp,ratio = detector_preprocess(frame,exp.test_size,CFG['detector_preprocess'])
        with torch.inference_mode():
            raw = detector(torch.from_numpy(inp).unsqueeze(0).float().to(DEVICE))
            output = postprocess(raw,exp.num_classes,.10,.7)[0]
        dets = np.empty((0,5),np.float32)
        if output is not None:
            o=output.detach().cpu().numpy()
            dets=np.column_stack([o[:,:4]/ratio,o[:,4]*o[:,5]]).astype(np.float32)
            dets[:,[0,2]]=np.clip(dets[:,[0,2]],0,w-1)
            dets[:,[1,3]]=np.clip(dets[:,[1,3]],0,h-1)
        all_dets.append(dets.copy())
        if fi==0:
            smoke=frame.copy()
            for b in dets:
                x1,y1,x2,y2=np.rint(b[:4]).astype(int)
                cv2.rectangle(smoke,(x1,y1),(x2,y2),(0,255,255),1)
            cv2.imwrite(str(folder/'detector_smoke.jpg'),smoke)
        tracks=tracker.update(dets.copy(),(h,w),(h,w),frame=frame)
        records=[]
        if len(dets) and tracks:
            overlaps=np.array([[box_iou(d[:4],t.tlbr) for t in tracks] for d in dets])
            rr,cc=linear_sum_assignment(1-overlaps)
            for di,ti in zip(rr,cc):
                if overlaps[di,ti]<.30:
                    continue
                b=dets[di,:4]
                rec=dict(id=int(tracks[ti].track_id),box=b.tolist(),kp=None,kc=None)
                records.append(rec)
        all_tracks.append(records)
        if fi%30==0:
            print(f'Tracking: {fi+1}/{len(frames)}')
    if not any(len(d) for d in all_dets):
        raise RuntimeError('Detector không có người trong cả clip. Kiểm tra checkpoint/preprocessing.')
    write_json(folder/'tracks.json',all_tracks)
    write_json(folder/'detections.json',[d.tolist() for d in all_dets])
    write_json(folder/'tracker_report.json',dict(detector='YOLOX nano mix_det; strict checkpoint load',
        DMA=True,LTC=True,ReID=True,pose_executed=False,frames=len(frames),
        preprocessing=CFG['detector_preprocess'],detections_per_frame=[len(d) for d in all_dets],
        tracked_people_per_frame=[len(r) for r in all_tracks]))
    del detector,tracker,exp
    reid_module._REID_EXTRACTOR_CACHE.clear()
    cleanup()
    return all_tracks,all_dets

def estimate_track_poses(frames,raw_tracks,folder):
    from ultralytics import YOLO
    import copy
    all_tracks=copy.deepcopy(raw_tracks)  # không sửa kết quả phase tracking đã duyệt
    pose=YOLO('yolov8n-pose.pt')
    h,w=frames.shape[1:3]
    for fi,(frame,records) in enumerate(zip(frames,all_tracks)):
        jobs=[]
        for rec in records:
            b=np.asarray(rec['box']); x1,y1,x2,y2=b; dx=(x2-x1)*.20; dy=(y2-y1)*.10
            xa,ya=max(0,int(x1-dx)),max(0,int(y1-dy))
            xb,yb=min(w,int(x2+dx)),min(h,int(y2+dy))
            if xb-xa>=8 and yb-ya>=8:
                jobs.append((rec,frame[ya:yb,xa:xb],np.array([xa,ya]),b-np.array([xa,ya,xa,ya])))
        results=pose.predict([j[1] for j in jobs],device=0,imgsz=640,conf=.15,verbose=False) if jobs else []
        for (rec,crop,origin,target), result in zip(jobs,results):
            if result.keypoints is None or result.keypoints.conf is None or not len(result.boxes):
                continue
            pb=result.boxes.xyxy.cpu().numpy()
            scores=np.array([box_iou(b,target) for b in pb]); order=np.argsort(scores)[::-1]
            j=order[0]
            if scores[j]<.30 or (len(order)>1 and scores[j]-scores[order[1]]<.15):
                continue
            rec['kp']=(result.keypoints.xy[j].cpu().numpy()+origin).tolist()
            rec['kc']=result.keypoints.conf[j].cpu().numpy().tolist()
        if fi%30==0:
            print(f'Pose trên ID đã duyệt: {fi+1}/{len(frames)}')
    write_json(folder/'tracks_pose.json',all_tracks)
    del pose
    cleanup()
    return all_tracks

# %% [markdown]
# ## 2. Mask hiện vật tự động, chọn reference ít bị người che
# Xem `artifact_detections_review.jpg` để kiểm tra box detector, `artifact_review.jpg` để kiểm tra mask,
# và `artifact_rejected_review.jpg`/`artifact_detections.json` để phân biệt kết quả bị loại.
# Mask tự động vẫn có thể sai; quality filter không thay cho ground-truth.
# Vùng người (bbox mở rộng) bị loại khỏi bề mặt 3D; không suy diễn mặt bị che là quan sát thật.

# %%
def _fix_mmgdino_decoder_heads(model, model_id):
    """Restore untied MM-Grounding-DINO decoder heads for checkpoints that require them."""
    import copy, glob, re
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    cfg=model.config
    if cfg.model_type!='mm-grounding-dino' or (cfg.decoder_bbox_embed_share and cfg.decoder_cls_embed_share):
        return
    model.bbox_embed=torch.nn.ModuleList(copy.deepcopy(m) for m in model.bbox_embed)
    model.class_embed=torch.nn.ModuleList(copy.deepcopy(m) for m in model.class_embed)
    model.model.decoder.bbox_embed=model.bbox_embed
    model.model.decoder.class_embed=model.class_embed
    ckpt_dir=model_id if Path(model_id).is_dir() else snapshot_download(model_id,allow_patterns=['*.safetensors'])
    pattern=re.compile(r'^(?:model\.decoder\.)?(bbox_embed|class_embed)\.(\d+)\.(.+)$')
    weights={}
    for path in glob.glob(str(Path(ckpt_dir)/'*.safetensors')):
        with safe_open(path,framework='pt') as f:
            for key in f.keys():
                match=pattern.match(key)
                if match:
                    name,idx,rest=match.groups(); weights[f'{name}.{idx}.{rest}']=f.get_tensor(key)
    if not weights:
        raise RuntimeError('MM-Grounding-DINO checkpoint không có decoder head weights để khôi phục.')
    result=model.load_state_dict(weights,strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f'Unexpected MM-Grounding-DINO head keys: {result.unexpected_keys[:5]}')
    print(f'MM-Grounding-DINO decoder heads restored: {len(weights)} tensors')

SEGMENTATION_CFG=dict(
    name='mmgdino-large-pedestal.yaml',model_id='openmmlab-community/mm_grounding_dino_large_all',
    sam_model=str(INPUT_FILES['sam2.1_b.pt']),fp16=True,
    sculpture_phrases=['statue','sculpture','bust'],painting_phrases=['painting','framed artwork'],
    pedestal_phrases=['pedestal'],negative_phrases=['person','chair','bench','window','door','rope barrier','floor','wall'],
    statue_person_ratio=.4,full_thr=.30,tile_thr=.35,nms_iou=.40,negative_margin=.10,
    containment=0.,containment_min_area=.15,min_box_size=6,sculpture_max_area=.12,sculpture_max_aspect=6.,
    painting_max_area=.25,painting_max_aspect=12.,no_tiles=False,tile_cols=3,tile_region=.50,
    tile_overlap=.25,tile_max_area=.40,tile_edge_margin=4,material_rescue=True,
    rescue_max_dark=.15,rescue_min_value=140,rescue_dark_value=70,mask_polygon=True,
    polygon_epsilon=1.,polygon_min_area=20,mask_overlap=.80,pedestal_merge=True,
    keep_empty_pedestals=False,min_mask_fill=.15,min_cc_ratio=.70,min_sam_score=.50,
)

def _museum_dino_detections(model,processor,image,phrases,threshold):
    """Phrase-aware postprocess shared by full-frame and tiled inference."""
    prompt=' '.join(f'{p}.' for p in phrases)
    tokenizer=processor.tokenizer; ids=tokenizer(prompt).input_ids
    dot_id=tokenizer.convert_tokens_to_ids('.'); spans=[]; start=1
    for i,tok in enumerate(ids):
        if tok==dot_id:
            spans.append((start,i)); start=i+1
    if len(spans)!=len(phrases):
        raise RuntimeError(f'Không tách được phrase MM-GDINO: {len(spans)} spans cho {len(phrases)} câu.')
    inputs=processor(images=image,text=prompt,return_tensors='pt').to(DEVICE)
    dtype=next(model.parameters()).dtype
    if dtype!=torch.float32: inputs['pixel_values']=inputs['pixel_values'].to(dtype)
    # HF MM-Grounding-DINO may produce FP32 positional/text activations even
    # when its weights and image tensor are FP16. Autocast keeps Linear inputs
    # compatible with the half-precision weights on Kaggle GPUs.
    if DEVICE=='cuda' and dtype==torch.float16:
        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16):
            output=model(**inputs)
    else:
        with torch.inference_mode(): output=model(**inputs)
    probs=output.logits[0].float().sigmoid(); boxes=output.pred_boxes[0].float()
    phrase_scores=torch.stack([probs[:,a:b].max(dim=1).values for a,b in spans],dim=1)
    best,best_idx=phrase_scores.max(dim=1)
    sc_cols=[i for i,p in enumerate(phrases) if p in SEGMENTATION_CFG['sculpture_phrases']]
    person_col=phrases.index('person') if 'person' in phrases else None
    sc_score,sc_idx=phrase_scores[:,sc_cols].max(dim=1) if sc_cols else (best,best_idx)
    h,w=image.height,image.width; result=[]
    negative=set(SEGMENTATION_CFG['negative_phrases'])
    for j in torch.where(best>threshold)[0].tolist():
        phrase=phrases[int(best_idx[j])]; score=float(best[j]); relabeled=False
        if person_col is not None and sc_cols and phrase=='person' and float(sc_score[j])/(float(phrase_scores[j,person_col])+1e-6)>=SEGMENTATION_CFG['statue_person_ratio']:
            phrase=phrases[sc_cols[int(sc_idx[j])]]; relabeled=True
        cx,cy,bw,bh=boxes[j].cpu().numpy().tolist()
        box=[float(np.clip((cx-bw/2)*w,0,w)),float(np.clip((cy-bh/2)*h,0,h)),float(np.clip((cx+bw/2)*w,0,w)),float(np.clip((cy+bh/2)*h,0,h))]
        label='_negative' if phrase in negative else ('painting' if phrase in ('painting','framed artwork') else ('pedestal' if phrase=='pedestal' else 'sculpture'))
        result.append(dict(box=box,score=score,phrase=phrase,label=label,relabeled=relabeled))
    del inputs,output
    return result

def _museum_nms(detections,iou_threshold=None,negative_margin=None):
    iou_threshold=SEGMENTATION_CFG['nms_iou'] if iou_threshold is None else iou_threshold
    negative_margin=SEGMENTATION_CFG['negative_margin'] if negative_margin is None else negative_margin
    detections=sorted(detections,key=lambda x:x['score'],reverse=True); keep=[]
    while detections:
        best=detections.pop(0); keep.append(best); remaining=[]
        for det in detections:
            if box_iou(best['box'],det['box'])<iou_threshold:
                remaining.append(det); continue
            if best['label']=='_negative' and det['label']!='_negative' and best['score']-det['score']<negative_margin:
                remaining.append(det)
        detections=remaining
    return keep

def _mask_bbox(mask):
    yy,xx=np.where(mask)
    return [int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]

def _masks_are_duplicate(a,b,threshold=.8):
    intersection=int(np.logical_and(a,b).sum()); smaller=min(int(a.sum()),int(b.sum()))
    return smaller>0 and intersection/smaller>=threshold

def _valid_segmentation_box(item,image_w,image_h):
    x1,y1,x2,y2=item['box']; bw=x2-x1; bh=y2-y1; label=item['label']
    if bw<SEGMENTATION_CFG['min_box_size'] or bh<SEGMENTATION_CFG['min_box_size']: return False
    area=bw*bh/(image_w*image_h); aspect=max(bw/max(bh,1e-6),bh/max(bw,1e-6))
    if label in ('sculpture','pedestal'):
        return area<=SEGMENTATION_CFG['sculpture_max_area'] and aspect<=SEGMENTATION_CFG['sculpture_max_aspect']
    if label=='painting':
        return area<=SEGMENTATION_CFG['painting_max_area'] and aspect<=SEGMENTATION_CFG['painting_max_aspect']
    return True

def _stone_stats(image_bgr,mask):
    value=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2HSV)[...,2][mask]
    if value.size==0: return dict(dark_ratio=1.,value_mean=0.)
    return dict(dark_ratio=float((value<SEGMENTATION_CFG['rescue_dark_value']).mean()),value_mean=float(value.mean()))

def _looks_like_stone(stats):
    return stats['dark_ratio']<=SEGMENTATION_CFG['rescue_max_dark'] and stats['value_mean']>=SEGMENTATION_CFG['rescue_min_value']

def _mask_polygons(mask):
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); polygons=[]
    for contour in sorted(contours,key=cv2.contourArea,reverse=True):
        if cv2.contourArea(contour)<SEGMENTATION_CFG['polygon_min_area']: continue
        contour=cv2.approxPolyDP(contour,SEGMENTATION_CFG['polygon_epsilon'],True)
        if len(contour)>=3: polygons.append(contour.reshape(-1,2).tolist())
    return polygons

def _statue_on_pedestal(statue_box,pedestal_box):
    sx1,sy1,sx2,sy2=statue_box; px1,py1,px2,py2=pedestal_box
    ped_h=py2-py1; ped_w=px2-px1; statue_w=sx2-sx1
    overlap_x=min(sx2,px2)-max(sx1,px1)
    if overlap_x<.5*min(statue_w,ped_w) or (sy1+sy2)/2>=(py1+py2)/2: return False
    return py1-.25*ped_h<=sy2<=py1+.5*ped_h

def artifact_masks(frames,dets,folder):
    """Apply the mmgdino-large-pedestal.yaml recipe to the chosen artifact reference frame."""
    from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
    from ultralytics import SAM
    h,w=frames.shape[1:3]
    samples=np.unique(np.linspace(0,len(frames)-1,min(12,len(frames)),dtype=int))
    keyframes=sorted(samples,key=lambda i:foreground_mask(dets[i],(h,w)).sum())[:CFG['reference_frames']]
    ref=int(keyframes[0]); rgb=cv2.cvtColor(frames[ref],cv2.COLOR_BGR2RGB); image=Image.fromarray(rgb)
    model_id=SEGMENTATION_CFG['model_id']
    phrases=(SEGMENTATION_CFG['sculpture_phrases']+SEGMENTATION_CFG['painting_phrases']+
             SEGMENTATION_CFG['pedestal_phrases']+SEGMENTATION_CFG['negative_phrases'])
    processor=AutoProcessor.from_pretrained(model_id)
    model_dtype=torch.float16 if SEGMENTATION_CFG['fp16'] and DEVICE!='cpu' else torch.float32
    model=AutoModelForZeroShotObjectDetection.from_pretrained(model_id,dtype=model_dtype).to(DEVICE).eval()
    _fix_mmgdino_decoder_heads(model,model_id)
    detections=[]
    for item in _museum_dino_detections(model,processor,image,phrases,SEGMENTATION_CFG['full_thr']):
        item['source']='full'; detections.append(item)
    # Full-frame pass catches large/front-of-camera works; tiles recover smaller wall pieces.
    ncols=SEGMENTATION_CFG['tile_cols']; tile_h=int(h*SEGMENTATION_CFG['tile_region']); base_w=w/ncols
    tile_w=min(w,int(base_w*(1+SEGMENTATION_CFG['tile_overlap']))); rejected=[]
    tiles=[]
    for col in range(ncols) if not SEGMENTATION_CFG['no_tiles'] else []:
        center=(col+.5)*base_w; x1=int(np.clip(center-tile_w/2,0,w-tile_w)); tiles.append((x1,0,x1+tile_w,tile_h))
    for ti,(x1,y1,x2,y2) in enumerate(tiles):
        tile=Image.fromarray(rgb[y1:y2,x1:x2])
        for item in _museum_dino_detections(model,processor,tile,phrases,SEGMENTATION_CFG['tile_thr']):
            local=item['box']; item['box']=[local[0]+x1,local[1]+y1,local[2]+x1,local[3]+y1]; item['source']=f'tile_{ti}'
            if item['label']!='_negative':
                area=(local[2]-local[0])*(local[3]-local[1])/(tile.width*tile.height)
                margin=SEGMENTATION_CFG['tile_edge_margin']
                cut_edge=((x1>0 and local[0]<=margin) or (x2<w and local[2]>=tile.width-margin)
                          or (y2<h and local[3]>=tile.height-margin))
                if area>SEGMENTATION_CFG['tile_max_area'] or cut_edge:
                    item['reject']='tile_size_or_cut_edge'; rejected.append(item); continue
            detections.append(item)
    del model,processor; cleanup()
    kept=_museum_nms(detections); positives=[]; rescue_candidates=[]
    for item in kept:
        if item['label']=='_negative':
            statue_hypothesis=dict(item,label='sculpture')
            if SEGMENTATION_CFG['material_rescue'] and item['phrase']=='person' and _valid_segmentation_box(statue_hypothesis,w,h):
                rescue_candidates.append(item)
            continue
        if not _valid_segmentation_box(item,w,h):
            item['reject']='geometry'; rejected.append(item); continue
        positives.append(item)
    dino_review=frames[ref].copy()
    for item in positives:
        x1,y1,x2,y2=np.rint(item['box']).astype(int); cv2.rectangle(dino_review,(x1,y1),(x2,y2),(0,220,255),2)
        cv2.putText(dino_review,f"{item['phrase']} {item['score']:.2f}",(x1,max(16,y1)),0,.42,(0,220,255),1)
    for item in rescue_candidates:
        x1,y1,x2,y2=np.rint(item['box']).astype(int); cv2.rectangle(dino_review,(x1,y1),(x2,y2),(255,180,0),2)
        cv2.putText(dino_review,f"person? rescue {item['score']:.2f}",(x1,max(16,y1)),0,.4,(255,180,0),1)
    cv2.imwrite(str(folder/'artifact_detections_review.jpg'),dino_review)
    sam_candidates=positives+rescue_candidates
    if not sam_candidates:
        write_json(folder/'artifact_detections.json',dict(config=SEGMENTATION_CFG['name'],model=model_id,positives=[],rejected=rejected))
        raise RuntimeError('MM-Grounding-DINO không tìm được candidate hiện vật; kiểm tra artifact_detections_review.jpg.')
    reference_path=folder/'reference.jpg'; cv2.imwrite(str(reference_path),frames[ref])
    sam=SAM(SEGMENTATION_CFG['sam_model'])
    sam_result=sam(source=str(reference_path),bboxes=[x['box'] for x in sam_candidates],device=0 if DEVICE=='cuda' else DEVICE,verbose=False)
    if not sam_result or sam_result[0].masks is None:
        raise RuntimeError('Ultralytics SAM2.1 không trả về mask. Kiểm tra checkpoint sam2.1_b.pt/runtime GPU.')
    sam_masks=sam_result[0].masks.data.detach().cpu().numpy()
    if len(sam_masks)!=len(sam_candidates):
        raise RuntimeError(f'SAM trả {len(sam_masks)} masks cho {len(sam_candidates)} boxes; dừng để tránh gán mask nhầm hiện vật.')
    sam_scores=(sam_result[0].boxes.conf.detach().cpu().numpy() if sam_result[0].boxes is not None and sam_result[0].boxes.conf is not None else None)
    candidates=[]; rejected_masks=[]; rejected_rescue=[]
    for ci,(item,raw_mask) in enumerate(zip(sam_candidates,sam_masks)):
        mask=raw_mask>0
        if mask.shape!=(h,w): mask=cv2.resize(mask.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
        score=float(sam_scores[ci]) if sam_scores is not None and ci<len(sam_scores) else None
        is_rescue=ci>=len(positives)
        if is_rescue:
            stone=_stone_stats(frames[ref],mask)
            if not _looks_like_stone(stone):
                rejected_rescue.append(dict(phrase=item['phrase'],box=[float(v) for v in item['box']],score=float(item['score']),dark_ratio=round(stone['dark_ratio'],3),value_mean=round(stone['value_mean'],1),reason='material_rescue_rejected'))
                continue
            item=dict(item,label='sculpture',phrase='statue',rescued='material',**stone)
        area=int(mask.sum())
        box_area=max(1.,(item['box'][2]-item['box'][0])*(item['box'][3]-item['box'][1]))
        _,_,cc,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
        largest=float(cc[1:,cv2.CC_STAT_AREA].max()/area) if area else 0.; fill=area/box_area
        if (area<CFG['min_surface_pixels'] or fill<SEGMENTATION_CFG['min_mask_fill']
                or largest<SEGMENTATION_CFG['min_cc_ratio']
                or (score is not None and score<SEGMENTATION_CFG['min_sam_score'])):
            rejected_masks.append(dict(phrase=item['phrase'],box=[float(v) for v in item['box']],score=round(item['score'],4),sam_score=None if score is None else round(score,4),mask_area=area,fill=round(fill,3),largest_cc_ratio=round(largest,3),reason='mask_quality')); continue
        candidates.append(dict(mask=mask,box=list(item['box']),label=item['label'],phrase=item['phrase'],score=float(item['score']),sam_score=score,fill=float(fill),source=item['source'],rescued=item.get('rescued'),parts=[item['phrase']]))
    del sam,sam_result; cleanup()
    # Merge only near-identical masks; nearby pieces with overlapping boxes remain separate.
    candidates.sort(key=lambda x:int(x['mask'].sum()),reverse=True); merged=[]
    for obj in candidates:
        target=next((old for old in merged if SEGMENTATION_CFG['mask_overlap']>0 and _masks_are_duplicate(old['mask'],obj['mask'],SEGMENTATION_CFG['mask_overlap'])),None)
        if target is None: merged.append(obj); continue
        target['mask']=np.logical_or(target['mask'],obj['mask']); target['score']=max(target['score'],obj['score'])
        scores=[v for v in (target['sam_score'],obj['sam_score']) if v is not None]; target['sam_score']=max(scores) if scores else None
        target['parts']+=obj['parts']
        target['box']=[min(target['box'][0],obj['box'][0]),min(target['box'][1],obj['box'][1]),max(target['box'][2],obj['box'][2]),max(target['box'][3],obj['box'][3])]
        if 'sculpture' in (target['label'],obj['label']) and 'pedestal' in (target['label'],obj['label']): target['label']='sculpture'
    # A pedestal directly under a detected sculpture is one contact target/surface.
    pedestal_merges=[]; empty_pedestals=[]
    if SEGMENTATION_CFG['pedestal_merge']:
        pedestals=[o for o in merged if o['label']=='pedestal']; merged=[o for o in merged if o['label']!='pedestal']
        for pedestal in pedestals:
            pb=_mask_bbox(pedestal['mask']); possible=[o for o in merged if o['label']=='sculpture' and not o.get('has_pedestal') and _statue_on_pedestal(_mask_bbox(o['mask']),pb)]
            if not possible:
                if SEGMENTATION_CFG['keep_empty_pedestals']: merged.append(pedestal)
                else: empty_pedestals.append(pedestal)
                continue
            statue=min(possible,key=lambda o:abs(_mask_bbox(o['mask'])[3]-pb[1]))
            pedestal_merges.append(dict(phrase=pedestal['phrase'],into=statue['phrase']))
            statue['mask']=np.logical_or(statue['mask'],pedestal['mask']); statue['box']=_mask_bbox(statue['mask'])
            statue['score']=max(statue['score'],pedestal['score'])
            pair_scores=[v for v in (statue['sam_score'],pedestal['sam_score']) if v is not None]; statue['sam_score']=max(pair_scores) if pair_scores else None
            statue['parts']+=pedestal['parts']; statue['has_pedestal']=True
    artifacts=[]
    for i,obj in enumerate(merged,1):
        artifact=dict(id=f'artifact_{i}',mask=obj['mask'],box=_mask_bbox(obj['mask']),label=obj['label'],phrase=obj['phrase'],score=obj['score'],sam_score=obj['sam_score'],mask_fill=obj['fill'],source=obj['source'],parts=obj['parts'],rescued=bool(obj.get('rescued')),polygons=_mask_polygons(obj['mask']) if SEGMENTATION_CFG['mask_polygon'] else [])
        artifacts.append(artifact)
    if not artifacts: raise RuntimeError('SAM2 không có mask qua quality filters; xem artifact_detections_review.jpg.')
    canvas=frames[ref].copy(); colors=[(50,210,80),(220,120,30),(220,80,220),(20,190,240),(180,220,20)]
    for i,a in enumerate(artifacts):
        color=np.array(colors[i%len(colors)],np.uint8); m=a['mask']; canvas[m]=(canvas[m].astype(np.float32)*.58+color*.42).astype(np.uint8)
        x,y=map(int,a['box'][:2]); rescued=' [RESCUED]' if a['rescued'] else ''
        cv2.putText(canvas,f"{a['id']} {a['label']}{rescued}",(x,max(16,y)),0,.45,(0,255,255),1)
    rejected_canvas=dino_review.copy()
    for item in rejected:
        x1,y1,x2,y2=np.rint(item['box']).astype(int); cv2.rectangle(rejected_canvas,(x1,y1),(x2,y2),(0,0,255),1)
    for item in rejected_masks:
        x1,y1,x2,y2=np.rint(item['box']).astype(int); cv2.rectangle(rejected_canvas,(x1,y1),(x2,y2),(0,0,255),2)
        q='n/a' if item['sam_score'] is None else f"{item['sam_score']:.2f}"
        cv2.putText(rejected_canvas,f"SAM reject {item['phrase']} q={q}",(x1,max(16,y1)),0,.38,(0,0,255),1)
    for item in rejected_rescue:
        x1,y1,x2,y2=np.rint(item['box']).astype(int); cv2.rectangle(rejected_canvas,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.putText(rejected_canvas,f"rescue rejected V={item['value_mean']:.0f}",(x1,max(16,y1)),0,.38,(0,0,255),1)
    cv2.imwrite(str(folder/'artifact_review.jpg'),canvas); cv2.imwrite(str(folder/'artifact_rejected_review.jpg'),rejected_canvas)
    mask_dir=folder/'masks'; mask_dir.mkdir(exist_ok=True)
    for a in artifacts: cv2.imwrite(str(mask_dir/f"{a['id']}.png"),a['mask'].astype(np.uint8)*255)
    np.savez_compressed(folder/'artifact_masks.npz',**{a['id']:a['mask'] for a in artifacts})
    metadata=[{k:v for k,v in a.items() if k!='mask'} for a in artifacts]
    write_json(folder/'artifacts.json',metadata)
    write_json(folder/'artifact_detections.json',dict(config=SEGMENTATION_CFG['name'],parameters=SEGMENTATION_CFG,model=model_id,phrases=phrases,detections=len(detections),positive_candidates=len(positives),material_rescue_candidates=len(rescue_candidates),accepted_masks=len(artifacts),rejected_detections=rejected,rejected_masks=rejected_masks,rejected_material_rescue=rejected_rescue,pedestal_merges=pedestal_merges,dropped_empty_pedestals=len(empty_pedestals),objects=metadata))
    print(f"Config: {SEGMENTATION_CFG['name']} | MM-GDINO positives: {len(positives)} | material-rescue candidates: {len(rescue_candidates)} | accepted masks: {len(artifacts)}")
    print(f"Rejected detections/masks/material rescue: {len(rejected)}/{len(rejected_masks)}/{len(rejected_rescue)} | pedestal merges: {len(pedestal_merges)} | empty pedestals dropped: {len(empty_pedestals)}")
    display(Image.fromarray(cv2.cvtColor(dino_review,cv2.COLOR_BGR2RGB)))
    display(Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)))
    display(Image.fromarray(cv2.cvtColor(rejected_canvas,cv2.COLOR_BGR2RGB)))
    return ref,[int(i) for i in keyframes],artifacts

# %% [markdown]
# ## 3. VGGT reference và Depth Anything V2 Metric Indoor
# VGGT dùng depth head + camera head, không nạp kết quả point head không cần thiết.
# Resize VGGT pad được đảo chính xác cho depth và intrinsics. DA-V2 suy luận trực tiếp mỗi frame bằng OpenCV.

# %%
def vggt_reference(frames,dets,keyframes,folder):
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    h,w=frames.shape[1:3]; paths=[]
    for i in keyframes:
        rgb=cv2.cvtColor(frames[i],cv2.COLOR_BGR2RGB)
        rgb[foreground_mask(dets[i],(h,w))]=0
        p=folder/f'vggt_input_{i:05d}.png'; Image.fromarray(rgb).save(p); paths.append(str(p))
    model=VGGT.from_pretrained('facebook/VGGT-1B').to(DEVICE).eval()
    model.point_head=None; model.track_head=None  # release unused head weights on GPU
    images=load_and_preprocess_images(paths,mode='pad').to(DEVICE)
    with torch.inference_mode(),torch.autocast('cuda',dtype=DTYPE):
        tokens,patch_start=model.aggregator(images[None])
    # Official forward keeps geometry heads in float32 for numerical stability.
    with torch.inference_mode(),torch.autocast('cuda',enabled=False):
        pose=model.camera_head(tokens)[-1]
        depth,confidence=model.depth_head(tokens,images[None],patch_start)
    extr,K=pose_encoding_to_extri_intri(pose,images.shape[-2:])
    d=depth[0,0,...,0].float().cpu().numpy()
    c=confidence[0,0].float().cpu().numpy()
    K=K[0,0].float().cpu().numpy()
    if w>=h:
        nw=518; nh=round(h*518/w/14)*14
    else:
        nh=518; nw=round(w*518/h/14)*14
    left=(518-nw)//2; top=(518-nh)//2
    A=np.array([[nw/w,0,left],[0,nh/h,top],[0,0,1]],np.float32)
    K=np.linalg.inv(A)@K
    d=cv2.resize(d[top:top+nh,left:left+nw],(w,h))
    c=cv2.resize(c[top:top+nh,left:left+nw],(w,h))
    np.savez_compressed(folder/'vggt_reference.npz',depth=d,confidence=c,K=K,
        frame=keyframes[0],extrinsics=extr.float().cpu().numpy())
    del model,images,tokens,pose,depth,confidence,extr
    cleanup()
    return d,c,K

def depth_anything_v2_depth(frames,folder):
    from huggingface_hub import hf_hub_download
    from depth_anything_v2.dpt import DepthAnythingV2
    checkpoint=hf_hub_download('depth-anything/Depth-Anything-V2-Metric-Hypersim-Small',
        'depth_anything_v2_metric_hypersim_vits.pth')
    model=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384],max_depth=20)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    model=model.to(DEVICE).eval()
    depths=[]
    for fi,bgr in enumerate(frames):
        with torch.inference_mode(),torch.autocast('cuda',dtype=DTYPE):
            depth=model.infer_image(bgr,input_size=CFG['da_v2_input_size'])
        depth=np.asarray(depth,dtype=np.float32)
        if depth.shape!=bgr.shape[:2] or not np.isfinite(depth).any():
            raise RuntimeError(f'Depth Anything V2 lỗi tại frame {fi}: shape/nonfinite.')
        depths.append(depth)
        if fi%30==0:
            print(f'Depth Anything V2: {fi+1}/{len(frames)}')
    depths=np.stack(depths)
    np.save(folder/'depth_anything_v2_metric_indoor.npy',depths.astype(np.float16))
    del model
    cleanup()
    return depths

# %% [markdown]
# ## 4. Căn chỉnh có kiểm tra chất lượng, cố định bề mặt hiện vật
# VGGT không có thang mét mặc định. Fit **scale dương, không shift** VGGT → DA-V2 Metric trên nền tĩnh.
# DA-V2 dự đoán độc lập từng frame; fit mỗi frame về reference và kiểm tra trên pixel holdout.
# Loại bbox người cả reference/current, biên depth, vùng confidence thấp; kiểm tra đủ phân bố không gian.
# Cảnh camera di chuyển, depth không khớp hoặc bàn tay mơ hồ → UNKNOWN.
# Độ bất định dưới đây là heuristic bảo thủ, không phải khoảng tin cậy đã hiệu chuẩn.

# %%
def stable_depth_mask(d):
    valid=np.isfinite(d)&(d>.05)&(d<100)
    filled=np.where(valid,d,0).astype(np.float32)
    gx=cv2.Sobel(filled,cv2.CV_32F,1,0,ksize=3)/8
    gy=cv2.Sobel(filled,cv2.CV_32F,0,1,ksize=3)/8
    return valid & (np.hypot(gx,gy)<.05*np.maximum(filled,.1))

def fit_scale(source,target,mask,min_pixels=800):
    valid=mask & np.isfinite(source)&np.isfinite(target)&(source>1e-6)&(target>1e-6)
    yy,xx=np.where(valid)
    bad=dict(ok=False,scale=None,median_rel=None,p90_rel=None,p90_abs=None,n=int(len(xx)))
    if len(xx)<min_pixels:
        return bad
    h,w=mask.shape
    tiles=np.unique((yy*4//h)*4+(xx*4//w))
    if len(tiles)<4:
        return bad
    fit=(xx+yy)%2==0
    if min(fit.sum(),(~fit).sum())<min_pixels//4:
        return bad
    x=source[yy,xx]; y=target[yy,xx]
    scale=float(np.median(y[fit]/x[fit]))
    absolute=np.abs(scale*x[~fit]-y[~fit])
    relative=absolute/np.maximum(y[~fit],1e-6)
    med,p90=np.percentile(relative,[50,90])
    return dict(ok=bool(np.isfinite(scale) and scale>0 and med<=CFG['max_alignment_median_rel'] and p90<=CFG['max_alignment_p90_rel']),
        scale=scale,median_rel=float(med),p90_rel=float(p90),p90_abs=float(np.percentile(absolute,90)),n=int(len(xx)))

def unproject(depth,K,mask):
    yy,xx=np.where(mask)
    rays=np.linalg.inv(K)@np.stack([xx,yy,np.ones(len(xx))])
    return (rays*depth[yy,xx]).T

def build_geometry(vg,confidence,K,depths,ref,dets,artifacts,folder):
    h,w=vg.shape; fg=foreground_mask(dets[ref],(h,w))
    finite=confidence[np.isfinite(confidence)]
    assert finite.size and np.isfinite(K).all() and K[0,0]>0 and K[1,1]>0
    conf_good=np.isfinite(confidence)&(confidence>=np.percentile(finite,40))
    observed=(~fg)&conf_good&np.isfinite(vg)&(vg>1e-6)
    artifact_union=np.logical_or.reduce([a['mask'] for a in artifacts])
    background=observed & ~artifact_union & stable_depth_mask(depths[ref])
    anchor=fit_scale(vg,depths[ref],background,CFG['min_alignment_pixels'])
    write_json(folder/'reference_alignment.json',anchor)
    surface_depth=vg*(anchor['scale'] or 1.)
    surfaces=[]
    for a in artifacts:
        support=a['mask']&observed&stable_depth_mask(depths[ref])
        # Cross-model disagreement on the actual artifact must also be acceptable.
        error=np.abs(surface_depth-depths[ref])
        local=error[support]
        uncertainty=float(np.percentile(local,90)) if local.size else None
        support &= error<CFG['max_wrist_gap_m']
        points=unproject(surface_depth,K,support)
        tree=cKDTree(points) if len(points)>=CFG['min_surface_pixels'] else None
        surfaces.append(dict(artifact=a,tree=tree,support=support,uncertainty=uncertainty,
            image_dt=cv2.distanceTransform((~a['mask']).astype(np.uint8),cv2.DIST_L2,3),
            support_dt=cv2.distanceTransform((~support).astype(np.uint8),cv2.DIST_L2,3)))
        np.save(folder/f"{a['id']}_surface.npy",points.astype(np.float32))
    np.savez_compressed(folder/'aligned_reference.npz',depth=surface_depth,K=K,background=background)
    write_json(folder/'surface_quality.json',[dict(id=s['artifact']['id'],pixels=int(s['support'].sum()),
        p90_cross_model_error_m=s['uncertainty']) for s in surfaces])
    return surface_depth,background,anchor,surfaces

def wrist_point(depth,K,xy):
    h,w=depth.shape; x,y=map(lambda v:int(round(v)),xy)
    if not (2<=x<w-2 and 2<=y<h-2):
        return None,None
    patch=depth[y-2:y+3,x-2:x+3]; valid=patch[np.isfinite(patch)&(patch>.05)&(patch<100)]
    if len(valid)<20:
        return None,None
    z=float(np.median(valid)); spread=float(np.percentile(valid,90)-np.percentile(valid,10))
    if spread>max(.03,.03*z):
        return None,None
    return (np.linalg.inv(K)@np.array([xy[0],xy[1],1.]))*z,spread

def camera_static_check(reference,frame,mask):
    # Feature correspondences only on unoccluded background; fail closed if not observable.
    orb=cv2.ORB_create(nfeatures=1000)
    m=mask.astype(np.uint8)*255
    ka,da=orb.detectAndCompute(cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY),m)
    kb,db=orb.detectAndCompute(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),m)
    if da is None or db is None:
        return False,None
    pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2)
    good=[p[0] for p in pairs if len(p)==2 and p[0].distance<.7*p[1].distance]
    if len(good)<20:
        return False,None
    delta=np.array([np.array(kb[p.trainIdx].pt)-ka[p.queryIdx].pt for p in good])
    shift=float(np.median(np.linalg.norm(delta,axis=1)))
    return shift<2.0,shift

# %% [markdown]
# ## 5. Khoảng cách, chuyển động và xuất video bằng chứng
# `NO_TOUCH` nghĩa là không đạt điều kiện hình học tại frame quan sát được, không phải nhãn ground truth.
# Mất pose/depth/alignment hoặc chuyển động quá mơ hồ → UNKNOWN và xóa lịch sử hit; không giữ cảnh báo đỏ cũ.
# Chỉ cổ tay cùng bên, cùng ID và cùng hiện vật được cộng dồn. Cần giữ gần bề mặt và tốc độ đủ thấp.

# %%
def temporal_gate(history,key,frame,valid,hit,speed,fps):
    if not valid or speed is None:
        history.pop(key,None)
        return 'UNKNOWN'
    if not hit:
        history.pop(key,None)
        return 'NO_TOUCH'
    queue=history.setdefault(key,deque())
    if queue and queue[-1][0]!=frame-1:
        queue.clear()
    queue.append((frame,speed<=CFG['max_hand_speed_m_s']))
    while queue and frame-queue[0][0]>=max(1,round(fps*CFG['history_seconds'])):
        queue.popleft()
    required=max(3,math.ceil(fps*CFG['hold_seconds']))
    if len(queue)>=required and sum(v for _,v in queue)>=required:
        return 'TOUCH_CANDIDATE'
    return 'UNKNOWN'  # near but insufficient temporal evidence

def render_and_score(frames,fps,tracks,dets,depths,K,ref,surface_depth,background,anchor,surfaces,folder,
                     motion_scores=None,camera_id=None):
    h,w=frames.shape[1:3]; rows=[]; align_rows=[]; history={}; previous={}; active={}; events=[]
    motion_scores=motion_scores or []
    def motion_at(tid,time_s):
        return [m for m in motion_scores if int(m['track_id'])==int(tid)
                and m['start_s']<=time_s<=m['end_s']]
    raw_path=folder/'preview_raw.mp4'
    writer=cv2.VideoWriter(str(raw_path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    assert writer.isOpened(), 'Không mở được VideoWriter.'
    def close_event(key):
        e=active.pop(key); e['start_s']=e['start_frame']/fps; e['end_s']=(e['end_frame']+1)/fps
        events.append(e)
    try:
        for fi,frame in enumerate(frames):
            bg=background & ~foreground_mask(dets[fi],(h,w)) & stable_depth_mask(depths[fi])
            static,shift=camera_static_check(frames[ref],frame,bg)
            alignment=fit_scale(depths[fi],surface_depth,bg,CFG['min_alignment_pixels'])
            scale=alignment['scale']
            ok=bool(anchor['ok'] and alignment['ok'] and static and scale is not None and abs(scale-1)<=CFG['max_scale_drift'])
            align_rows.append(dict(frame=fi,time_s=fi/fps,usable=ok,camera_shift_px=shift,**alignment))
            depth=depths[fi]*(scale or 1.)
            canvas=frame.copy()
            for s in surfaces:
                a=s['artifact']; m=a['mask']
                canvas[m]=(canvas[m]*.75+np.array([60,220,80])*.25).astype(np.uint8)
                x,y=map(int,a['box'][:2]); cv2.putText(canvas,a['id'],(x,max(15,y)),0,.4,(0,230,100),1)
            frame_candidates={}; seen=set(); new_evidence=[]
            for track in tracks[fi]:
                tid=track['id']; person_states=[]
                for wi in (9,10):
                    wrist='left' if wi==9 else 'right'
                    valid_pose=track['kp'] is not None and track['kc'][wi]>=CFG['wrist_conf']
                    xy=track['kp'][wi] if valid_pose else None
                    point,spread=wrist_point(depth,K,xy) if valid_pose and ok else (None,None)
                    motion_key=(tid,wi); speed=None
                    if point is not None:
                        old=previous.get(motion_key)
                        if old is not None and old[0]==fi-1:
                            speed=float(np.linalg.norm(point-old[1])*fps)
                        previous[motion_key]=(fi,point)
                        cv2.circle(canvas,tuple(np.rint(xy).astype(int)),3,(0,255,255),-1)
                    else:
                        previous.pop(motion_key,None)
                    for s in surfaces:
                        aid=s['artifact']['id']; key=(tid,aid,wi); seen.add(key)
                        gap3=gap2=uncertainty=None; reason='missing_pose_or_depth'
                        valid=point is not None and s['tree'] is not None and s['uncertainty'] is not None
                        if not ok:
                            reason='alignment_or_camera_failed'
                        if valid:
                            x,y=np.rint(xy).astype(int)
                            gap2=float(s['image_dt'][y,x]); gap3=float(s['tree'].query(point)[0])
                            # Near unobserved surface: do not use a distant visible part as proof of no contact.
                            valid=bool(s['support_dt'][y,x]<=CFG['max_surface_gap_px'] or gap2>CFG['max_image_gap_px'])
                            uncertainty=max(s['uncertainty'],alignment['p90_abs'] or 0.,spread or 0.)
                            valid=valid and uncertainty<CFG['max_wrist_gap_m']
                            reason='geometry_observed' if valid else 'surface_or_depth_uncertain'
                        hit=bool(valid and gap2<=CFG['max_image_gap_px'] and gap3+uncertainty<=CFG['max_wrist_gap_m'])
                        state=temporal_gate(history,key,fi,valid,hit,speed,fps)
                        if valid and gap2<=CFG['max_image_gap_px'] and abs(gap3-CFG['max_wrist_gap_m'])<=uncertainty:
                            state='UNKNOWN'; history.pop(key,None); reason='distance_within_uncertainty'
                        if state=='UNKNOWN' and valid and hit:
                            reason='insufficient_hold_or_motion_evidence'
                        row=dict(frame=fi,time_s=fi/fps,person_id=tid,artifact_id=aid,wrist=wrist,
                            status=state,reason=reason,image_gap_px=gap2,estimated_wrist_gap_m=gap3,
                            uncertainty_heuristic_m=uncertainty,wrist_speed_m_s=speed,
                            wrist_confidence=track['kc'][wi] if track['kc'] is not None else None)
                        rows.append(row); person_states.append(state)
                        if state=='TOUCH_CANDIDATE':
                            pair=(tid,aid)
                            if pair not in frame_candidates or gap3<frame_candidates[pair]['estimated_wrist_gap_m']:
                                frame_candidates[pair]=row
                status='TOUCH_CANDIDATE' if 'TOUCH_CANDIDATE' in person_states else ('UNKNOWN' if 'UNKNOWN' in person_states else 'NO_TOUCH')
                color=(0,0,255) if status=='TOUCH_CANDIDATE' else ((0,200,255) if status=='UNKNOWN' else (255,200,0))
                x1,y1,x2,y2=np.rint(track['box']).astype(int)
                cv2.rectangle(canvas,(x1,y1),(x2,y2),color,2)
                cv2.putText(canvas,f'ID {tid} {status}',(max(0,x1),max(14,y1-4)),0,.38,color,1)
                nearby_motion=motion_at(tid,fi/fps)
                if nearby_motion:
                    peak=max(nearby_motion,key=lambda m:m['camera_percentile'])
                    unstable=int(camera_id) in IF_UNSTABLE_CAMERAS
                    tag=f"IF p{peak['camera_percentile']:.3f}"+(' UNVALIDATED' if unstable else '')
                    cv2.putText(canvas,tag,(max(0,x1),min(h-5,max(28,y1+12))),0,.34,(200,80,200) if unstable else (0,180,255),1)
            for key in list(history):
                if key not in seen:
                    history.pop(key,None)
            for key in list(active):
                if key not in frame_candidates:
                    close_event(key)
            for pair,row in frame_candidates.items():
                if pair not in active:
                    event_id=f"{folder.name}_touch_{len(events)+len(active)+1:04d}"
                    nearby_motion=motion_at(pair[0],fi/fps)
                    peak=max(nearby_motion,key=lambda m:m['camera_percentile']) if nearby_motion else None
                    active[pair]=dict(event_id=event_id,person_id=pair[0],artifact_id=pair[1],
                        start_frame=fi,end_frame=fi,min_estimated_wrist_gap_m=row['estimated_wrist_gap_m'],
                        status='TOUCH_CANDIDATE',evidence=f'{event_id}.jpg',
                        motion_anomaly_percentile=peak['camera_percentile'] if peak else None,
                        motion_context=('camera_unvalidated_score_only' if peak and int(camera_id) in IF_UNSTABLE_CAMERAS
                                        else ('unusual_motion_outlier' if peak and peak['camera_percentile']>=.99 else 'no_high_motion_score')))
                    new_evidence.append(folder/f'{event_id}.jpg')
                else:
                    active[pair]['end_frame']=fi
                    active[pair]['min_estimated_wrist_gap_m']=min(active[pair]['min_estimated_wrist_gap_m'],row['estimated_wrist_gap_m'])
                    nearby_motion=motion_at(pair[0],fi/fps)
                    if nearby_motion:
                        peak=max(nearby_motion,key=lambda m:m['camera_percentile'])
                        old=active[pair]['motion_anomaly_percentile']
                        if old is None or peak['camera_percentile']>old:
                            active[pair]['motion_anomaly_percentile']=peak['camera_percentile']
                        if int(camera_id) in IF_UNSTABLE_CAMERAS:
                            active[pair]['motion_context']='camera_unvalidated_score_only'
                        elif peak['camera_percentile']>=.99:
                            active[pair]['motion_context']='unusual_motion_outlier'
            banner=f"frame {fi} | alignment {'OK' if ok else 'UNKNOWN'} | wrist proxy; metric ESTIMATE"
            cv2.rectangle(canvas,(0,h-24),(w,h),(25,25,25),-1)
            cv2.putText(canvas,banner,(5,h-8),0,.4,(255,255,255),1)
            if frame_candidates:
                pair,row=next(iter(frame_candidates.items()))
                cv2.putText(canvas,f"SUSPECT ID {pair[0]} -> {pair[1]} ~{row['estimated_wrist_gap_m']:.2f}m",
                    (6,20),0,.5,(0,0,255),2)
            for evidence_path in new_evidence:
                cv2.imwrite(str(evidence_path),canvas)
            writer.write(canvas)
        for key in list(active):
            close_event(key)
    finally:
        writer.release()
    fields=['frame','time_s','person_id','artifact_id','wrist','status','reason','image_gap_px',
        'estimated_wrist_gap_m','uncertainty_heuristic_m','wrist_speed_m_s','wrist_confidence']
    write_csv(folder/'observations.csv',rows,fields)
    write_csv(folder/'alignment.csv',align_rows,['frame','time_s','usable','camera_shift_px','ok','scale','median_rel','p90_rel','p90_abs','n'])
    write_json(folder/'events.json',events)
    write_csv(folder/'events.csv',events,['event_id','person_id','artifact_id','start_frame','end_frame','start_s','end_s',
        'min_estimated_wrist_gap_m','status','motion_anomaly_percentile','motion_context','evidence'])
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(raw_path),
        '-c:v','libx264','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'preview.mp4')],check=True)
    raw_path.unlink()  # chỉ xóa bản render tạm vừa tạo; giữ preview H.264.
    report=dict(frames=len(frames),fps=fps,events=len(events),
        usable_alignment_frames=sum(r['usable'] for r in align_rows),
        unknown_observations=sum(r['status']=='UNKNOWN' for r in rows),
        events_with_motion_context=sum(e.get('motion_anomaly_percentile') is not None for e in events),
        motion_model_camera_status='unstable_holdout_review_only' if camera_id in IF_UNSTABLE_CAMERAS else 'stable_holdout_pilot',
        observation_count=len(rows),metric_calibrated=False,hand_landmarks=False)
    write_json(folder/'summary.json',report)
    return report

# %% [markdown]
# ## Chạy toàn pipeline theo phase — mỗi lần một video
# Mỗi phase tự hiện preview/báo cáo và tự chuyển sang phase tiếp theo; không cần nhập xác nhận.
# Có thể chạy lần lượt từng cell để quan sát, hoặc Run all để chạy hết video hiện tại.
# Chạy lại phase trước sẽ xóa trạng thái hoàn tất của các phase sau, nhưng giữ file cũ trong attempt riêng.
# Không tự fallback khi thiếu trọng số/hết VRAM. Đổi CFG thì khởi tạo lượt mới ở phase 0.

# %%
from datetime import datetime
PHASE_NAMES=['input','tracking','pose','motion_anomaly','artifacts','vggt','depth_anything_v2','alignment','touch']

def begin_phase(index):
    if STATE['config']!=CFG:
        raise RuntimeError('CFG đã đổi. Chạy lại phase 0 để tạo lượt mới, tránh trộn cấu hình.')
    required=set(PHASE_NAMES[:index])
    if not required.issubset(set(STATE['completed'])):
        raise RuntimeError(f"Phase trước chưa chạy xong: {sorted(required-set(STATE['completed']))}")
    STATE['completed']=[p for p in STATE['completed'] if p in PHASE_NAMES[:index]]
    STATE['ready']=[p for p in STATE['ready'] if p in PHASE_NAMES[:index]]
    STATE['attempts']={k:v for k,v in STATE['attempts'].items() if k in PHASE_NAMES[:index]}
    phase=PHASE_NAMES[index]
    folder=RUN/f'{index:02d}_{phase}'/datetime.now().strftime('attempt_%H%M%S_%f')
    folder.mkdir(parents=True)
    STATE['attempts'][phase]=str(folder.relative_to(RUN))
    write_json(RUN/'phase_review.json',STATE)
    return folder

def review_phase(index):
    # Chỉ gọi sau khi inference, lưu kết quả và hiển thị preview đều thành công.
    phase=PHASE_NAMES[index]
    if phase not in STATE['attempts']:
        raise RuntimeError('Phase chưa bắt đầu.')
    if phase not in STATE['ready']:
        STATE['ready'].append(phase)
    if STATE['config']!=CFG or not set(PHASE_NAMES[:index]).issubset(set(STATE['completed'])):
        raise RuntimeError('Cấu hình hoặc kết quả phase trước đã đổi. Chạy lại đúng thứ tự.')
    if phase not in STATE['completed']:
        STATE['completed'].append(phase)
    write_json(RUN/'phase_review.json',STATE)
    print(f'Phase {index} ({phase}) hoàn tất; notebook sẽ tự chạy tiếp khi chạy Run all.')

def phase_diagnostics(folder, phase, metrics, checks, files):
    """Save a compact, actionable health report for the completed phase."""
    rows=[dict(status=status,check=name,observed=observed,next_action=action)
          for status,name,observed,action in checks]
    report=dict(phase=phase,metrics=metrics,checks=rows,files=files)
    write_json(Path(folder)/'phase_diagnostic.json',report)
    write_csv(Path(folder)/'phase_diagnostic.csv',rows,['status','check','observed','next_action'])
    print(f'\n=== KIỂM TRA PHASE: {phase} ===')
    display(pd.DataFrame(rows))
    print('Báo cáo chẩn đoán:',Path(folder)/'phase_diagnostic.json')
    print('File cần mở để xem lại:',', '.join(files))
    if any(r['status']=='FAIL' for r in rows):
        print('Có mục FAIL: nên dừng tại đây, xử lý nguyên nhân rồi chạy lại phase này.')
    elif any(r['status']=='WARN' for r in rows):
        print('Có mục WARN: xem kỹ preview/báo cáo; notebook vẫn tiếp tục chạy.')
    else:
        print('Các kiểm tra tự động đều PASS; vẫn cần xem preview bằng mắt.')

def preview_video(path,images,fps):
    import imageio_ffmpeg
    path=Path(path); raw=path.with_name(path.stem+'_raw.mp4')
    writer=None
    try:
        for image in images:
            if writer is None:
                writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(image.shape[1],image.shape[0]))
                assert writer.isOpened()
            writer.write(image)
    finally:
        if writer is not None:
            writer.release()
    if writer is None:
        raise RuntimeError('Không có frame preview.')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(raw),
        '-c:v','libx264','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],check=True)
    raw.unlink()
    display(Video(str(path),embed=True,width=800))

def person_preview(frames,records,with_pose=False):
    bones=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    for frame,people in zip(frames,records):
        image=frame.copy()
        for p in people:
            x1,y1,x2,y2=np.rint(p['box']).astype(int)
            cv2.rectangle(image,(x1,y1),(x2,y2),(255,210,0),2)
            label=f"ID {p['id']}"
            if with_pose and p['kp'] is None:
                label+=' | POSE MISSING'
            cv2.putText(image,label,(max(x1,0),max(y1-4,15)),0,.45,(255,210,0),1)
            if with_pose and p['kp'] is not None:
                xy=np.rint(p['kp']).astype(int); conf=np.asarray(p['kc'])
                for a,b in bones:
                    if min(conf[a],conf[b])>=CFG['wrist_conf']:
                        cv2.line(image,tuple(xy[a]),tuple(xy[b]),(0,220,0),1)
                for wi in (9,10):
                    if conf[wi]>=CFG['wrist_conf']:
                        cv2.circle(image,tuple(xy[wi]),4,(0,255,255),-1)
        yield image

def _if_row_features(rec,frame_w,frame_h):
    if rec.get('kp') is None or rec.get('kc') is None or rec.get('box') is None: return None
    coords=np.asarray(rec['kp'],dtype=np.float32); conf=np.asarray(rec['kc'],dtype=np.float32)
    box=np.asarray(rec['box'],dtype=np.float32)
    if coords.shape!=(17,2) or conf.shape!=(17,) or not np.isfinite(box).all(): return None
    x1,y1,x2,y2=box; bw=max(float(x2-x1),1.); bh=max(float(y2-y1),1.)
    visible=conf>=MIN_KP_CONF
    hips=[11,12]; shoulders=[5,6]
    origin=coords[hips].mean(axis=0) if visible[hips].all() else np.array([(x1+x2)/2,(y1+y2)/2],np.float32)
    scale=max(bh,math.sqrt(bw*bh),1.)
    normalized=(coords-origin)/scale; normalized[~visible]=np.nan
    hip_height=np.nan; torso_angle=np.nan
    if visible[hips].all():
        hc=coords[hips].mean(axis=0); hip_height=(hc[1]-y1)/bh
        if visible[shoulders].all():
            sc=coords[shoulders].mean(axis=0); vec=sc-hc
            torso_angle=math.atan2(float(vec[0]),float(-vec[1]))
    bbox=np.asarray([((x1+x2)/2)/frame_w,((y1+y2)/2)/frame_h,
        bw/frame_w,bh/frame_h,bw/bh,hip_height,torso_angle],np.float32)
    return normalized.reshape(-1),conf,bbox

def _make_if_windows(by_track):
    names=IF_FEATURE_NAMES; output=[]
    for track_key,rows in by_track.items():
        rows.sort(key=lambda x:x[0]); times=np.asarray([r[0] for r in rows])
        if len(rows)<2: continue
        first=math.ceil(times[0]); last=times[-1]-WINDOW_SECONDS
        start=float(first)
        while start<=last+1e-8:
            grid=np.linspace(start,start+WINDOW_SECONDS,N_STEPS)
            pose=np.full((N_STEPS,34),np.nan,np.float32); confidence=np.full((N_STEPS,17),np.nan,np.float32)
            bbox=np.full((N_STEPS,7),np.nan,np.float32); valid=np.zeros(N_STEPS,bool)
            for si,target in enumerate(grid):
                right=int(np.searchsorted(times,target,side='left'))
                if right==0 or right==len(times): continue
                left=right-1; gap=times[right]-times[left]
                if gap<=0 or gap>MAX_POSE_GAP_S: continue
                a=(target-times[left])/gap
                for dest,ai,bi in ((pose,rows[left][1],rows[right][1]),
                                   (confidence,rows[left][2],rows[right][2]),
                                   (bbox,rows[left][3],rows[right][3])):
                    ok=np.isfinite(ai)&np.isfinite(bi); dest[si,ok]=ai[ok]+(bi[ok]-ai[ok])*a
                valid[si]=np.isfinite(bbox[si,:5]).all()
            ratio=float(valid.mean()); visibility=float(np.isfinite(pose.reshape(N_STEPS,17,2)).all(axis=2)[valid].mean()) if valid.any() else 0.
            if ratio>=MIN_VALID_RATIO and visibility>=MIN_VISIBILITY:
                dt=WINDOW_SECONDS/(N_STEPS-1)
                pv=np.full_like(pose,np.nan); pv[1:]=(pose[1:]-pose[:-1])/dt
                bv=np.full((N_STEPS,2),np.nan,np.float32); bv[1:]=(bbox[1:,:2]-bbox[:-1,:2])/dt
                x=np.concatenate([pose,confidence,pv,bbox,bv],axis=1).astype(np.float32)
                if x.shape[1]!=len(names): raise ValueError('Schema live pose khác schema train.')
                if isinstance(track_key,tuple):
                    camera_id,segment_id,tid=track_key
                else:
                    camera_id,segment_id,tid=None,'live',track_key
                output.append({'track_id':int(tid),'camera_id':camera_id,'segment_id':segment_id,
                    'start_s':start,'end_s':start+WINDOW_SECONDS,'features':x,
                    'valid_step_ratio':ratio,'keypoint_visibility':visibility})
            start+=1.
    return output

IF_FEATURE_NAMES=([f'pose_{name}_{axis}' for name in ['nose','left_eye','right_eye','left_ear','right_ear',
    'left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip',
    'right_hip','left_knee','right_knee','left_ankle','right_ankle'] for axis in ('x','y')]
    +[f'keypoint_confidence_{name}' for name in ['nose','left_eye','right_eye','left_ear','right_ear',
    'left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip',
    'right_hip','left_knee','right_knee','left_ankle','right_ankle']]
    +[f'pose_velocity_{name}_{axis}' for name in ['nose','left_eye','right_eye','left_ear','right_ear',
    'left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip',
    'right_hip','left_knee','right_knee','left_ankle','right_ankle'] for axis in ('x','y')]
    +['bbox_center_x','bbox_center_y','bbox_width','bbox_height','bbox_aspect','hip_height','torso_angle',
      'bbox_velocity_x','bbox_velocity_y'])

def build_live_if_windows(frames,tracks,fps):
    h,w=frames.shape[1:3]; byid=defaultdict(list)
    for fi,people in enumerate(tracks):
        for rec in people:
            vals=_if_row_features(rec,w,h)
            if vals is not None: byid[int(rec['id'])].append((fi/fps,*vals))
    return _make_if_windows(byid)

def extract_normal_yolo_windows(normal_root):
    from ultralytics import YOLO
    normal_root=Path(normal_root)
    frame_times={}; mf_path=locate_input_file('manifest.csv')
    mf=pd.read_csv(mf_path)
    for (cam,seg),g in mf.groupby(['camera_id','segment_id']):
        for sec,sg in g.groupby('timestamp_source'):
            sg=sg.sort_values('frame_id'); n=len(sg)
            for rank,row in enumerate(sg.itertuples(index=False)):
                frame_times[(int(cam),str(seg),int(row.frame_id))]=float(sec)+(rank+.5)/n
    grouped=defaultdict(list)
    with locate_input_file('annotations.jsonl').open() as f:
        for line in f:
            r=json.loads(line); key=(int(r['camera_id']),str(r['segment_id']),int(r['frame_id']))
            if key in frame_times: grouped[key].append(r)
    model=YOLO('yolov8n-pose.pt'); by_track=defaultdict(list); processed=missing=0
    for key,people in sorted(grouped.items()):
        cam,seg,fid=key; parts=PurePosixPath(people[0]['filename_raw']).parts
        image_path=normal_root/'frames'/str(cam)/Path(*parts[-2:])
        frame=cv2.imread(str(image_path))
        if frame is None: missing+=1; continue
        h,w=frame.shape[:2]; jobs=[]
        for ann in people:
            b=np.asarray([ann['bb_x'],ann['bb_y'],ann['bb_x']+ann['bb_w'],ann['bb_y']+ann['bb_h']],np.float32)
            x1,y1,x2,y2=b; dx=(x2-x1)*.20; dy=(y2-y1)*.10
            xa,ya=max(0,int(x1-dx)),max(0,int(y1-dy)); xb,yb=min(w,int(x2+dx)),min(h,int(y2+dy))
            if xb-xa<8 or yb-ya<8: continue
            jobs.append((ann,frame[ya:yb,xa:xb],np.array([xa,ya]),b-np.array([xa,ya,xa,ya]),b))
        results=model.predict([j[1] for j in jobs],device=0,imgsz=640,conf=.15,verbose=False) if jobs else []
        for (ann,crop,origin,target,b),result in zip(jobs,results):
            rec={'id':int(ann['visitor_id']),'box':b.tolist(),'kp':None,'kc':None}
            if result.keypoints is not None and result.keypoints.conf is not None and len(result.boxes):
                pb=result.boxes.xyxy.cpu().numpy(); ious=np.asarray([box_iou(q,target) for q in pb])
                order=np.argsort(ious)[::-1]
                if ious[order[0]]>=.30 and (len(order)==1 or ious[order[0]]-ious[order[1]]>=.15):
                    j=order[0]; rec['kp']=(result.keypoints.xy[j].cpu().numpy()+origin).tolist()
                    rec['kc']=result.keypoints.conf[j].cpu().numpy().tolist()
            vals=_if_row_features(rec,w,h)
            if vals is not None: by_track[(cam,seg,int(ann['visitor_id']))].append((frame_times[key],*vals))
        processed+=1
        if processed%100==0: print(f'YOLOv8-Pose normal frames: {processed}/{len(grouped)}')
    del model; cleanup()
    if missing: print(f'Không đọc được {missing} ảnh normal từ Kaggle Input.')
    windows=_make_if_windows(by_track)
    if not windows: raise RuntimeError('Không tạo được normal windows dài 2.5s; kiểm tra dataset/pose quality.')
    X=np.stack([w.pop('features') for w in windows]); meta=windows
    names=IF_FEATURE_NAMES
    np.savez_compressed(IF_DIR/'normal_yolov8_pose_windows.npz',features=X,
        metadata_json=np.asarray([json.dumps(m) for m in meta]),feature_names=np.asarray(names))
    print(f'Normal windows cùng YOLOv8-Pose: {X.shape}; tracks {len(by_track)}; frames {processed}/{len(grouped)}')
    return X,meta,names

def score_live_if_windows(windows,camera_id,folder):
    rows=[]
    if windows:
        X=np.stack([x['features'] for x in windows])
        model_names=IF_BUNDLE.get('source_feature_names',IF_BUNDLE.get('feature_names'))
        if model_names!=IF_FEATURE_NAMES: raise ValueError('Feature schema trong checkpoint không khớp với pose inference.')
        emb=summarize_if_windows(X,model_names)
        raw=-IF_BUNDLE['pipeline'].decision_function(emb)
        reference=IF_BUNDLE['camera_reference_scores'].get(int(camera_id))
        if reference is None: raise ValueError(f'Không có normal reference cho camera {camera_id}.')
        pct=if_percentile(reference,raw)
        for win,s,p in zip(windows,raw,pct):
            unstable=int(camera_id) in IF_UNSTABLE_CAMERAS
            kind=('unvalidated_camera_score_only' if unstable else
                  ('unusual_motion_candidate' if p>=.99 else 'below_review_threshold'))
            rows.append({k:v for k,v in win.items() if k!='features'}|{'camera_id':int(camera_id),
                'anomaly_score':float(s),'camera_percentile':float(p),
                'camera_validation':'unstable_holdout' if unstable else 'stable_holdout_pilot',
                'event_type':kind})
    write_csv(folder/'motion_anomaly_scores.csv',rows,
        ['track_id','start_s','end_s','valid_step_ratio','keypoint_visibility','camera_id','anomaly_score',
         'camera_percentile','camera_validation','event_type'])
    return rows

def motion_anomaly_preview(frames,tracks,scores,fps,path):
    active=defaultdict(list)
    for s in scores: active[int(s['track_id'])].append(s)
    def render():
        for fi,(frame,people) in enumerate(zip(frames,tracks)):
            canvas=frame.copy(); time_s=fi/fps
            for p in people:
                tid=int(p['id']); candidates=[s for s in active.get(tid,[]) if s['start_s']<=time_s<=s['end_s']]
                x1,y1,x2,y2=np.rint(p['box']).astype(int)
                if candidates:
                    hit=max(candidates,key=lambda s:s['camera_percentile']); pct=hit['camera_percentile']
                    if int(hit['camera_id']) in IF_UNSTABLE_CAMERAS:
                        color=(200,80,200) if pct>=.95 else (140,140,140)
                        label=f'ID {tid} p{pct:.3f} UNVALIDATED'
                    else:
                        color=(0,0,255) if pct>=.99 else ((0,180,255) if pct>=.95 else (255,210,0))
                        label=f'ID {tid} motion p{pct:.3f}'+(' CANDIDATE' if pct>=.99 else '')
                else: color=(140,140,140); label=f'ID {tid} motion: insufficient window'
                cv2.rectangle(canvas,(x1,y1),(x2,y2),color,2)
                cv2.putText(canvas,label,(max(0,x1),max(14,y1-4)),0,.38,color,1)
            yield canvas
    preview_video(path,render(),fps)

def depth_color(depth,lo,hi):
    valid=np.isfinite(depth)&(depth>0)
    gray=np.clip((np.where(valid,depth,lo)-lo)/max(hi-lo,1e-6),0,1)
    color=cv2.applyColorMap((gray*255).astype(np.uint8),cv2.COLORMAP_TURBO)
    color[~valid]=0
    return color

def depth_preview(frames,depths):
    sample=depths[::max(1,len(depths)//20)]
    finite=sample[np.isfinite(sample)&(sample>0)]
    if not finite.size:
        raise RuntimeError('Depth không có pixel hợp lệ.')
    lo,hi=np.percentile(finite,[2,98])
    for image,depth in zip(frames,depths):
        # Color range is fixed for the entire clip, never independently normalized per frame.
        canvas=np.concatenate([image,depth_color(depth,lo,hi)],axis=1)
        cv2.putText(canvas,f'Depth Anything V2 metric estimate | fixed colors {lo:.2f}..{hi:.2f} m',(8,20),0,.5,(255,255,255),1)
        yield canvas

def show_geometry_reference(frames,ref,vg,confidence,K,folder):
    import matplotlib.pyplot as plt
    valid=np.isfinite(vg)&(vg>0)&np.isfinite(confidence)
    if not valid.any():
        raise RuntimeError('VGGT không có depth/confidence hợp lệ.')
    fig,ax=plt.subplots(1,3,figsize=(16,5))
    ax[0].imshow(cv2.cvtColor(frames[ref],cv2.COLOR_BGR2RGB)); ax[0].set_title(f'Reference frame {ref}')
    im=ax[1].imshow(np.where(valid,vg,np.nan),cmap='turbo'); ax[1].set_title('VGGT depth (NOT meters)'); fig.colorbar(im,ax=ax[1])
    im=ax[2].imshow(confidence,cmap='viridis'); ax[2].set_title('Confidence (not probability)'); fig.colorbar(im,ax=ax[2])
    fig.tight_layout(); fig.savefig(folder/'vggt_review.png'); plt.show(); plt.close(fig)
    points=unproject(vg,K,valid)
    points=points[::max(1,len(points)//3000)]
    fig=plt.figure(figsize=(7,5)); ax=fig.add_subplot(111,projection='3d')
    ax.scatter(points[:,0],points[:,2],-points[:,1],s=1,c=points[:,2],cmap='turbo')
    ax.set(xlabel='X',ylabel='Z depth',zlabel='-Y',title='Reference 3D, arbitrary scale')
    fig.savefig(folder/'pointcloud_review.png'); plt.show(); plt.close(fig)
    print('Intrinsics trong tọa độ ảnh xử lý:\n',K)

def show_alignment(frames,depths,dets,ref,surface_depth,background,anchor,surfaces,folder):
    import matplotlib.pyplot as plt
    reports=[]
    for fi,frame in enumerate(frames):
        mask=background&~foreground_mask(dets[fi],surface_depth.shape)&stable_depth_mask(depths[fi])
        static,shift=camera_static_check(frames[ref],frame,mask)
        fit=fit_scale(depths[fi],surface_depth,mask,CFG['min_alignment_pixels'])
        usable=bool(anchor['ok'] and fit['ok'] and static and fit['scale'] is not None and abs(fit['scale']-1)<=CFG['max_scale_drift'])
        reports.append(dict(frame=fi,usable=usable,camera_shift_px=shift,**fit))
    write_json(folder/'alignment_review.json',reports)
    fig,axes=plt.subplots(2,1,figsize=(12,6))
    axes[0].plot([r['scale'] for r in reports]); axes[0].set_ylabel('Scale DA-V2 → reference')
    axes[1].plot([r['median_rel'] for r in reports],label='median relative error')
    axes[1].plot([r['p90_rel'] for r in reports],label='p90 relative error')
    axes[1].axhline(CFG['max_alignment_p90_rel'],color='red',linestyle='--',label='p90 limit')
    axes[1].legend(); axes[1].set_xlabel('Frame'); fig.tight_layout()
    fig.savefig(folder/'alignment_review.png'); plt.show(); plt.close(fig)
    canvas=frames[ref].copy()
    for s in surfaces:
        canvas[s['artifact']['mask']&~s['support']]=(180,0,200)
        canvas[s['support']]=(0,220,70)
    cv2.imwrite(str(folder/'surface_support_review.jpg'),canvas)
    display(Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)))
    print('Xanh: bề mặt quan sát hợp lệ. Tím: vùng mask bị loại, không dùng để đo.')
    print('Reference alignment:',anchor)
    print(f"Frame căn chỉnh đạt: {sum(r['usable'] for r in reports)}/{len(reports)}")
    if not any(r['usable'] for r in reports):
        print('CẢNH BÁO: không có frame đủ tin cậy; phase touch sẽ cho UNKNOWN, không được nới ngưỡng để ép kết quả.')
    return reports

# %% [markdown]
# ### Phase 0 — Chọn video, xem đầu vào
# Đổi `CLIP` thành '1', '2' hoặc '3'. Chạy lại cell này bắt đầu lượt mới, giữ nguyên file lượt cũ.
# Sau khi xong video này, quay lại đây chọn video tiếp theo. Không phải cài lại model/dependency.

# %%
CLIP = '1'  # đổi thành '2' hoặc '3' khi muốn kiểm tra video tiếp theo
assert CLIP in ('1','2','3')
RUN = OUT / (datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')+f'_video_{CLIP}')
RUN.mkdir()
STATE=dict(clip=CLIP,config=json.loads(json.dumps(CFG)),completed=[],ready=[],attempts={})
manifest['config']=json.loads(json.dumps(CFG))
manifest['artifact_segmentation_config']=json.loads(json.dumps(SEGMENTATION_CFG))
write_json(RUN/'manifest.json',manifest)
phase_dir=begin_phase(0)
frames,fps=read_clip(INPUT_FILES[f'{CLIP}.mp4'])
print(f'Video {CLIP}: {len(frames)} frames | {fps:.3f} FPS | {frames.shape[2]}×{frames.shape[1]} | {len(frames)/fps:.2f}s')
preview_video(phase_dir/'input_preview.mp4',iter(frames),fps)
phase_diagnostics(phase_dir,'input',dict(frames=int(len(frames)),fps=float(fps),width=int(frames.shape[2]),height=int(frames.shape[1]),duration_s=round(len(frames)/fps,3)),
    [('PASS','Video đọc được',f'{len(frames)} frames ở {fps:.3f} FPS','Nếu hình/âm thanh/thời lượng không đúng, kiểm tra lại file video và FPS.'),
     ('PASS','Kích thước khung hình',f'{frames.shape[2]} × {frames.shape[1]} px','Nếu ảnh bị xoay/crop bất thường, chuẩn hóa video nguồn trước khi chạy tiếp.')],
    ['input_preview.mp4'])
review_phase(0)

# %% [markdown]
# ### Phase 1 — Detector + ByteTrack DMA/LTC
# Xem video bbox/ID: có mất người, đổi ID hoặc nhận tượng thành người không? Phase này chưa chạy pose.

# %%
phase_dir=begin_phase(1)
raw_tracks,dets=track_people(frames,fps,phase_dir)
preview_video(phase_dir/'tracking_preview.mp4',person_preview(frames,raw_tracks),fps)
print('Số ID:',len({p['id'] for r in raw_tracks for p in r}))
print('Frame có track:',sum(bool(r) for r in raw_tracks),'/',len(frames))
track_ids={p['id'] for r in raw_tracks for p in r}; tracked_frames=sum(bool(r) for r in raw_tracks)
track_coverage=tracked_frames/max(1,len(frames)); det_count=sum(len(d) for d in dets)
phase_diagnostics(phase_dir,'tracking',dict(detections=int(det_count),person_ids=int(len(track_ids)),tracked_frames=int(tracked_frames),frame_coverage=round(track_coverage,4)),
    [('PASS' if det_count else 'FAIL','Person detections',str(det_count),'Nếu bằng 0, kiểm tra detector checkpoint, BGR preprocessing và detector_smoke.jpg.'),
     ('PASS' if track_ids else 'FAIL','Person IDs',str(len(track_ids)),'Nếu không có ID, kiểm tra tracker_report.json và video tracking.'),
     ('PASS' if track_coverage>=.5 else 'WARN','Độ phủ track',f'{track_coverage:.1%} frame có người','Nếu thấp, xem tracking_preview.mp4 để tìm mất detection, ngưỡng hoặc track bị đứt.')],
    ['detector_smoke.jpg','tracking_preview.mp4','tracker_report.json','tracks.json'])
review_phase(1)

# %% [markdown]
# ### Phase 2 — Pose trên người đã có ID
# Xem skeleton và chấm vàng cổ tay: đúng người, đúng tay không? POSE MISSING nghĩa không đủ bằng chứng.

# %%
phase_dir=begin_phase(2)
tracks=estimate_track_poses(frames,raw_tracks,phase_dir)
preview_video(phase_dir/'pose_preview.mp4',person_preview(frames,tracks,with_pose=True),fps)
print('Lượt người có pose:',sum(p['kp'] is not None for r in tracks for p in r),'/',sum(len(r) for r in tracks))
pose_total=sum(len(r) for r in tracks); pose_ok=sum(p['kp'] is not None for r in tracks for p in r)
wrist_ok=sum(p['kp'] is not None and p['kc'] is not None and max(p['kc'][9],p['kc'][10])>=CFG['wrist_conf'] for r in tracks for p in r)
pose_ratio=pose_ok/max(1,pose_total); wrist_ratio=wrist_ok/max(1,pose_total)
phase_diagnostics(phase_dir,'pose',dict(track_observations=int(pose_total),pose_observations=int(pose_ok),pose_coverage=round(pose_ratio,4),wrist_coverage=round(wrist_ratio,4)),
    [('PASS' if pose_ratio>=.7 else 'WARN','Pose ghép vào track',f'{pose_ok}/{pose_total} ({pose_ratio:.1%})','Nếu thấp, xem pose_preview.mp4 và kiểm tra crop/ghép bbox; đừng diễn giải missing pose thành hành vi.'),
     ('PASS' if wrist_ratio>=.5 else 'WARN','Cổ tay đủ confidence',f'{wrist_ok}/{pose_total} ({wrist_ratio:.1%})','Nếu thấp, cổ tay thường bị che/nhỏ; xem chấm vàng và giữ kết luận UNKNOWN khi thiếu bằng chứng.')],
    ['pose_preview.mp4','tracks_pose.json'])
review_phase(2)

# %% [markdown]
# ### Phase 3 — Chấm bất thường pose/chuyển động bằng Isolation Forest
# Model vừa train trên normal windows và đã xem đánh giá holdout. Chọn camera ID đúng với
# camera nguồn để dùng percentile reference tương ứng. Điểm cao chỉ là unusual-motion candidate.

# %%
phase_dir=begin_phase(3)
ANOMALY_CAMERA_ID=int(CLIP)  # đổi nếu số clip không tương ứng camera 1/2/3 trong train set
live_if_windows=build_live_if_windows(frames,tracks,fps)
motion_scores=score_live_if_windows(live_if_windows,ANOMALY_CAMERA_ID,phase_dir)
motion_anomaly_preview(frames,tracks,motion_scores,fps,phase_dir/'motion_anomaly_preview.mp4')
print(f'Cửa sổ hợp lệ: {len(motion_scores)} | unusual-motion candidates (p99):',
      sum(r['event_type']=='unusual_motion_candidate' for r in motion_scores))
if ANOMALY_CAMERA_ID in IF_UNSTABLE_CAMERAS:
    print(f'CẢNH BÁO: camera {ANOMALY_CAMERA_ID} holdout không ổn định; điểm chỉ để review, không sinh candidate.')
if motion_scores: display(pd.DataFrame(motion_scores).drop(columns='features',errors='ignore'))
else: print('Không đủ pose liên tục 2.5 giây để chấm điểm; không suy ra hành vi bình thường/bất thường.')
motion_candidates=sum(r['event_type']=='unusual_motion_candidate' for r in motion_scores)
phase_diagnostics(phase_dir,'motion_anomaly',dict(valid_windows=int(len(motion_scores)),unusual_candidates=int(motion_candidates),camera_status='unstable_holdout_review_only' if ANOMALY_CAMERA_ID in IF_UNSTABLE_CAMERAS else 'stable_holdout_pilot'),
    [('PASS' if motion_scores else 'WARN','Cửa sổ chuyển động đủ chất lượng',str(len(motion_scores)),'Nếu 0, kiểm tra pose coverage/track liên tục; clip ngắn hoặc thiếu pose sẽ không tạo score.'),
     ('WARN' if ANOMALY_CAMERA_ID in IF_UNSTABLE_CAMERAS else 'PASS','Độ tin cậy camera của Isolation Forest','unstable / review-only' if ANOMALY_CAMERA_ID in IF_UNSTABLE_CAMERAS else 'stable holdout pilot','Camera drift chỉ cho điểm tham khảo; score cao không đồng nghĩa touch hay vi phạm.')],
    ['motion_anomaly_preview.mp4','motion_anomaly_scores.csv'])
review_phase(3)

# %% [markdown]
# ### Phase 4 — Mask hiện vật
# Xem lần lượt bbox detector, mask cuối, candidate bị loại. Nếu box đúng nhưng mask sai thì lỗi ở SAM;
# nếu box thiếu/sai thì lỗi ở detector/prompt. Đây là segmentation tự động, chưa phải ground-truth.

# %%
phase_dir=begin_phase(4)
ref,keyframes,artifacts=artifact_masks(frames,dets,phase_dir)
artifact_run=json.loads((phase_dir/'artifact_detections.json').read_text())
print('Segmentation config:',artifact_run['config'],'| Model:',artifact_run['model'],'| Reference frame:',ref,'| VGGT keyframes:',keyframes,
      '| Positive candidates:',artifact_run['positive_candidates'],'| Accepted masks:',len(artifacts),
      '| Rejected detections/masks:',len(artifact_run['rejected_detections']),'/',len(artifact_run['rejected_masks']))
print([(a['id'],a['label']) for a in artifacts])
phase_diagnostics(phase_dir,'artifacts',dict(reference_frame=int(ref),keyframes=keyframes,config=artifact_run['config'],model=artifact_run['model'],
    raw_detections=int(artifact_run['detections']),positive_candidates=int(artifact_run['positive_candidates']),
    material_rescue_candidates=int(artifact_run['material_rescue_candidates']),dropped_empty_pedestals=int(artifact_run['dropped_empty_pedestals']),
    accepted_masks=int(len(artifacts)),rejected_detections=int(len(artifact_run['rejected_detections'])),
    rejected_masks=int(len(artifact_run['rejected_masks']))),
    [('PASS' if artifacts else 'FAIL','Mask hiện vật',f'{len(artifacts)} masks','Nếu mask sai/lẫn người/nền, dừng và xử lý prompt/reference hoặc mask trước khi đo khoảng cách.'),
     ('PASS','Reference frame',str(ref),'Chọn frame ít bị người che; mask/reference không đại diện thì chạy lại phase này.')],
    ['artifact_detections_review.jpg','artifact_review.jpg','artifact_rejected_review.jpg','artifact_detections.json','artifacts.json','reference.jpg'])
review_phase(4)

# %% [markdown]
# ### Phase 5 — VGGT: depth, confidence và hình học 3D
# Xem bề mặt có méo hoặc đứt bất thường không. VGGT ở phase này chưa có thang mét.

# %%
phase_dir=begin_phase(5)
vg,confidence,K=vggt_reference(frames,dets,keyframes,phase_dir)
show_geometry_reference(frames,ref,vg,confidence,K,phase_dir)
vgood=np.isfinite(vg)&(vg>0); cgood=np.isfinite(confidence); valid_ratio=float(vgood.mean()); conf_mean=float(np.nanmean(confidence)) if cgood.any() else 0.
phase_diagnostics(phase_dir,'vggt',dict(depth_shape=list(vg.shape),valid_depth_fraction=round(valid_ratio,4),mean_confidence=round(conf_mean,5),intrinsics_valid=bool(np.isfinite(K).all() and K[0,0]>0 and K[1,1]>0)),
    [('PASS' if valid_ratio>=.8 else 'WARN','Depth VGGT hợp lệ',f'{valid_ratio:.1%} pixels','Nếu thấp hoặc depth/point cloud méo, kiểm tra vggt_review.png và keyframes.'),
     ('PASS' if np.isfinite(K).all() and K[0,0]>0 and K[1,1]>0 else 'FAIL','Camera intrinsics','hợp lệ' if np.isfinite(K).all() and K[0,0]>0 and K[1,1]>0 else 'không hợp lệ','Nếu FAIL, không tiếp tục unproject/đo 3D; kiểm tra preprocessing và VGGT output.')],
    ['vggt_review.png','pointcloud_review.png','vggt_reference.npz'])
review_phase(5)

# %% [markdown]
# ### Phase 6 — Depth Anything V2 Metric Indoor Small
# Xem video RGB cạnh depth: từng frame có độ sâu hợp lý không? Màu dùng chung thang cho cả clip.
# Model được huấn luyện Hypersim; mét là ước lượng chuyển domain, chưa hiệu chuẩn tại bảo tàng.

# %%
phase_dir=begin_phase(6)
depths=depth_anything_v2_depth(frames,phase_dir)
preview_video(phase_dir/'depth_anything_v2_preview.mp4',depth_preview(frames,depths),fps)
print('Depth shape:',depths.shape,'| Tỷ lệ pixel hữu hạn, dương:',float((np.isfinite(depths)&(depths>0)).mean()))
depth_valid=float((np.isfinite(depths)&(depths>0)).mean()); depth_sample=depths[np.isfinite(depths)&(depths>0)]
depth_p50=float(np.median(depth_sample)) if depth_sample.size else 0.
phase_diagnostics(phase_dir,'depth_anything_v2',dict(shape=list(depths.shape),valid_positive_fraction=round(depth_valid,4),median_depth_estimate_m=round(depth_p50,4)),
    [('PASS' if depth_valid>=.95 else 'WARN','Pixel depth hợp lệ',f'{depth_valid:.1%}','Nếu thấp, kiểm tra frame lỗi và depth preview; không coi depth estimate là đo đã hiệu chuẩn.'),
     ('PASS' if depth_sample.size and 0<depth_p50<100 else 'FAIL','Thang depth ước lượng',f'median {depth_p50:.3f} m','Nếu giá trị không hợp lý, dừng và kiểm tra model/checkpoint/input size.')],
    ['depth_anything_v2_preview.mp4','depth_anything_v2_metric_indoor.npy'])
review_phase(6)

# %% [markdown]
# ### Phase 7 — Căn chỉnh hai model và bề mặt dùng để đo
# Xem biểu đồ lỗi, số frame đạt và vùng bề mặt còn quan sát được. Không có bề mặt tin cậy thì không kết luận touch.

# %%
phase_dir=begin_phase(7)
surface_depth,background,anchor,surfaces=build_geometry(vg,confidence,K,depths,ref,dets,artifacts,phase_dir)
alignment_review=show_alignment(frames,depths,dets,ref,surface_depth,background,anchor,surfaces,phase_dir)
alignment_ok=sum(bool(r['usable']) for r in alignment_review); alignment_ratio=alignment_ok/max(1,len(alignment_review)); surface_pixels=[int(s['support'].sum()) for s in surfaces]
phase_diagnostics(phase_dir,'alignment',dict(anchor_ok=bool(anchor['ok']),usable_frames=int(alignment_ok),total_frames=int(len(alignment_review)),usable_fraction=round(alignment_ratio,4),surface_support_pixels=surface_pixels),
    [('PASS' if anchor['ok'] else 'FAIL','Căn chỉnh reference',str(anchor),'Nếu FAIL, không dùng khoảng cách; xem reference_alignment.json và background/reference.'),
     ('PASS' if alignment_ok else 'FAIL','Frame đủ điều kiện căn chỉnh',f'{alignment_ok}/{len(alignment_review)}','Nếu 0, touch phải UNKNOWN; xem alignment_review.png, camera shift và depth disagreement.'),
     ('PASS' if surface_pixels and max(surface_pixels)>0 else 'FAIL','Pixel bề mặt hiện vật được hỗ trợ',str(surface_pixels),'Nếu 0, xem surface_support_review.jpg; mask/độ sâu chưa đủ để tính khoảng cách.')],
    ['alignment_review.png','surface_support_review.jpg','alignment_review.json','surface_quality.json'])
review_phase(7)

# %% [markdown]
# ### Phase 8 — Touch candidates và ảnh sự kiện
# Xem ID nghi chạm, thời gian và từng ảnh bằng chứng. UNKNOWN không đồng nghĩa không chạm.

# %%
phase_dir=begin_phase(8)
summary=render_and_score(frames,fps,tracks,dets,depths,K,ref,surface_depth,background,anchor,surfaces,
                         phase_dir,motion_scores=motion_scores,camera_id=ANOMALY_CAMERA_ID)
display(Video(str(phase_dir/'preview.mp4'),embed=True,width=800))
print(json.dumps(summary,indent=2,ensure_ascii=False))
events=json.loads((phase_dir/'events.json').read_text())
for event in events:
    print(event)
    display(Image.open(phase_dir/event['evidence']))
if not events:
    print('Không có candidate. Hãy kiểm tra số UNKNOWN và alignment trước khi đánh giá model.')
obs=pd.read_csv(phase_dir/'observations.csv')
obs_counts={str(k):int(v) for k,v in obs['status'].value_counts().items()} if len(obs) and 'status' in obs else {}
unknown_ratio=obs_counts.get('UNKNOWN',0)/max(1,len(obs))
phase_diagnostics(phase_dir,'touch',dict(events=int(summary['events']),observations=int(len(obs)),status_counts=obs_counts,unknown_fraction=round(unknown_ratio,4),metric_calibrated=bool(summary.get('metric_calibrated',False))),
    [('PASS' if summary['events'] else 'WARN','Số touch candidates',str(summary['events']),'Nếu 0, kiểm tra alignment và UNKNOWN trước; không nới ngưỡng chỉ để tạo candidate.'),
     ('WARN' if unknown_ratio>.5 else 'PASS','Tỷ lệ quan sát UNKNOWN',f'{unknown_ratio:.1%}','Nếu cao, lần ngược pose, depth, che khuất và alignment; UNKNOWN không đồng nghĩa NO_TOUCH.'),
     ('WARN','Độ hiệu chuẩn mét','chưa hiệu chuẩn tại bảo tàng','Candidate là kết quả cần review, không phải xác nhận touch; cần đo chuẩn/nhãn thực tế để hiệu chuẩn.')],
    ['preview.mp4','events.csv','events.json','observations.csv','summary.json'] )
review_phase(8)

# %% [markdown]
# ## Tải kết quả sau khi chạy xong
# ZIP chứa preview/báo cáo của từng phase và nhật ký chạy. Với video tiếp theo: quay lại phase 0, đổi CLIP.

# %%
assert set(PHASE_NAMES).issubset(set(STATE['completed'])), 'Chưa chạy xong đủ các phase.'
import zipfile
archive=ROOT/f'{RUN.name}_vggt_depth_anything_v2_touch_results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in RUN.rglob('*'):
        if p.is_file():
            z.write(p,p.relative_to(RUN))
    z.write(INPUT_FILES['isolation_forest_normal_yolov8_pose.joblib'],'isolation_forest_model.joblib')
    z.write(INPUT_FILES['evaluation.json'],'isolation_forest_evaluation.json')
from IPython.display import FileLink
display(FileLink(str(archive)))

# %% [markdown]
# Nguồn API: https://github.com/facebookresearch/vggt ; https://github.com/DepthAnything/Depth-Anything-V2 ;
# https://github.com/khangkaka066/ByteTrack-DMA-LTC-Motion-Tracker ; https://github.com/facebookresearch/sam2
# VGGT gốc, không phải Omega. Giữ UniDepth cũ làm baseline riêng; notebook này không sửa/xóa pipeline cũ.
