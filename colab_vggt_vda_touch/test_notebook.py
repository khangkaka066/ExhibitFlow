"""CPU-only regression checks. Does NOT execute model downloads or Colab cells."""
import ast
import csv
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from collections import deque
from datetime import datetime
import numpy as np
import cv2
from scipy.spatial import cKDTree
import nbformat

ROOT=Path(__file__).resolve().parent
tree=ast.parse((ROOT/'notebook_source.py').read_text())
scope=globals().copy()
# Load only pure functions and CFG; never run installation/upload/inference cells.
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
nodes += [n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('CFG','PHASE_NAMES') for t in n.targets)]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<notebook-functions>','exec'),scope)

class NotebookTests(unittest.TestCase):
    def test_phase_completion_and_upstream_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            scope['RUN']=Path(tmp)
            state=dict(config=scope['CFG'].copy(),completed=[],ready=[],attempts={})
            scope['STATE']=state
            with self.assertRaises(RuntimeError):
                scope['begin_phase'](1)
            scope['begin_phase'](0)
            scope['review_phase'](0)
            self.assertEqual(state['completed'],['input'])
            scope['begin_phase'](1)
            scope['review_phase'](1)
            self.assertEqual(state['completed'],['input','tracking'])
            scope['begin_phase'](2)
            scope['review_phase'](2)
            old_attempt=state['attempts']['tracking']
            scope['begin_phase'](1)
            self.assertEqual(state['completed'],['input'])
            self.assertNotIn('pose',state['ready'])
            self.assertNotIn('pose',state['attempts'])
            self.assertTrue((Path(tmp)/old_attempt).exists())
            state['config']={}
            with self.assertRaises(RuntimeError):
                scope['begin_phase'](1)

    def test_person_and_depth_preview(self):
        frames=np.zeros((2,80,120,3),np.uint8)
        person=dict(id=7,box=[10,10,60,70],kp=[[30,40]]*17,kc=[.9]*17)
        out=list(scope['person_preview'](frames,[[person],[dict(person,kp=None)]],True))
        self.assertEqual(out[0].shape,frames[0].shape)
        self.assertTrue(out[0].any())
        depth=np.ones((2,80,120),np.float32)
        previews=list(scope['depth_preview'](frames,depth))
        self.assertEqual(previews[0].shape,(80,240,3))
        self.assertEqual(len(previews),2)

    def test_detector_preprocess_keeps_bgr_pixel_scale(self):
        frame=np.zeros((10,20,3),np.uint8); frame[:]=[20,40,200]
        image,ratio=scope['detector_preprocess'](frame,(32,32),'bgr255')
        self.assertAlmostEqual(ratio,1.6)
        np.testing.assert_allclose(image[:,0,0],[20,40,200])
        np.testing.assert_allclose(image[:,31,31],[114,114,114])
    def test_notebook_schema_and_python(self):
        nb=nbformat.read(ROOT/'touch_vggt_vda_bytetrack_kaggle.ipynb',as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type=='code':
                ast.parse(cell.source)
                self.assertEqual(cell.outputs,[])

    def test_scale_recovered_and_holdout_rejects_mismatch(self):
        source=np.linspace(1,5,80*80).reshape(80,80)
        mask=np.ones_like(source,bool)
        fit=scope['fit_scale'](source,source*3,mask)
        self.assertTrue(fit['ok']); self.assertAlmostEqual(fit['scale'],3)
        target=source*3
        yy,xx=np.indices(source.shape)
        target[(xx+yy)%2==1]*=1.5
        self.assertFalse(scope['fit_scale'](source,target,mask)['ok'])

    def test_invalid_alignment_and_spatial_support(self):
        d=np.ones((80,80))
        self.assertFalse(scope['fit_scale'](d,d,np.zeros_like(d,bool))['ok'])
        d[0,0]=np.nan
        self.assertTrue(scope['fit_scale'](d,d,np.ones_like(d,bool))['ok'])
        mask=np.zeros((160,160),bool); mask[:39,:39]=True
        self.assertFalse(scope['fit_scale'](np.ones(mask.shape),np.ones(mask.shape),mask)['ok'])

    def test_unprojection(self):
        d=np.full((8,8),2.)
        K=np.array([[4,0,4],[0,4,4],[0,0,1.]])
        mask=np.zeros_like(d,bool); mask[4,6]=True
        np.testing.assert_allclose(scope['unproject'](d,K,mask),[[1,0,2]])

    def test_intrinsics_pad_roundtrip(self):
        w,h=640,400; nw,nh=518,322; top=(518-nh)//2
        A=np.array([[nw/w,0,0],[0,nh/h,top],[0,0,1.]])
        K=np.array([[700,0,320],[0,700,200],[0,0,1.]])
        np.testing.assert_allclose(np.linalg.inv(A)@(A@K),K,atol=1e-6)

    def test_foreground_excluded_from_reference_surface(self):
        h,w=100,100; depth=np.full((h,w),4.,np.float32)
        mask=np.zeros((h,w),bool); mask[30:70,30:70]=True
        dets=[np.array([[40,30,55,65,.9]])]
        a=dict(id='artifact_1',mask=mask)
        with tempfile.TemporaryDirectory() as tmp:
            _,_,anchor,surfaces=scope['build_geometry'](depth,np.ones_like(depth),np.eye(3),depth[None],0,dets,[a],Path(tmp))
        fg=scope['foreground_mask'](dets[0],(h,w))
        self.assertFalse((surfaces[0]['support']&fg).any())
        self.assertTrue(anchor['ok'])
        self.assertIsNotNone(surfaces[0]['tree'])

    def test_wrist_depth_edge_rejected(self):
        d=np.ones((12,12),np.float32)
        point,spread=scope['wrist_point'](d,np.eye(3),(6,6))
        np.testing.assert_allclose(point,[6,6,1])
        d[:,6:]=3
        self.assertIsNone(scope['wrist_point'](d,np.eye(3),(6,6))[0])
        self.assertIsNone(scope['wrist_point'](d,np.eye(3),(-1,6))[0])

    def test_temporal_no_stale_candidate_or_cross_wrist(self):
        gate=scope['temporal_gate']; history={}; key=(2,'artifact_1',9)
        for fi in range(3):
            state=gate(history,key,fi,True,True,.1,24)
        self.assertEqual(state,'TOUCH_CANDIDATE')
        self.assertEqual(gate(history,(2,'artifact_1',10),3,True,True,.1,24),'UNKNOWN')
        self.assertEqual(gate(history,key,3,False,False,None,24),'UNKNOWN')
        self.assertNotIn(key,history)
        self.assertEqual(gate(history,key,4,True,False,.1,24),'NO_TOUCH')

    def test_fast_pass_not_candidate(self):
        history={}
        for fi in range(20):
            self.assertNotEqual(scope['temporal_gate'](history,(1,'a',9),fi,True,True,3.,24),'TOUCH_CANDIDATE')

    def test_frame_gap_resets_temporal_history(self):
        history={}; key=(1,'a',9); gate=scope['temporal_gate']
        gate(history,key,0,True,True,.1,24)
        gate(history,key,1,True,True,.1,24)
        self.assertEqual(gate(history,key,8,True,True,.1,24),'UNKNOWN')
        self.assertEqual(len(history[key]),1)

    def test_synthetic_render_events_and_missing_track(self):
        h,w=100,160; n=7
        frames=np.full((n,h,w,3),100,np.uint8)
        depth=np.ones((n,h,w),np.float32)
        K=np.array([[100,0,80],[0,100,50],[0,0,1.]])
        mask=np.zeros((h,w),bool); mask[25:75,50:110]=True
        background=~mask
        a=dict(id='artifact_1',mask=mask,box=[50,25,110,75])
        surface=dict(artifact=a,tree=cKDTree(scope['unproject'](depth[0],K,mask)),
            support=mask,uncertainty=0.,image_dt=cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,3),
            support_dt=cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,3))
        person=dict(id=2,box=[60,15,100,90],kp=[[80,50]]*17,kc=[.95]*17)
        tracks=[[person] for _ in range(5)]+[[],[]]
        dets=[np.empty((0,5),np.float32) for _ in range(n)]
        real_camera_check=scope['camera_static_check']
        scope['camera_static_check']=lambda *args:(True,0.)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp)
                report=scope['render_and_score'](frames,24,tracks,dets,depth,K,0,depth[0],background,
                    {'ok':True},[surface],folder)
                events=json.loads((folder/'events.json').read_text())
                self.assertEqual(report['events'],1)
                self.assertEqual(events[0]['person_id'],2)
                self.assertEqual(events[0]['end_frame'],4)
                self.assertTrue((folder/events[0]['evidence']).exists())
                self.assertTrue((folder/'preview.mp4').exists())
                cap=cv2.VideoCapture(str(folder/'preview.mp4'))
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),n)
                cap.release()
        finally:
            scope['camera_static_check']=real_camera_check

    def test_static_camera_checker(self):
        rng=np.random.default_rng(4)
        image=rng.integers(0,256,(240,320,3),dtype=np.uint8)
        mask=np.ones((240,320),bool)
        self.assertTrue(scope['camera_static_check'](image,image,mask)[0])
        shifted=cv2.warpAffine(image,np.array([[1,0,12],[0,1,0]],np.float32),(320,240))
        self.assertFalse(scope['camera_static_check'](image,shifted,mask)[0])

if __name__=='__main__':
    unittest.main(verbosity=2)
