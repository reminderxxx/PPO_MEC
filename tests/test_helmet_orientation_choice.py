"""Four new bounded software cases; no pretrained model, network or old audit."""
import hashlib
import io
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_helmet_orientation_choice as q


def test_exif_display_rois_against_numpy_rotation(tmp_path):
    yy,xx=np.mgrid[:64,:96];raw=np.stack([xx*2,yy*3,(xx+yy)%255],axis=-1).astype('uint8')
    for orientation,k in [(1,0),(3,2),(6,-1),(8,1)]:
        path=tmp_path/('o%d.jpg'%orientation);e=Image.Exif();e[274]=orientation;Image.fromarray(raw).save(path,exif=e)
        native=np.asarray(Image.open(path).convert('RGB'));expected=np.rot90(native,k)[4:36,3:35,:]
        binary,sha,size,o=q.crop(path,[3,4,32,32],'display')
        assert o==orientation and size==[32,32] and sha==hashlib.sha256(expected.tobytes()).hexdigest()
        assert np.array_equal(np.asarray(Image.open(io.BytesIO(binary))),expected)


def test_gray_and_invalid_bbox(tmp_path):
    path=tmp_path/'source.png';Image.new('RGB',(64,64),'red').save(path)
    _,rgb,size,_=q.crop(path,[2,3,32,32],'gray')
    assert size==[32,32] and rgb==hashlib.sha256(bytes([128])*(32*32*3)).hexdigest()
    with pytest.raises(ValueError):q.crop(path,[2.5,3,32,32],'display')
    with pytest.raises(ValueError):q.crop(path,[50,3,32,32],'display')


def test_fixed_opaque_matrix_and_no_gt():
    s={'arm':'base','runtime':{},'resources':{},'source_commit':'x','source_sha':{},'caps':q.CAPS,'prompts':q.PROMPTS,'candidate_ids':q.IDS,'generation':q.h.GENERATION,'generation_prompt':q.h.PROMPT,'inputs':[{'input_id':n,'case_id':n[:3],'view':n.split('_',1)[1],'path':'/opaque/'+n+'.png','sha256':'x','rgb_sha256':'x','size':[32,32]} for n in q.INPUTS]}
    q.worker_schema(s);s['inputs'][0]['GT']='helmet'
    with pytest.raises(ValueError):q.worker_schema(s)
    del s['inputs'][0]['GT'];s['inputs']=s['inputs'][:-1]
    with pytest.raises(ValueError):q.worker_schema(s)


def test_real_forward_hook_under_declared_toy_call(tmp_path):
    import torch
    old=q.h._BUDGET;q.h._BUDGET=q.h.Budget([tmp_path],1048576,1048576,reserve=1024)
    calls=q.Calls(tmp_path/'ops.jsonl','base');model=torch.nn.Linear(2,2).eval();model.register_forward_pre_hook(calls.pre);model.register_forward_hook(calls.post)
    try:
        with calls.call('score_forward','toy'),torch.inference_mode():model(torch.ones((1,2)))
        assert calls.count=={'score_forward':1,'native_forward':1}
        with pytest.raises(ValueError):model(torch.ones((1,2)))
    finally:calls.stream.close();q.h._BUDGET=old
