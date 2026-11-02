"""Exactly four synthetic cases; zero network/model/original-image reads."""
import hashlib
import http.client
import io
import json
from pathlib import Path
import sys
import time
from email.message import Message
import unittest
import os
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import acquire_helmet_prospective as prod
import audit_helmet_prospective as cons


def put(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);data=json.dumps(obj).encode();path.write_bytes(data)
    return {'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}


def case_exif_display_crop_matches_independent_consumer(tmp_path):
    # Two tiny images: an Orientation6 JPEG and a metadata-free PNG.
    im=Image.new('RGB',(8,4));im.putdata([(x*20,y*40,15) for y in range(4) for x in range(8)])
    exif=Image.Exif();exif[274]=6;exif[315]='private fixture metadata';path=tmp_path/'oriented.jpg';im.save(path,exif=exif)
    sample={'native_annotation':{'bbox':[0,0,3,4]}}
    a,ga=prod.geometry(path,sample);b,gb=cons.crop_original(path,sample)
    assert ga==gb and ga['display_size']==[4,8] and ga['exif_orientation']==6 and a.tobytes()==b.tobytes()
    q=tmp_path/'clean.png';a.save(q);assert not Image.open(q).getexif()
    with unittest.TestCase().assertRaisesRegex(ValueError, r'bounds'):cons.crop_original(path,{'native_annotation':{'bbox':[3,7,2,2]}})


class FakeResponse:
    def __init__(self,url,code,body,extra=(),error=None):
        self.url,self.code,self.body,self.error=url,code,body,error;self.headers=Message();self.headers['Content-Length']=str(len(body) if error is None else 100);self.headers['Content-Type']='image/jpeg'
        for k,v in extra:self.headers[k]=v
    def geturl(self):return self.url
    def close(self):pass
    def read1(self,n):
        if self.error:raise self.error
        b,self.body=self.body[:n],self.body[n:];return b


def capture(tmp_path,monkeypatch,responses):
    selection={'samples':[{'publisher_file':{'content_details':{'download_url':'https://data.mendeley.com/public-files/datasets/j82bnw7gsr/files/synthetic/file_downloaded'}}}]}
    monkeypatch.setattr(prod,'preflight',lambda _:selection);monkeypatch.setattr(prod.transport,'boot',lambda:'fixture')
    plan={'new_caps':prod.CAPS,'producer_seconds':1380,'storage_roots':[str(tmp_path)]};claim={'start_clock':prod.clock(),'boot_uuid':'fixture'}
    cap=prod.FixedCapture(tmp_path,plan,claim)
    class Opener:
        def open(self,req,timeout):
            response=responses.pop(0);assert req.full_url==response.url;return response
    cap.opener=Opener();return cap


def case_literal_single_redirect_and_no_duplicate_retry(tmp_path,monkeypatch):
    initial='https://data.mendeley.com/public-files/datasets/j82bnw7gsr/files/synthetic/file_downloaded';target='https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/synthetic'
    payload=b'not-decoded-transport-fixture';responses=[FakeResponse(initial,302,b'', [('Location',target),('Server-Timing','a'),('Server-Timing','b')]),FakeResponse(target,200,payload)]
    cap=capture(tmp_path,monkeypatch,responses);index=cap.asset(initial,'image_sample_00',65536,len(payload),hashlib.sha256(payload).hexdigest())
    assert index==1 and len(cap.records)==2 and cap.records[1]['redirect_from']==0 and not responses
    with unittest.TestCase().assertRaisesRegex(ValueError, r'duplicate'):cap.request(initial,'image_sample_00',65536)
    records,bill=cons.bill_prefix(tmp_path);assert bill['request_claims']==2 and bill['retained_body_bytes']==len(payload) and bill['interrupted_read_upper_bound']==0
    assert cons.headers(records[0][1]['header_pairs'])['server-timing']=='a\nb'


def case_incomplete_read_prefix_and_uncertainty_are_charged(tmp_path,monkeypatch):
    url='https://data.mendeley.com/public-files/datasets/j82bnw7gsr/files/synthetic/file_downloaded';error=http.client.IncompleteRead(b'xyz',97)
    cap=capture(tmp_path,monkeypatch,[FakeResponse(url,200,b'',error=error)])
    with unittest.TestCase().assertRaises(http.client.IncompleteRead):cap.request(url,'image_sample_00',65536)
    _,bill=cons.bill_prefix(tmp_path);assert bill['retained_body_bytes']==3 and bill['interrupted_read_upper_bound']==65533 and bill['request_claims']==1
    assert cap.records[0]['status']=='failed'
    with unittest.TestCase().assertRaises(ValueError):cap.request(url,'image_sample_00',65536)
    assert len(cap.records)==1


def case_complete_native_metadata_and_gt_identity_not_producer_labels(tmp_path):
    # Full-sized metadata shape but no image files; 24 selected synthetic GT rows.
    images=[{'id':i,'file_name':'I%d.jpg'%i,'width':128,'height':128} for i in range(2392)]
    anns=[{'id':i,'image_id':i%2392,'category_id':2 if i%2 else 3,'bbox':[0,0,64,64]} for i in range(8275)]
    coco=put(tmp_path/'coco.json',{'images':images,'annotations':anns,'categories':[{'id':2,'name':'NoHelmet'},{'id':3,'name':'Helmet'}]})
    rows=[]
    for i in range(2392):
        fid='f%d'%i;rows.append({'id':fid,'filename':'I%d.jpg'%i,'folder_id':'39245ed4-43d6-497c-9fd9-45adca1a6ca0','size':100,'content_details':{'size':100,'sha256_hash':hashlib.sha256(str(i).encode()).hexdigest(),'download_url':f'https://data.mendeley.com/public-files/datasets/j82bnw7gsr/files/{fid}/file_downloaded'}})
    pages=[]
    for i in range(3):
        part=rows[i*1000:(i+1)*1000];body=put(tmp_path/('page%d.json'%i),part);url='https://data.mendeley.com/public-api/datasets/j82bnw7gsr/files?'+cons.urlencode({'folder_id':'39245ed4-43d6-497c-9fd9-45adca1a6ca0','version':1,'$start':i*1000,'$limit':1000})
        receipt=put(tmp_path/('receipt%d.json'%i),{'url':url,'final_url':url,'status':'complete','status_code':[206,206,200][i],'body_sha256':body['sha256'],'retained_bytes':body['bytes'],'header_pairs':[['content-range',f'items {i*1000}-{min(i*1000+999,2391)}/2392'],['content-length',str(body['bytes'])]]})
        pages.append({'body':body,'receipt':receipt,'exact_url':url})
    old=put(tmp_path/'old.json',{'samples':[{'image_id':3000+i,'image_sha256':hashlib.sha256(('old%d'%i).encode()).hexdigest()} for i in range(6)]})
    samples=[]
    for i in range(24):
        samples.append({'sample_id':'prospective_%02d'%i,'image_id':i,'annotation_id':i,'native_annotation':anns[i],'native_image':images[i],'publisher_file':rows[i],'category_name':'Helmet' if i%2==0 else 'NoHelmet','selection_hash':hashlib.sha256(f'helmet-prospective-20261102-v1|{i}|{i}'.encode()).hexdigest()})
    selection={'source_provenance':{'coco':coco,'metadata_pages':pages,'old_six_manifest':old},'samples':samples};assert len(cons.native_inputs(selection))==2392
    samples[0]['category_name']='NoHelmet'
    with unittest.TestCase().assertRaisesRegex(ValueError, r'association'):cons.native_inputs(selection)
    with unittest.TestCase().assertRaisesRegex(ValueError, r'conflict'):cons.headers([['Content-Length','1'],['content-length','2']])


class Patch:
    def __init__(self): self.saved=[]
    def setattr(self, obj, name, value):
        self.saved.append((obj,name,getattr(obj,name)));setattr(obj,name,value)
    def restore(self):
        for obj,name,value in reversed(self.saved):setattr(obj,name,value)


class Helmet24Synthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ['HELMET24_SOFTWARE_FIXTURES']);cls.root.mkdir(exist_ok=False)
    def path(self):
        p=self.root/self._testMethodName;p.mkdir(exist_ok=False);return p
    def test_01_exif(self):case_exif_display_crop_matches_independent_consumer(self.path())
    def test_02_redirect(self):
        patch=Patch()
        try:case_literal_single_redirect_and_no_duplicate_retry(self.path(),patch)
        finally:patch.restore()
    def test_03_partial(self):
        patch=Patch()
        try:case_incomplete_read_prefix_and_uncertainty_are_charged(self.path(),patch)
        finally:patch.restore()
    def test_04_native(self):case_complete_native_metadata_and_gt_identity_not_producer_labels(self.path())


if __name__=='__main__':unittest.main(verbosity=2)
