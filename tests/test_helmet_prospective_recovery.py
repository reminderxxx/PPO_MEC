"""Four bounded micro cases only; no real network, image or model acquisition."""
import hashlib
import http.client
import json
import os
from pathlib import Path
import sys
import unittest
from email.message import Message
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import recover_helmet_prospective as prod
import audit_helmet_prospective_recovery as cons


class FakeResponse:
    def __init__(self,url,payload=b'full-file',error=None):
        self.code=200;self.url=url;self.payload=payload;self.error=error;self.headers=Message();self.headers['Content-Type']='image/jpeg';self.headers['Content-Length']=str(len(payload)if error is None else 100)
    def geturl(self):return self.url
    def close(self):pass
    def read1(self,n):
        if self.error:raise self.error
        b,self.payload=self.payload[:n],self.payload[n:];return b


class Patch:
    def __init__(self):self.restorelist=[]
    def set(self,obj,k,v):self.restorelist.append((obj,k,getattr(obj,k)));setattr(obj,k,v)
    def restore(self):
        for obj,k,v in reversed(self.restorelist):setattr(obj,k,v)


class RecoveryFour(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.root=Path(os.environ['HELMET24_RECOVERY_FIXTURES']);cls.root.mkdir(exist_ok=False)
    def setUp(self):self.path=self.root/self._testMethodName;self.path.mkdir(exist_ok=False);self.patch=Patch()
    def tearDown(self):self.patch.restore()
    def capture(self,error=None):
        literal='https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/frozen-sample06'
        selection={'samples':[{'publisher_file':{'content_details':{'download_url':'https://data.mendeley.com/public-files/datasets/j82bnw7gsr/files/sample%02d/file_downloaded'%i}}}for i in range(24)]}
        plan={'storage_roots':[str(self.path)]};claim={'start_clock':prod.clock(),'boot_uuid':'fixture'}
        self.patch.set(prod,'preflight',lambda *a,**k:None);self.patch.set(prod.transport,'boot',lambda:'fixture')
        cap=prod.RecoveryCapture(self.path,plan,claim,selection,literal);calls=[]
        class Opener:
            def open(self,request,timeout):calls.append((request.full_url,timeout));return FakeResponse(request.full_url,error=error)
        cap.opener.delegate=Opener();return cap,literal,calls
    def test_01_inherited_s3_uses120_not_old60(self):
        cap,literal,calls=self.capture();index=cap.request(literal,'image_sample_06',65536,redirect_from=-1)
        self.assertEqual(index,0);self.assertEqual(calls,[(literal,120)]);self.assertEqual(cap.records[0]['redirect_from'],-1)
        control=json.loads((self.path/'open_control_000.json').read_text());self.assertEqual(control['helper_timeout_argument'],60);self.assertEqual(control['effective_socket_open_and_read_timeout'],120)
        self.assertEqual(cap.records[0]['bytes'],9)
        with self.assertRaises(ValueError):cap.request(literal,'image_sample_06',65536,redirect_from=-1)
        self.assertEqual(len(calls),1)
    def test_02_new_partial_charged_and_prefix_stops(self):
        cap,literal,calls=self.capture(http.client.IncompleteRead(b'xyz',97))
        with self.assertRaises(http.client.IncompleteRead):cap.request(literal,'image_sample_06',65536,redirect_from=-1)
        _,bill=cons.new_prefix(self.path);self.assertEqual(bill['retained_body_bytes'],3);self.assertEqual(bill['interrupted_read_upper_bound'],65533)
        url=cap.selection['samples'][7]['publisher_file']['content_details']['download_url']
        with self.assertRaises(ValueError):cap.request(url,'image_sample_07',65536)
        self.assertEqual(len(calls),1);self.assertEqual(cap.records[0]['status'],'failed')
    def test_03_parent_nonrecursive_and_cumulative_quotas(self):
        pp={'schema':'helmet24_acquisition_v1','source_commit':prod.PARENT_COMMIT,'new_caps':{'http':48,'body':67108864,'wall':1500,'storage':134217728}}
        audit={'plan_sha256':prod.PARENT_PLAN_SHA,'input_qualified':False,'bill':{'request_claims':14,'receipts':14,'retained_body_bytes':4644800,'interrupted_read_upper_bound':65536,'missing_receipt':False}}
        term={'sha256':prod.PARENT_AUDIT_SHA,'audit_sha256':prod.PARENT_AUDIT_SHA,'input_qualified':False};outer={'returncode':2,'elapsed_seconds':514.709546917}
        prod.parent_facts(pp,audit,term,outer);cons.parent_facts(pp,audit,term,outer)
        self.assertEqual(4644800+65536+1408825+prod.CAPS['body'],67108864)
        pp['schema']='helmet24_recovery_v1'
        for check in (prod.parent_facts,cons.parent_facts):
            with self.assertRaises(ValueError):check(pp,audit,term,outer)
        pp['schema']='helmet24_acquisition_v1';outer['elapsed_seconds']=500
        for check in (prod.parent_facts,cons.parent_facts):
            with self.assertRaises(ValueError):check(pp,audit,term,outer)
    def test_04_independent_display_and_opaque_no_metadata(self):
        image=Image.new('RGB',(8,4));image.putdata([(x*20,y*40,11)for y in range(4)for x in range(8)]);exif=Image.Exif();exif[274]=6;exif[315]='private';path=self.path/'tiny.jpg';image.save(path,exif=exif)
        sample={'native_annotation':{'bbox':[0,0,3,4]}};a,ga=prod.original.geometry(path,sample);b,gb=cons.native.crop_original(path,sample)
        self.assertEqual(ga,gb);self.assertEqual(ga['display_size'],[4,8]);self.assertEqual(a.tobytes(),b.tobytes());q=self.path/'opaque.png';b.save(q);self.assertFalse(Image.open(q).getexif())
        with self.assertRaises(ValueError):cons.native.crop_original(path,{'native_annotation':{'bbox':[3,7,2,2]}})


if __name__=='__main__':unittest.main(verbosity=2)
