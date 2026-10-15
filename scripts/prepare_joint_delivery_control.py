"""Copy a verified immutable protocol/predictor into the independent delivery run."""
import argparse,json,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.launch_eight_vehicle_joint_experiment import _hash_file


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,default=ROOT/'artifacts/joint_tail_control/protocol/manifest.json');p.add_argument('--output_root',type=Path,required=True);a=p.parse_args()
    parent=a.parent.resolve();m=json.loads(parent.read_text())
    for name,key in [('protocol.pt','protocol_sha256'),('predictor.pt','predictor_sha256')]:assert _hash_file(parent.parent/name)==m[key]
    a.output_root.mkdir(parents=True,exist_ok=False)
    for name in ('protocol.pt','predictor.pt'):shutil.copyfile(parent.parent/name,a.output_root/name)
    m.update(parent_prepared_manifest=str(parent),parent_prepared_manifest_sha256=_hash_file(parent),
        created_at=datetime.now(timezone.utc).isoformat(),command=sys.argv,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        reuse='byte-identical train/dev scenarios and frozen GRU, no new data selection or fitting',
        producer_sha256=_hash_file(Path(__file__)))
    (a.output_root/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output_root),protocol_sha256=m['protocol_sha256'],predictor_sha256=m['predictor_sha256'],git_dirty=m['git_dirty'])))


if __name__=='__main__':main()
