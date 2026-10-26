"""Join immutable independent seed shards; never select or retrain a cell."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.probe_stream_main_opportunity import file_hash


def train_unit_hash(git):
    source=subprocess.check_output(['git','show',f'{git}:scripts/run_stream_main_ablation.py'],cwd=ROOT,text=True)
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='train_unit')
    return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shards',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plans=[json.loads((s/'protocol.json').read_text()) for s in a.shards]
    reference=plans[0];cells=[];inventory=[];identities=set();unfinished=[]
    keys=['variants','training_cases','frames_per_training_case','actual_training_frames_per_cell',
        'update_group_size','update_calls_per_cell','epochs','evaluation_cases','evaluation_mode',
        'training_return','exposure','optimizer','baseline_source_sha256','environment_source_sha256',
        'actor_source_sha256','main_source_sha256','parameter_count','parent_protocol_sha256','diagnosis_sha256']
    for shard,plan in zip(a.shards,plans):
        assert not plan['preflight']
        assert all(plan[k]==reference[k] for k in keys)
        assert all(reference['parents'][seed]==parent for seed,parent in plan['parents'].items())
        assert train_unit_hash(plan['source_commit'])==train_unit_hash(reference['source_commit'])
        paths={f.parent for f in shard.glob('*/manifest.json')}
        for home in sorted(paths):
            meta=json.loads((home/'manifest.json').read_text())
            if meta['status']!='complete':raise ValueError('incomplete terminal manifest')
            identity=(meta['variant'],meta['seed'])
            if identity in identities:raise ValueError('duplicate completed scientific cell')
            identities.add(identity)
            cells.append(dict(path=str(home.resolve()),source_protocol=str((shard/'protocol.json').resolve()),
                source_protocol_sha256=file_hash(shard/'protocol.json'),**meta))
        for home in shard.iterdir():
            if home.is_dir() and not (home/'manifest.json').exists():
                unfinished.append(dict(path=str(home.resolve()),reason='execution-boundary interruption; no completed training cell reused',
                    files=[str(f.name) for f in home.iterdir() if f.is_file()]))
        inventory.append(dict(path=str(shard.resolve()),source_commit=plan['source_commit'],
            protocol_sha256=file_hash(shard/'protocol.json'),train_unit_ast_sha256=train_unit_hash(plan['source_commit'])))
    expected={(v,s) for v in reference['variants'] for s in [7,17,27]}
    assert identities==expected
    plan=dict(reference,seeds=[7,17,27],execution_shards=inventory,
        preserved_incomplete_execution_directories=unfinished,
        collection='All 18 predeclared cells, no score selection; identical scientific training-unit AST and all core settings')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    (a.output/'cells.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in cells))
    (a.output/'complete.json').write_text(json.dumps(dict(status='complete',cells=len(cells),
        cells_sha256=file_hash(a.output/'cells.jsonl')),indent=2)+'\n')
    print(json.dumps(dict(cells=len(cells),shards=len(a.shards),preserved_partial_dirs=len(unfinished))))


if __name__=='__main__':main()
