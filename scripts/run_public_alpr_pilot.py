"""Bounded local ALPR sample collection and offline correctness pilot. No training."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROJECT = Path('/Users/howen/Projects/PPO_MEC')
DATA = PROJECT / 'data/raw/ai_task_acceptance/unique_data_license_plates_5c1678c_20261004'
OUT = PROJECT / 'artifacts/analysis/alpr_public_sample_20261004_v1'
ORIGINAL_OUT = OUT
BASE_ONLY = False
REV = '5c1678c7350bdc7f9d674156bb1926567c6f9a30'
REPO = 'UniqueData/license_plates'
PYTHON = PROJECT / 'artifacts/environments/adapter_state_acceptance_py39_v1/bin/python'
BASE = PROJECT / 'data/raw/ai_workflow_pilot/model_a7da5b9_20260929'
ADAPTER = PROJECT / 'data/raw/ai_workflow_pilot/adapter_downloads_20260930/alpr_edccd0e'
PROTECTED = ['scripts/train_sa_ghmappo_real_sample.py', 'src/agents/sa_ghmappo_agent.py',
             'src/agents/sa_ghmappo_core.py', 'src/encoders/fusion_encoder.py',
             'src/evaluators/real_eval_support.py', 'tests/test_algo_pool_contract.py',
             'tests/test_checkpoint_compat.py']
PROMPT = 'Read the vehicle license plate in this image. Output only the plate text, without explanation.'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def snapshot():
    return {name: sha(PROJECT / name) for name in PROTECTED}


def verify_original():
    inventory = json.loads((ORIGINAL_OUT / 'artifact_integrity.json').read_text())
    for row in inventory:
        path = ORIGINAL_OUT / row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], row['path']
    assert json.loads((ORIGINAL_OUT / 'completion_receipt.json').read_text())['status'] == 'COMPLETED'
    return sha(ORIGINAL_OUT / 'artifact_integrity.json')


def prepare_base_comparison():
    assert BASE_ONLY
    source_hash = verify_original()
    OUT.mkdir(parents=True, exist_ok=False)
    save(OUT / 'protection_start.json', snapshot())
    for name in ['model_inputs.json', 'private_labels_and_selection.json', 'data_acceptance.json']:
        with (OUT / name).open('xb') as f:
            f.write((ORIGINAL_OUT / name).read_bytes())
    plan = json.loads((ORIGINAL_OUT / 'design.json').read_text())
    plan.update(model_mode='base_only', comparison_type='exploratory_paired_after_adapter_results',
                original_inventory_sha256=source_hash,
                original_root=str(ORIGINAL_OUT), new_independent_test=False,
                decision='all original 12 samples; no prompt or preprocessing changes; no selection or tuning')
    save(OUT / 'design.json', plan)
    print('BASE_COMPARISON_PREPARED: same 12 inputs, no adapter loading')


def paired_summary(adapter_rows, base_rows):
    a = {r['sample_id']: r for r in adapter_rows}
    b = {r['sample_id']: r for r in base_rows}
    assert len(a) == len(adapter_rows) and len(b) == len(base_rows) and set(a) == set(b)
    result = {}
    for split in ['development', 'locked_check']:
        keys = [k for k in a if a[k]['split'] == split]
        assert all(a[k]['split'] == b[k]['split'] and a[k]['reference_characters'] == b[k]['reference_characters'] for k in keys)
        result[split] = {
            'n': len(keys),
            'both_correct': sum(a[k]['exact_match'] and b[k]['exact_match'] for k in keys),
            'adapter_only_correct': sum(a[k]['exact_match'] and not b[k]['exact_match'] for k in keys),
            'base_only_correct': sum(not a[k]['exact_match'] and b[k]['exact_match'] for k in keys),
            'both_wrong': sum(not a[k]['exact_match'] and not b[k]['exact_match'] for k in keys)}
    return result


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).upper() if c.isalnum())


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j-1] + (x != y)))
        row = new
    return row[-1]


def get(url, limit):
    with urllib.request.urlopen(url, timeout=45) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError('download limit exceeded')
    return data


def prepare():
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=False)
    save(OUT / 'protection_start.json', snapshot())
    # Freeze selection and scoring BEFORE reading labels or model outputs.
    save(OUT / 'design.json', {
        'revision': REV, 'license_declared': 'CC-BY-NC-ND-4.0',
        'purpose': 'non-commercial local task correctness pilot; no redistribution',
        'max_download_bytes': 20000000, 'expected_images': 100,
        'selection': 'country sorted; SHA256(1401:path) sorted; first 3 distinct duplicate components per country',
        'split': 'first per country development, next two locked_check; no tuning in this run',
        'grouping': 'connected components of equal country+normalized plate, equal SHA256 or 64bit dHash distance <=4',
        'independence': 'proxy only; vehicle/video IDs and adapter training overlap unavailable',
        'prompt': PROMPT, 'max_new_tokens': 32, 'do_sample': False, 'seed': 1401,
        'generate_limit': 12, 'wall_limit_seconds': 1800,
        'normalization': 'NFKC uppercase alphanumeric only; no O/0 mapping; entire decoded answer scored',
        'metrics': ['exact match', 'micro character error rate'],
        'input': 'full original image; processor in-memory preprocessing only; no crop/augmentation files',
        'formal': False, 'holdout': False,
    })
    tree = json.loads(get(f'https://huggingface.co/api/datasets/{REPO}/tree/{REV}?recursive=true&limit=1000', 2000000))
    files = [r for r in tree if r['type'] == 'file' and (r['path'].endswith('.jpg') or r['path'] in ['README.md', 'Car License Plate Detection Dataset.tsv'])]
    assert len(files) == 102 and sum(r['size'] for r in files) <= 20000000
    save(OUT / 'source_tree.json', files)
    DATA.mkdir(parents=True, exist_ok=False)
    assert subprocess.run(['git', '-C', str(PROJECT), 'check-ignore', '-q', str(DATA / 'probe')]).returncode == 0
    def download(row):
        rel = Path(row['path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        raw = get(f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/' + urllib.parse.quote(row['path']), row['size'])
        assert len(raw) == row['size'], row['path']
        if 'lfs' in row:
            assert digest(raw) == row['lfs']['oid'], row['path']
        else:
            assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == row['oid']
        path = DATA / rel
        path.parent.mkdir(exist_ok=True)
        with path.open('xb') as f:
            f.write(raw)
        return {'path': row['path'], 'bytes': len(raw), 'sha256': digest(raw)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        inventory = list(pool.map(download, files))
    save(OUT / 'download_inventory.json', inventory)
    assert 'license: cc-by-nc-nd-4.0' in (DATA / 'README.md').read_text()
    with (DATA / 'Car License Plate Detection Dataset.tsv').open(newline='') as f:
        labels = list(csv.DictReader(f, delimiter='\t'))
    images = sorted(DATA.rglob('*.jpg'))
    assert len(images) == len(labels) == 100
    indexed = {str(p.relative_to(DATA)): p for p in images}
    records = []
    seen = set()
    for label in labels:
        rel = label['country'] + '/' + label['filename']
        assert rel in indexed and rel not in seen and normalize(label['plate_text'])
        seen.add(rel)
        with Image.open(indexed[rel]) as im:
            im.load()
            pixels = list(im.convert('L').resize((9, 8)).getdata())
            bits = ''.join('1' if pixels[y*9+x] > pixels[y*9+x+1] else '0' for y in range(8) for x in range(8))
            size = list(im.size)
        records.append({'path': rel, 'country': label['country'], 'reference': label['plate_text'],
                        'sha256': sha(indexed[rel]), 'dhash': int(bits, 2), 'dimensions': size})
    parent = list(range(100))
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    edges = []
    for i, a in enumerate(records):
        for j in range(i):
            b = records[j]
            same_label = a['country'] == b['country'] and normalize(a['reference']) == normalize(b['reference'])
            if same_label or a['sha256'] == b['sha256'] or bin(a['dhash'] ^ b['dhash']).count('1') <= 4:
                parent[root(i)] = root(j)
                edges.append([i, j])
    selected, used = [], set()
    for country in sorted({r['country'] for r in records}):
        indices = sorted([i for i, r in enumerate(records) if r['country'] == country],
                         key=lambda i: digest(('1401:' + records[i]['path']).encode()))
        choices = []
        for i in indices:
            if root(i) not in used:
                used.add(root(i))
                choices.append(i)
            if len(choices) == 3:
                break
        assert len(choices) == 3
        for k, i in enumerate(choices):
            selected.append(dict(records[i], sample_id=f'sample_{len(selected):02d}',
                                 split='development' if k == 0 else 'locked_check', proxy_component=root(i)))
    assert len(selected) == 12
    save(OUT / 'private_labels_and_selection.json', selected)
    inputs = [{k: v for k, v in r.items() if k != 'reference'} for r in selected]
    save(OUT / 'model_inputs.json', inputs)
    save(OUT / 'data_acceptance.json', {'images': 100, 'bytes': sum(r['bytes'] for r in inventory),
         'country_counts': {c: sum(r['country'] == c for r in records) for c in sorted({r['country'] for r in records})},
         'duplicate_edges': edges, 'components': len({root(i) for i in range(100)}),
         'selected': 12, 'development': 4, 'locked_check': 8, 'physical_vehicle_independence': 'UNVERIFIED',
         'training_overlap': 'UNVERIFIED', 'third_party_underlying_image_rights': 'not independently established',
         'license_basis': 'publisher declaration only; local non-commercial analysis, no redistribution'})
    print('DATA_READY: 100 images, 4 development + 8 locked check', flush=True)


def infer():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from peft import PeftModel
    torch.set_num_threads(1)
    torch.manual_seed(1401)
    assert sha(BASE / 'model.safetensors') == 'd05b567eeaf534e83d375551f068ed57b5f52d37c657197f644af5ef9db091a2'
    assert sha(ADAPTER / 'adapter_model.safetensors') == '95103f5d6547c194770b03ee0ac24fe6fc125eb8890e992977be8dcbfea10f83'
    assert sha(ADAPTER / 'adapter_config.json') == 'd4e1d0e6c2c77424d4ed76956cb5642a2fb8e64f233b5a4874e5341e9e085216'
    plan = json.loads((OUT / 'design.json').read_text())
    assert plan['prompt'] == PROMPT and plan['generate_limit'] == 12 and plan['max_new_tokens'] == 32
    rows = json.loads((OUT / 'model_inputs.json').read_text())
    assert len(rows) == 12
    processor = AutoProcessor.from_pretrained(BASE, local_files_only=True, trust_remote_code=False)
    base = AutoModelForVision2Seq.from_pretrained(BASE, torch_dtype=torch.float32, local_files_only=True, trust_remote_code=False)
    assert plan.get('model_mode', 'adapter') == ('base_only' if BASE_ONLY else 'adapter')
    if BASE_ONLY:
        model = base
    else:
        model = PeftModel.from_pretrained(base, ADAPTER, adapter_name='alpr', is_trainable=False, local_files_only=True)
        model.set_adapter('alpr')
    model.eval()
    with (OUT / 'predictions.jsonl').open('x') as f:
        for row in rows:
            path = DATA / row['path']
            assert sha(path) == row['sha256']
            with Image.open(path) as im:
                image = im.convert('RGB')
            message = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT}]}]
            rendered = processor.apply_chat_template(message, add_generation_prompt=True)
            inputs = processor(text=rendered, images=[image], return_tensors='pt')
            started = time.monotonic()
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=32, do_sample=False, num_beams=1, use_cache=True)
            tokens = output[:, inputs['input_ids'].shape[-1]:]
            result = {'sample_id': row['sample_id'], 'split': row['split'], 'image_sha256': row['sha256'],
                      'token_ids': tokens[0].tolist(), 'prediction': processor.batch_decode(tokens, skip_special_tokens=True)[0],
                      'seconds': time.monotonic()-started, 'reached_token_limit': tokens.shape[-1] == 32}
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
            print(row['sample_id'] + ' completed', flush=True)


def score():
    labels = json.loads((OUT / 'private_labels_and_selection.json').read_text())
    predictions = [json.loads(line) for line in (OUT / 'predictions.jsonl').read_text().splitlines()]
    assert len(predictions) == len(labels) == 12
    matched = {p['sample_id']: p for p in predictions}
    assert len(matched) == 12
    rows = []
    for ref in labels:
        pred = matched[ref['sample_id']]
        assert pred['image_sha256'] == ref['sha256'] and pred['split'] == ref['split']
        truth, answer = normalize(ref['reference']), normalize(pred['prediction'])
        rows.append({'sample_id': ref['sample_id'], 'split': ref['split'], 'country': ref['country'],
                     'exact_match': answer == truth, 'edit_distance': distance(truth, answer),
                     'reference_characters': len(truth), 'seconds': pred['seconds'], 'token_limit': pred['reached_token_limit']})
    summary = {}
    for split in ['development', 'locked_check']:
        group = [r for r in rows if r['split'] == split]
        summary[split] = {'n': len(group), 'exact_matches': sum(r['exact_match'] for r in group),
                          'micro_cer': sum(r['edit_distance'] for r in group)/sum(r['reference_characters'] for r in group)}
    save(OUT / 'task_results_redacted.json', {'summary': summary, 'rows': rows,
         'claim_scope': 'small public sample task check; not formal, holdout, algorithm or mechanism evidence'})
    if BASE_ONLY:
        plan = json.loads((OUT / 'design.json').read_text())
        assert verify_original() == plan['original_inventory_sha256']
        original = json.loads((ORIGINAL_OUT / 'task_results_redacted.json').read_text())
        save(OUT / 'paired_comparison_redacted.json', {
            'adapter': original['summary'], 'base': summary,
            'paired_counts': paired_summary(original['rows'], rows),
            'exploratory': True, 'new_independent_test': False,
            'mechanism_or_algorithm_advantage': 'NOT_ESTABLISHED'})


def supervise():
    import importlib.metadata
    save(OUT / 'execution_identity.json', {'argv': sys.argv, 'python': sys.executable,
         'model_mode': 'base_only' if BASE_ONLY else 'adapter',
         'prefix': sys.prefix, 'base_prefix': sys.base_prefix,
         'versions': {name: importlib.metadata.version(name) for name in
                      ['torch', 'transformers', 'peft', 'accelerate', 'safetensors', 'Pillow']},
         'model_file_inventory': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                                  for directory in [BASE, ADAPTER] for p in sorted(directory.iterdir()) if p.is_file()],
         'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
         'script_sha256': sha(__file__), 'plan_sha256': sha(OUT / 'design.json'),
         'inputs_sha256': sha(OUT / 'model_inputs.json'), 'retry': 0})
    start = time.monotonic()
    rc, error = None, None
    try:
        with (OUT / 'child.stdout.log').open('x') as stdout, (OUT / 'child.stderr.log').open('x') as stderr:
            rc = subprocess.run([str(PYTHON), '-B', str(Path(__file__).resolve()), 'infer'] + (['--base-comparison'] if BASE_ONLY else []),
                                stdout=stdout, stderr=stderr, timeout=1800).returncode
        if rc == 0:
            score()
    except Exception as exc:
        error = repr(exc)
    final = snapshot()
    save(OUT / 'protection_end.json', final)
    protection = final == json.loads((OUT / 'protection_start.json').read_text())
    save(OUT / 'completion_receipt.json', {'status': 'COMPLETED' if rc == 0 and error is None and protection else 'FAILED',
         'child_returncode': rc, 'error': error, 'elapsed_seconds': time.monotonic()-start,
         'protection_unchanged': protection, 'formal': False, 'holdout_opened': False, 'automatic_retry': 0})
    save(OUT / 'artifact_integrity.json', [{ 'path': str(p.relative_to(OUT)), 'bytes': p.stat().st_size,
         'sha256': sha(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'supervisor.log'])


def launch():
    assert (OUT / 'data_acceptance.json').exists()
    assert not (OUT / 'execution_identity.json').exists()
    # Exclusive log creation prevents an accidental second launch.
    with (OUT / 'supervisor.log').open('x') as log:
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        env['PYTHONUNBUFFERED'] = '1'
        child = subprocess.Popen([str(PYTHON), '-B', str(Path(__file__).resolve()), 'supervise'] + (['--base-comparison'] if BASE_ONLY else []),
                                 stdout=log, stderr=subprocess.STDOUT, env=env,
                                 start_new_session=True, stdin=subprocess.DEVNULL)
    save(OUT / 'launch_receipt.json', {'pid': child.pid, 'started_unix': time.time(),
         'automatic_retry': 0, 'completion_receipt': str(OUT / 'completion_receipt.json')})
    print(json.dumps({'status': 'LAUNCHED_NOT_COMPLETED', 'pid': child.pid}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'prepare_base', 'infer', 'supervise', 'launch'])
    parser.add_argument('--base-comparison', action='store_true')
    args = parser.parse_args()
    BASE_ONLY = args.base_comparison
    if BASE_ONLY:
        assert args.action != 'prepare'
        OUT = PROJECT / 'artifacts/analysis/alpr_public_base_comparison_20261004_v1'
    try:
        {'prepare': prepare, 'prepare_base': prepare_base_comparison, 'infer': infer, 'supervise': supervise, 'launch': launch}[args.action]()
    except Exception as exc:
        # Preserve a startup failure as well as failures after the scientific child.
        if args.action == 'supervise' and not (OUT / 'completion_receipt.json').exists():
            save(OUT / 'completion_receipt.json', {'status': 'FAILED_BEFORE_OR_DURING_SUPERVISION',
                 'error': repr(exc), 'automatic_retry': 0, 'formal': False, 'holdout_opened': False})
        raise
