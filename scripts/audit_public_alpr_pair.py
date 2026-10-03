"""Read-only evidence audit; writes a new redacted report, never calls a model."""
import hashlib
import json
from pathlib import Path
import statistics
import unicodedata

ROOT = Path('/Users/howen/Projects/PPO_MEC/artifacts/analysis')
SOURCES = [ROOT / 'alpr_public_sample_20261004_v1', ROOT / 'alpr_public_base_comparison_20261004_v1']
DEST = ROOT / 'alpr_public_pair_review_20261004_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_answer(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).upper() if c.isalnum())


def edits(reference, prediction):
    table = [[0] * (len(prediction) + 1) for _ in range(len(reference) + 1)]
    for i in range(len(reference) + 1):
        table[i][0] = i
    for j in range(len(prediction) + 1):
        table[0][j] = j
    for i, x in enumerate(reference, 1):
        for j, y in enumerate(prediction, 1):
            table[i][j] = min(table[i-1][j] + 1, table[i][j-1] + 1, table[i-1][j-1] + (x != y))
    return table[-1][-1]


def audit():
    checked, identity, answers, labels, reports, times = [], [], [], [], [], []
    for root in SOURCES:
        manifest = json.loads((root / 'artifact_integrity.json').read_text())
        for row in manifest:
            p = root / row['path']
            assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], str(p)
        checked.append({'root': str(root), 'files': len(manifest), 'manifest_sha256': sha(root / 'artifact_integrity.json')})
        assert json.loads((root / 'completion_receipt.json').read_text())['status'] == 'COMPLETED'
        identity.append(json.loads((root / 'execution_identity.json').read_text()))
        answer = [json.loads(line) for line in (root / 'predictions.jsonl').read_text().splitlines()]
        reference = json.loads((root / 'private_labels_and_selection.json').read_text())
        report = json.loads((root / 'task_results_redacted.json').read_text())
        assert len(answer) == len(reference) == len(report['rows']) == 12
        for a, r, stored in zip(answer, reference, report['rows']):
            assert a['sample_id'] == r['sample_id'] == stored['sample_id']
            assert a['image_sha256'] == r['sha256']
            x, y = canonical_answer(r['reference']), canonical_answer(a['prediction'])
            assert stored['exact_match'] == (x == y)
            assert stored['edit_distance'] == edits(x, y)
            assert stored['reference_characters'] == len(x)
        answers.append(answer)
        labels.append(reference)
        reports.append(report)
        times.append(statistics.median(a['seconds'] for a in answer))
    assert labels[0] == labels[1]
    assert identity[0]['inputs_sha256'] == identity[1]['inputs_sha256']
    assert identity[0]['versions'] == identity[1]['versions']
    assert identity[0]['model_file_inventory'] == identity[1]['model_file_inventory']
    plans = [json.loads((p / 'design.json').read_text()) for p in SOURCES]
    for key in ['prompt', 'max_new_tokens', 'do_sample', 'seed', 'normalization', 'input', 'generate_limit']:
        assert plans[0][key] == plans[1][key]
    pairs = []
    for a, b, r, ar, br in zip(answers[0], answers[1], labels[0], reports[0]['rows'], reports[1]['rows']):
        pairs.append({'sample_id': a['sample_id'], 'split': r['split'],
                      'same_raw_tokens': a['token_ids'] == b['token_ids'],
                      'same_normalized_answer': canonical_answer(a['prediction']) == canonical_answer(b['prediction']),
                      'adapter_exact': ar['exact_match'], 'base_exact': br['exact_match'],
                      'adapter_edits': ar['edit_distance'], 'base_edits': br['edit_distance'],
                      'adapter_normalized_length': len(canonical_answer(a['prediction'])),
                      'base_normalized_length': len(canonical_answer(b['prediction'])),
                      'reference_length': len(canonical_answer(r['reference'])),
                      'truncated_either': a['reached_token_limit'] or b['reached_token_limit']})
    DEST.mkdir(exist_ok=False)
    result = {'reviewed_at': '2026-10-04', 'literature_cutoff': '2026-10-04; novelty not reviewed',
              'policy_version': 'tmc_review_policy_v3_20260621', 'target_venue': 'IEEE TMC',
              'artifact_run_id': DEST.name, 'scientific_commits': [i['commit'] for i in identity],
              'evidence_level': 'SCOPED_ARTIFACT_AUDIT_ONLY_NOT_PAPER_READY',
              'inventory': checked, 'score_mismatch': 0, 'input_and_generation_settings_match': True,
              'adapter_summary': reports[0]['summary'], 'base_summary': reports[1]['summary'],
              'pairs': pairs, 'observed_median_generation_seconds': dict(adapter=times[0], base=times[1]),
              'timing_caveat': 'one sequential run per arm; not randomized or a latency benchmark',
              'model_card': 'ALPR README explicitly says unknown dataset; task-specific training format unavailable',
              'decision': 'DO_NOT_PROMOTE_THIS_ADAPTER_FOR_CACHE_BENEFIT_STUDY',
              'limits': ['12 public samples only', 'ground truth not visually reannotated',
                         'underlying image rights and training overlap unverified',
                         'no significance, generalized inferiority or system-mechanism conclusion',
                         'weights were previously compatibility-tested; current run lacks an independent active-tensor fingerprint']}
    with (DEST / 'review_redacted.json').open('x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
    with (DEST / 'integrity.json').open('x') as f:
        json.dump({'review_sha256': sha(DEST / 'review_redacted.json'), 'sources': checked}, f, indent=2)
    print(json.dumps({'status': 'AUDIT_COMPLETE', 'score_mismatch': 0, 'review': str(DEST / 'review_redacted.json'),
                      'same_normalized_answers': sum(p['same_normalized_answer'] for p in pairs),
                      'median_seconds': result['observed_median_generation_seconds']}))


if __name__ == '__main__':
    audit()
