"""One authorized metadata request followed by one independent terminal consumer."""
import argparse
import json
from pathlib import Path
import signal

if __package__:
    from . import acquire_public_task_metadata as producer
    from . import audit_public_task_metadata as consumer
else:
    import acquire_public_task_metadata as producer
    import audit_public_task_metadata as consumer


def run(plan_path, plan_sha, grant_path):
    entered = producer.clock()
    plan = json.loads(plan_path.read_text())
    if 'single_route_extension' not in plan or producer.digest(plan_path) != plan_sha:
        raise ValueError('fixed extension plan required')
    root = Path(plan['raw_root']); output = Path(plan['completion_output']); audit_path = Path(plan['audit_output'])
    if output.exists() or audit_path.exists() or root.exists():
        raise ValueError('single launch/create-only outputs; no retry')
    errors = []; result = None; claim = None
    try:
        producer.initialize(plan_path, plan_sha, grant_path, root)
        intent_path = root/'single_intent.json'
        producer.save(intent_path, {'url': producer.PUBLIC_FILES, 'source': 'edgevision_v1', 'kind': 'metadata'})
        producer.acquire_one(root, intent_path, producer.digest(intent_path))
    except BaseException as error:
        errors.append({'stage': 'producer', 'type': type(error).__name__, 'error': str(error)})
    if (root/'claim.json').exists():
        try:
            claim = json.loads((root/'claim.json').read_text())
            if not entered <= claim['extension_start_clock'] <= producer.clock():
                raise ValueError('claim start outside this new invocation')
        except BaseException as error:
            claim = None
            errors.append({'stage': 'claim', 'type': type(error).__name__, 'error': str(error)})
    if claim is not None:
        remaining = 120 - (producer.clock() - claim['extension_start_clock'])
        def deadline(signum, frame):
            raise TimeoutError('extension consumer total deadline')
        previous = signal.signal(signal.SIGALRM, deadline)
        try:
            if remaining <= 0:
                raise TimeoutError('no consumer time remaining')
            signal.setitimer(signal.ITIMER_REAL, remaining)
            ledger = consumer.audit(root, plan_sha, 'ONE_EXPLICIT_METADATA_EXTENSION_ONLY_NO_D_OR_Q')
            producer.save(audit_path, ledger)
            producer.save(root/'terminal.json', {'audit': str(audit_path), 'sha256': producer.digest(audit_path)})
            result = ledger  # Both immutable outputs have committed; a partial save is not audit success.
        except BaseException as error:
            errors.append({'stage': 'consumer', 'type': type(error).__name__, 'error': str(error)})
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
    elapsed = producer.clock()-claim['extension_start_clock'] if claim is not None else None
    completion = {'plan_sha256': plan_sha, 'errors': errors, 'audit': str(audit_path) if result is not None else None,
                  'elapsed_including_terminal': elapsed, 'new_deadline_pass': elapsed is not None and elapsed <= 120,
                  'producer_success': not any(e['stage']=='producer' for e in errors),
                  'audit_success': result is not None, 'old_metadata_deadline_pass': False,
                  'images': 0, 'model_calls': 0, 'paper_goal_complete': False}
    attempts = list(root.glob('request_*')) if root.exists() else []
    completion['observed_attempt_directories'] = len(attempts)
    if result is None:
        retained = sum((p/'body.bin').stat().st_size for p in attempts if (p/'body.bin').exists())
        completion.update(new_request_upper_on_audit_failure=len(attempts), retained_body_bytes=retained,
                          body_upper_on_audit_failure=1048576 if attempts else 0,
                          prefix_note='Conservative failed-consumer prefix, not a verified receipt ledger')
    completion['runner_elapsed_before_completion_write'] = producer.clock()-entered
    producer.save(output, completion)
    return completion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True); p.add_argument('--sha256', required=True)
    p.add_argument('--grant', type=Path, required=True); a = p.parse_args()
    print(json.dumps(run(a.plan, a.sha256, a.grant), ensure_ascii=False))


if __name__ == '__main__':
    main()
