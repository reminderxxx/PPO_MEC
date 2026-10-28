"""Explicitly granted transport retries and hash-bound ancestor resource accounting."""
import json
import math
from pathlib import Path

try:
    from scripts.prepare_tgsim_input import BudgetLimit, digest, fetch, save_new
    from scripts.prepare_tgsim_paged_input import ROOT, PRIVATE
except ModuleNotFoundError:
    from prepare_tgsim_input import BudgetLimit, digest, fetch, save_new
    from prepare_tgsim_paged_input import ROOT, PRIVATE

RETRY_POLICY_SHA = '98b13e620983b304c4adb50a42c93594c50824b4c34eadeb53d0dc2367bd50f1'
ORIGINAL_CAPS = dict(max_download_bytes_new_task=536870912, max_wall_clock_seconds=2700,
                     max_rows_to_index=6000000, maximum_http_requests=144,
                     max_decoded_csv_bytes=1288490188)
TRANSPORT_ERRORS = {'TimeoutError', 'IncompleteRead', 'ConnectionResetError',
                    'ConnectionAbortedError', 'RemoteDisconnected', 'BrokenPipeError'}


def retryable(receipt):
    """Only a transport failure while opening/reading; no semantic/data repair."""
    kind = receipt.get('error_type')
    if kind == 'URLError':
        kind = receipt.get('transport_reason_type')
    return (receipt.get('status') == 'failed'
            and receipt.get('failure_phase') in ('open', 'read')
            and receipt.get('status_code', 200) == 200 and kind in TRANSPORT_ERRORS)


def ancestor_bill(plan, *, allow_indexed_parent=False):
    """Read only metadata, verifying every parent completion AND its frozen plan."""
    current = plan; chain = []; seen = set()
    while current.get('continuation_parent_completion'):
        path = Path(current['continuation_parent_completion']).resolve()
        if path in seen or len(chain) >= 32:
            raise ValueError('cyclic/duplicated ancestor completion chain')
        seen.add(path)
        if not path.is_relative_to(ROOT.resolve()) or digest(path) != current['continuation_parent_completion_sha256']:
            raise ValueError('ancestor completion SHA/path differs')
        completion = json.loads(path.read_text())
        parent_path = Path(completion['command_arguments']['proposal']).resolve()
        if (not any(parent_path.is_relative_to(root.resolve()) for root in (ROOT, PRIVATE))
                or digest(parent_path) != completion['proposal_sha256']
                or completion['proposal_sha256'] != current['continuation_parent_plan_sha256']):
            raise ValueError('ancestor plan SHA/path differs')
        parent = json.loads(parent_path.read_text())
        if parent['sources'] != plan['sources'] or not set(parent['disk_roots']).issubset(plan['disk_roots']):
            raise ValueError('ancestor source pool or cumulative disk roots were dropped')
        claim = json.loads((path.parent/'execution_claim.json').read_text())
        if (claim['proposal'] != parent or claim['proposal_sha256'] != completion['proposal_sha256']
                or claim['source_commit'] != completion['source_commit']
                or completion['status'] != 'failed_preserved_do_not_retry'
                or (completion['rows_indexed'] != 0 and not allow_indexed_parent)):
            raise ValueError('ancestor provenance/status/native indexing differs')
        for key in ('training_frames', 'evaluation_frames', 'hidden_openings'):
            if completion[key] != 0 or claim[key] != 0:
                raise ValueError('ancestor input-only boundary differs')
        values = [completion[k] for k in ('total_body_download_bytes_upper_bound',
                   'http_requests', 'elapsed_seconds', 'decoded_csv_bytes','rows_indexed')]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('invalid ancestor resource spend')
        chain.append(dict(path=str(path), sha256=digest(path), plan_sha256=completion['proposal_sha256'],
                          completion=completion, plan=parent))
        current = parent
    if not chain:
        raise ValueError('bounded retry requires the complete frozen ancestor chain')
    binding = [{k:item[k] for k in ('path','sha256','plan_sha256')} for item in chain]
    if binding != plan['ancestor_completions']:
        raise ValueError('ancestor inventory differs; omitted/doubled spend is forbidden')
    caps = dict(ORIGINAL_CAPS); prior = current['prior_accounted_body_bound']; elapsed = 0.
    for item in reversed(chain):
        parent = item['plan']; previous = item['completion']
        expected = dict(caps)
        if any(parent[k] != v for k,v in expected.items()):
            raise ValueError('ancestor budget was reset or enlarged')
        for spend,limit in (('total_body_download_bytes_upper_bound','max_download_bytes_new_task'),
                            ('http_requests','maximum_http_requests'),('elapsed_seconds','max_wall_clock_seconds'),
                            ('decoded_csv_bytes','max_decoded_csv_bytes'),('rows_indexed','max_rows_to_index')):
            if previous[spend] > parent[limit]:
                raise ValueError('ancestor already exceeded its resource budget')
        if parent['prior_accounted_body_bound'] != prior:
            raise ValueError('ancestor accumulated transfer bill not carried forward')
        prior += previous['total_body_download_bytes_upper_bound']
        if prior != previous['two_task_body_upper_bound']:
            raise ValueError('ancestor terminal transfer bill differs')
        caps['max_download_bytes_new_task'] -= previous['total_body_download_bytes_upper_bound']
        caps['maximum_http_requests'] -= previous['http_requests']
        caps['max_decoded_csv_bytes'] -= previous['decoded_csv_bytes']
        caps['max_rows_to_index'] -= previous['rows_indexed']
        # Carry the stricter integer executor cap at each boundary. Do not
        # recover discarded fractional seconds in a later continuation.
        caps['max_wall_clock_seconds'] = int(caps['max_wall_clock_seconds']-previous['elapsed_seconds'])
        elapsed += previous['elapsed_seconds']
    if plan['prior_accounted_body_bound'] != prior or any(v <= 0 for v in caps.values()):
        raise ValueError('no remaining budget or current accumulated bill differs')
    return dict(caps=caps, ancestors=binding, prior_accounted_body_bound=prior,
                ancestor_executor_seconds=elapsed)


def validate_retry_grant(plan, authorization):
    policy_path = Path(authorization['proposal'])
    if (not authorization.get('execution_authorized')
            or authorization.get('scope') != 'same_source_input_recovery_zero_science'
            or not authorization.get('bounded_transport_retry_authorized')
            or authorization.get('answer_verbatim') != '允许有限重试，继续完成'
            or digest(policy_path) != RETRY_POLICY_SHA
            or authorization.get('proposal_sha256') != RETRY_POLICY_SHA
            or plan.get('retry_policy_sha256') != RETRY_POLICY_SHA
            or plan.get('max_total_attempts_per_trajectory_page') != 3
            or plan.get('max_extra_attempts_global') != 8):
        raise ValueError('explicit bounded retry policy grant differs')
    policy = json.loads(policy_path.read_text())
    mapping = dict(maximum_http_requests='max_new_http_requests',
                   max_download_bytes_new_task='max_new_encoded_body_and_read_bounds',
                   max_wall_clock_seconds='max_new_executor_wall_seconds',
                   max_decoded_csv_bytes='max_new_decoded_text_bytes',
                   max_rows_to_index='max_native_index_rows',
                   max_cumulative_new_disk_bytes_including_old_failure='max_cumulative_disk_bytes')
    if any(plan[k] != policy[v] for k,v in mapping.items()):
        raise ValueError('granted bounded retry caps differ')
    bill = ancestor_bill(plan)
    if any(plan[k] != v for k,v in bill['caps'].items()):
        raise ValueError('all ancestor spend must be deducted before retry')
    return bill


class BoundedPageFetcher:
    """Each full attempt is immutable; only one complete attempt becomes a page."""
    def __init__(self, plan, analysis):
        self.plan = plan; self.analysis = analysis; self.extra_attempts = 0
        self.attempt_directory = analysis/'attempts'
        self.attempt_directory.mkdir(exist_ok=False)

    def __call__(self, url, path, budget, receipt_path, *, opener, read_timeout_seconds):
        entries = []; selected = None; final = None
        try:
            for number in range(1, self.plan['max_total_attempts_per_trajectory_page']+1):
                budget.guard()
                if budget.proposal['max_download_bytes'] <= budget.download_bytes + budget.interrupted_body_upper_bound:
                    raise BudgetLimit('no conservative byte budget for another attempt')
                if number > 1:
                    if self.extra_attempts >= self.plan['max_extra_attempts_global']:
                        raise BudgetLimit('global extra transport-attempt cap reached')
                    # Only this qualified wrapper can relax one repeated URL.
                    # PagedBudget still enforces the global request limit.
                    budget.urls[url] -= 1
                budget.request(url)
                if number > 1:self.extra_attempts += 1
                attempt_path = path.with_name(path.name+f'.attempt_{number}')
                attempt_receipt = self.attempt_directory/f'{receipt_path.stem}_attempt_{number}.json'
                try:
                    final = fetch(url,attempt_path,budget,attempt_receipt,opener=opener,
                                  accept_gzip=True,read_timeout_seconds=read_timeout_seconds)
                except BaseException:
                    if attempt_receipt.exists():
                        final = json.loads(attempt_receipt.read_text())
                        entries.append(dict(path=str(attempt_receipt),sha256=digest(attempt_receipt)))
                    if final is None or not retryable(final) or number == self.plan['max_total_attempts_per_trajectory_page']:
                        raise
                    continue
                entries.append(dict(path=str(attempt_receipt),sha256=digest(attempt_receipt)))
                with attempt_path.open('rb') as inp,path.open('xb') as out:
                    for block in iter(lambda:inp.read(1024**2),b''):
                        budget.guard();out.write(block);out.flush()
                budget.guard(); selected = number
                return dict(final,path=str(path))
        finally:
            if entries:
                attempts = [json.loads(Path(e['path']).read_text()) for e in entries]
                logical = dict(final,operation='bounded_trajectory_attempts',automatic_retry=True,
                               path=str(path) if selected is not None else final['path'],
                               status='complete' if selected is not None else 'failed',
                               attempts=entries,selected_attempt=selected,
                               network_body_bytes=sum(a['bytes'] for a in attempts),
                               interrupted_read_body_upper_bound=sum(a['interrupted_read_body_upper_bound'] for a in attempts),
                               http_request_count=len(attempts),
                               receipt_contract='tgsim_bounded_page_attempts_v1')
                save_new(receipt_path,logical)
