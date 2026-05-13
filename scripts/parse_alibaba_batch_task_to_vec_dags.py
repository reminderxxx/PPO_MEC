import json
import os
import random
import re
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

import pandas as pd


INPUT_CSV = r"D:\PPO_MEC\data\raw\workflow\alibaba2018\batch_task.csv"
OUTPUT_JSONL = r"D:\PPO_MEC\data\processed\sampled_vec_dags\sampled_vec_dags.jsonl"
OUTPUT_STATS = r"D:\PPO_MEC\data\processed\sampled_vec_dags\sampled_vec_dags_stats.json"

COLS = [
    "task_name",
    "instance_num",
    "job_name",
    "task_type",
    "status",
    "start_time",
    "end_time",
    "plan_cpu",
    "plan_mem",
]

MIN_TASKS = 5
MAX_TASKS = 15
MAX_JOBS = 2000
CHUNK_SIZE = 500_000
RANDOM_SEED = 42


def parse_task_name(task_name: str) -> Tuple[Optional[int], List[int]]:
    """
    解析 Alibaba v2018 的 task_name:
    - M1      -> task_id=1, parents=[]
    - R2_1    -> task_id=2, parents=[1]
    - J4_2_3  -> task_id=4, parents=[2,3]
    """
    if not isinstance(task_name, str) or not task_name:
        return None, []

    parts = task_name.split("_")
    head = parts[0]

    m = re.search(r"(\d+)$", head)
    if not m:
        return None, []

    task_id = int(m.group(1))
    parents = []

    for p in parts[1:]:
        if p.isdigit():
            parents.append(int(p))

    return task_id, parents


def is_dag(num_nodes: int, edges: List[Tuple[int, int]]) -> bool:
    """
    Kahn 拓扑排序判断是否无环
    """
    indegree = defaultdict(int)
    graph = defaultdict(list)

    nodes = set()
    for u, v in edges:
        graph[u].append(v)
        indegree[v] += 1
        nodes.add(u)
        nodes.add(v)

    # 把孤立点也算进去
    for i in range(1, num_nodes + 1):
        nodes.add(i)

    q = deque([n for n in nodes if indegree[n] == 0])
    visited = 0

    while q:
        cur = q.popleft()
        visited += 1
        for nxt in graph[cur]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)

    return visited == len(nodes)


def ensure_dirs():
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_STATS), exist_ok=True)


def first_pass_count_jobs() -> Dict[str, int]:
    """
    第一遍扫描：统计每个 job 的任务数
    """
    job_counts = defaultdict(int)

    for chunk in pd.read_csv(
        INPUT_CSV,
        header=None,
        names=COLS,
        usecols=["job_name"],
        chunksize=CHUNK_SIZE,
    ):
        vc = chunk["job_name"].value_counts()
        for job_name, cnt in vc.items():
            job_counts[job_name] += int(cnt)

    return dict(job_counts)


def second_pass_collect_jobs(target_jobs: set) -> Dict[str, List[dict]]:
    """
    第二遍扫描：只收集目标 job 的详细记录
    """
    jobs_data = defaultdict(list)

    for chunk in pd.read_csv(
        INPUT_CSV,
        header=None,
        names=COLS,
        usecols=COLS,
        chunksize=CHUNK_SIZE,
    ):
        chunk = chunk[chunk["job_name"].isin(target_jobs)]
        if chunk.empty:
            continue

        # 只保留终止完成的任务
        chunk = chunk[chunk["status"] == "Terminated"]
        if chunk.empty:
            continue

        for _, row in chunk.iterrows():
            jobs_data[row["job_name"]].append({
                "task_name": str(row["task_name"]),
                "instance_num": int(row["instance_num"]),
                "job_name": str(row["job_name"]),
                "task_type": int(row["task_type"]),
                "status": str(row["status"]),
                "start_time": int(row["start_time"]),
                "end_time": int(row["end_time"]),
                "plan_cpu": float(row["plan_cpu"]),
                "plan_mem": float(row["plan_mem"]),
            })

    return jobs_data


def build_workflow(job_name: str, rows: List[dict]) -> Optional[dict]:
    """
    把一个 job 的多行记录构造成 DAG workflow
    """
    nodes_by_id = {}
    edges = []
    missing_parent = False

    for row in rows:
        task_id, parents = parse_task_name(row["task_name"])
        if task_id is None:
            return None

        duration_raw = max(1, int(row["end_time"]) - int(row["start_time"]))

        # 如果同一个 task_id 重复出现，先保留第一条，避免歧义
        if task_id not in nodes_by_id:
            nodes_by_id[task_id] = {
                "task_id": task_id,
                "task_name": row["task_name"],
                "parents": parents[:],
                "instance_num": int(row["instance_num"]),
                "plan_cpu": float(row["plan_cpu"]),
                "plan_mem": float(row["plan_mem"]),
                "start_time": int(row["start_time"]),
                "end_time": int(row["end_time"]),
                "duration_raw": duration_raw,
                # 先保留一个可直接进环境的时间字段，后续再缩放
                "duration_vec": duration_raw,
                # 下面这些字段给后续 AI caching / adapter 叙事留口
                "required_base_model": None,
                "required_adapter": None,
                "input_size_mb": None,
                "output_size_mb": None,
            }

    task_ids = set(nodes_by_id.keys())

    for task_id, node in nodes_by_id.items():
        for p in node["parents"]:
            if p not in task_ids:
                missing_parent = True
                break
            edges.append((p, task_id))
        if missing_parent:
            break

    if missing_parent:
        return None

    # 为了做 DAG 检查，这里要求 task_id 尽量连续不是必须，所以直接用真实节点数
    if not is_dag(len(nodes_by_id), edges):
        return None

    roots = sorted([tid for tid, node in nodes_by_id.items() if len(node["parents"]) == 0])

    workflow = {
        "workflow_id": job_name,
        "source": "alibaba_cluster_trace_2018_batch_task",
        "num_tasks": len(nodes_by_id),
        "num_edges": len(edges),
        "roots": roots,
        "nodes": [nodes_by_id[k] for k in sorted(nodes_by_id.keys())],
    }
    return workflow


def main():
    ensure_dirs()

    print("Step 1/4: 统计每个 job 的任务数...")
    job_counts = first_pass_count_jobs()

    eligible_jobs = [
        j for j, c in job_counts.items()
        if MIN_TASKS <= c <= MAX_TASKS
    ]
    print(f"满足任务数范围 [{MIN_TASKS}, {MAX_TASKS}] 的 job 数量: {len(eligible_jobs)}")

    random.seed(RANDOM_SEED)
    if len(eligible_jobs) > MAX_JOBS:
        eligible_jobs = random.sample(eligible_jobs, MAX_JOBS)

    target_jobs = set(eligible_jobs)
    print(f"实际采样 job 数量: {len(target_jobs)}")

    print("Step 2/4: 收集目标 job 的详细记录...")
    jobs_data = second_pass_collect_jobs(target_jobs)

    print("Step 3/4: 构建 DAG workflows...")
    workflows = []
    skipped = {
        "empty_or_filtered": 0,
        "parse_failed": 0,
        "not_dag_or_missing_parent": 0,
    }

    for job_name, rows in jobs_data.items():
        if not rows:
            skipped["empty_or_filtered"] += 1
            continue

        wf = build_workflow(job_name, rows)
        if wf is None:
            skipped["not_dag_or_missing_parent"] += 1
            continue

        workflows.append(wf)

    workflows.sort(key=lambda x: x["workflow_id"])

    print("Step 4/4: 写出 JSONL...")
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for wf in workflows:
            f.write(json.dumps(wf, ensure_ascii=False) + "\n")

    stats = {
        "input_csv": INPUT_CSV,
        "output_jsonl": OUTPUT_JSONL,
        "min_tasks": MIN_TASKS,
        "max_tasks": MAX_TASKS,
        "max_jobs": MAX_JOBS,
        "sampled_jobs": len(target_jobs),
        "final_workflows": len(workflows),
        "skipped": skipped,
    }

    with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n===== DONE =====")
    print(f"输出文件: {OUTPUT_JSONL}")
    print(f"统计文件: {OUTPUT_STATS}")
    print(f"最终 workflow 数量: {len(workflows)}")
    print(f"跳过统计: {skipped}")


if __name__ == "__main__":
    main()