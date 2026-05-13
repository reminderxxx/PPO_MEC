"""最小冒烟脚本。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.mobility.replay_provider import ReplayProvider
from src.data.model_catalog.adapter_catalog import AdapterCatalog
from src.data.workflow.toy_workflow_generator import ToyWorkflowGenerator
from src.envs.core.vec_workflow_core_env import VecWorkflowCoreEnv
from src.envs.specs import ControlAction


def 构造最小控制动作(state: dict) -> ControlAction:
    """围绕当前节点生成最小可运行控制。"""
    current_node = state.get("current_workflow_node") or {}
    required_adapter = current_node.get("required_adapter")
    return ControlAction(
        cache_action={
            "operation": "cache",
            "adapter_id": required_adapter,
        }
        if required_adapter
        else {},
        offload_action={"mode": "hybrid"},
        migration_action={"mode": "migrate"},
    )


def main() -> None:
    mobility_provider = ReplayProvider()
    workflow_state = ToyWorkflowGenerator().generate()
    catalog_path = (
        ROOT_DIR
        / "src"
        / "data"
        / "model_catalog"
        / "sample_model_catalog.json"
    )
    adapter_catalog = AdapterCatalog.from_json(catalog_path)
    env = VecWorkflowCoreEnv(
        mobility_provider=mobility_provider,
        workflow_state=workflow_state,
        adapter_catalog=adapter_catalog,
    )

    state, info = env.reset()
    print("开始执行 toy smoke test")
    print(f"初始时间: {state['time_index']}")
    print(f"初始车辆数: {len(state['vehicles'])}")
    print(f"初始节点: {state['current_workflow_node']['node_id']}")
    print(f"初始 handoff 事件数: {len(info['handoff_events'])}")

    terminated = False
    truncated = False
    step_index = 0
    while not terminated and not truncated and step_index < 8:
        control = 构造最小控制动作(state)
        state, reward, terminated, truncated, info = env.step(control)
        step_index += 1
        current_node = state.get("current_workflow_node")
        current_node_id = current_node["node_id"] if current_node else "已完成"
        print(
            f"步骤 {step_index}: 节点={current_node_id}, "
            f"奖励={reward.total:.2f}, "
            f"cache_hit={info['cache_hit']}, "
            f"handoff={len(info['handoff_events'])}, "
            f"stall={info['stall_occurred']}"
        )

    print(f"terminated={terminated}, truncated={truncated}")
    print(f"完成节点数: {len(state['workflow']['completed_node_ids'])}")
    print("smoke test 完成")


if __name__ == "__main__":
    main()
