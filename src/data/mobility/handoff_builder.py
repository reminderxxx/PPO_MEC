"""切换事件构造器。"""

from __future__ import annotations

from src.envs.specs import HandoffEvent


class HandoffBuilder:
    """根据关联变化生成 enter / leave / handoff 事件。"""

    def build_events(
        self,
        previous_associations: dict[str, str | None],
        current_associations: dict[str, str | None],
        time_index: int,
    ) -> list[HandoffEvent]:
        events: list[HandoffEvent] = []
        vehicle_ids = set(previous_associations) | set(current_associations)
        for vehicle_id in sorted(vehicle_ids):
            previous_rsu_id = previous_associations.get(vehicle_id)
            current_rsu_id = current_associations.get(vehicle_id)
            if previous_rsu_id == current_rsu_id:
                continue
            if previous_rsu_id is None and current_rsu_id is not None:
                event_type = "enter"
            elif previous_rsu_id is not None and current_rsu_id is None:
                event_type = "leave"
            else:
                event_type = "handoff"
            events.append(
                HandoffEvent(
                    vehicle_id=vehicle_id,
                    time_index=time_index,
                    previous_rsu_id=previous_rsu_id,
                    current_rsu_id=current_rsu_id,
                    event_type=event_type,
                )
            )
        return events
