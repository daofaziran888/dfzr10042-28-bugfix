"""漏斗引擎：按事件时间核验步骤顺序与转化窗口，只产出有完整证据的转化路径。

两类历史误计在这里被同时修掉：

1. 先购买后进店（早到事件）：旧实现只判断“用户是否出现过某事件”，
   购买时间早于进店也会被错位计入后续步骤。本引擎严格要求每一步事件时间
   严格晚于上一步，且整条路径从首步起落在 ``conversion_window_hours`` 时窗内。
2. 超过 24 小时的购买（超窗事件）：旧实现没有时窗概念，超窗购买残留在
   归因复盘的转化分母里。本引擎把超窗路径放进 ``rejected``，永不进入
   已发布分母。

引擎只依赖标准库、全部为确定性纯函数：同一份原始事件无论乱序送入还是
按时间送入，匹配结果完全一致，已发布结果随时可用原事件数据复算。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

# 早到购买：购买（末步事件）发生在进店等前置步骤之前
REASON_OUT_OF_ORDER = "out_of_order_purchase"
# 超窗购买：路径首步到末步的时间差超过漏斗转化窗口
REASON_OUTSIDE_WINDOW = "outside_window"
# 重复订单：同一 order_id 被重复提交
REASON_DUPLICATE_ORDER = "duplicate_order"
# 路径不完整：前置步骤做了但始终没有末步事件
REASON_INCOMPLETE = "incomplete_path"


def parse_event_time(value: Any) -> datetime:
    """解析 ISO-8601 事件时间，统一归一化为 UTC aware datetime。

    不带时区的时间戳按 UTC 处理（示例数据均为无时区 ISO 串）；
    无法解析的输入抛出 ``ValueError``，由调用方决定丢弃还是报错。
    """
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _event_identity(event: dict[str, Any]) -> tuple[Any, ...]:
    """重复提交判定键：同名事件、同一时刻、同一订单视为同一次提交。"""
    return (
        event.get("event_name"),
        event.get("timestamp"),
        event.get("order_id"),
    )


def normalize_user_events(
    user_events: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """按事件时间排序并去重，保证结果与事件送入顺序无关。"""
    normalized: dict[str, list[dict[str, Any]]] = {}
    for uid, events in user_events.items():
        seen: set[tuple[Any, ...]] = set()
        deduped: list[dict[str, Any]] = []
        for event in events:
            key = _event_identity(event)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(dict(event))
        # 稳定排序：同时间事件保持相对顺序，匹配时再用严格大于避免并列误计
        deduped.sort(key=lambda e: parse_event_time(e.get("timestamp", "")))
        normalized[uid] = deduped
    return normalized


@dataclass
class FunnelMatchResult:
    """单次漏斗匹配的完整产物。"""

    step_event_names: list[str]
    window_hours: int
    step_counts: list[int] = field(default_factory=list)
    step_rates: list[float] = field(default_factory=list)
    total_conversion_rate: float = 0.0
    # 有完整证据（步骤有序且全程在窗内）的转化路径
    confirmed_paths: list[dict[str, Any]] = field(default_factory=list)
    # 被拒绝的路径尝试及原因，供归因复盘排查，不进入任何分母
    rejected: list[dict[str, Any]] = field(default_factory=list)
    # 被去重的重复事件提交
    duplicate_events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def conversion_order_ids(self) -> list[str]:
        orders = [
            path.get("order_id")
            for path in self.confirmed_paths
            if path.get("order_id") is not None
        ]
        return list(dict.fromkeys(orders))


def match_funnel(
    step_event_names: list[str],
    conversion_window_hours: int,
    user_events: dict[str, list[dict[str, Any]]],
) -> FunnelMatchResult:
    """按时间顺序读取用户事件并匹配漏斗步骤。

    规则：
    - 每个用户在每一步最多计一人（集合去重）；
    - 下一步事件时间必须严格晚于上一步（早到购买无法错位到后续步骤）；
    - 末步时间与首步时间之差不得超过 ``conversion_window_hours``；
    - 同一 ``order_id`` 只确认一次，重复提交只记一次；
    - 每个用户取最早成立的一条完整路径。
    """
    result = FunnelMatchResult(
        step_event_names=list(step_event_names),
        window_hours=conversion_window_hours,
    )
    if not step_event_names:
        return result

    window = timedelta(hours=conversion_window_hours)
    step_users: list[set[str]] = [set() for _ in step_event_names]
    confirmed_orders: set[Any] = set()

    for uid, raw_events in user_events.items():
        # 先统计重复提交，再在归一化后的事件流上匹配
        identities = [_event_identity(e) for e in raw_events]
        counted_duplicates: set[tuple[Any, ...]] = set()
        for event, key in zip(raw_events, identities):
            if (
                identities.count(key) > 1
                and event.get("event_name") == step_event_names[-1]
                and key not in counted_duplicates
            ):
                result.duplicate_events.append({"user_id": uid, **event})
                counted_duplicates.add(key)

        events = normalize_user_events({uid: raw_events})[uid]
        if not events:
            continue

        entry_events = [e for e in events if e.get("event_name") == step_event_names[0]]
        if not entry_events:
            continue
        step_users[0].add(uid)

        # 中间步骤：只要存在按时间顺序且在时窗内的步骤前缀，该用户就计入
        longest_prefix = _longest_ordered_prefix(
            events, entry_events, step_event_names, window
        )
        for index in range(1, longest_prefix):
            step_users[index].add(uid)

        confirmed = _match_from_entries(
            uid, events, entry_events, step_event_names, window, confirmed_orders
        )
        if confirmed is not None:
            step_users[-1].add(uid)
            order_id = confirmed["order_id"]
            if order_id is not None:
                confirmed_orders.add(order_id)
            result.confirmed_paths.append(confirmed)
            continue

        # 未形成完整路径：判定末步事件被拒绝的具体原因
        rejection = _classify_rejection(uid, events, step_event_names, window, confirmed_orders)
        if rejection is not None:
            result.rejected.append(rejection)

    result.step_counts = [len(users) for users in step_users]
    result.step_rates = [1.0]
    for i in range(1, len(result.step_counts)):
        prev = result.step_counts[i - 1]
        curr = result.step_counts[i]
        result.step_rates.append(round(curr / prev, 4) if prev > 0 else 0.0)
    if result.step_counts and result.step_counts[0] > 0:
        result.total_conversion_rate = round(
            result.step_counts[-1] / result.step_counts[0], 4
        )
    return result


def _longest_ordered_prefix(
    events: list[dict[str, Any]],
    entry_events: list[dict[str, Any]],
    step_event_names: list[str],
    window: timedelta,
) -> int:
    """返回用户能按时间顺序走到的最长步骤前缀长度（至少为 1）。"""
    best = 1
    for entry in entry_events:
        entry_time = parse_event_time(entry["timestamp"])
        cursor_time = entry_time
        depth = 1
        for step_name in step_event_names[1:]:
            candidate = next(
                (
                    e for e in events
                    if e.get("event_name") == step_name
                    and parse_event_time(e["timestamp"]) > cursor_time
                    and parse_event_time(e["timestamp"]) - entry_time <= window
                ),
                None,
            )
            if candidate is None:
                break
            depth += 1
            cursor_time = parse_event_time(candidate["timestamp"])
        best = max(best, depth)
        if best == len(step_event_names):
            break
    return best


def _match_from_entries(
    uid: str,
    events: list[dict[str, Any]],
    entry_events: list[dict[str, Any]],
    step_event_names: list[str],
    window: timedelta,
    confirmed_orders: set[Any],
) -> dict[str, Any] | None:
    """从每个可能的进店起点尝试匹配，返回最早成立的完整路径。"""
    for entry in entry_events:
        entry_time = parse_event_time(entry["timestamp"])
        matched = [entry]
        cursor_time = entry_time
        ok = True
        for step_name in step_event_names[1:]:
            candidate = next(
                (
                    e for e in events
                    if e.get("event_name") == step_name
                    and parse_event_time(e["timestamp"]) > cursor_time
                    and parse_event_time(e["timestamp"]) - entry_time <= window
                ),
                None,
            )
            if candidate is None:
                ok = False
                break
            matched.append(candidate)
            cursor_time = parse_event_time(candidate["timestamp"])
        if not ok:
            continue
        order_id = matched[-1].get("order_id")
        if order_id is not None and order_id in confirmed_orders:
            # 同一订单已在另一条路径确认，重复提交不能再次进入分母
            continue
        return {
            "user_id": uid,
            "channel": entry.get("channel"),
            "order_id": order_id,
            "entry_time": entry["timestamp"],
            "conversion_time": matched[-1]["timestamp"],
            "steps": [
                {"event_name": e.get("event_name"), "timestamp": e.get("timestamp")}
                for e in matched
            ],
        }
    return None


def _classify_rejection(
    uid: str,
    events: list[dict[str, Any]],
    step_event_names: list[str],
    window: timedelta,
    confirmed_orders: set[Any],
) -> dict[str, Any] | None:
    """对未转化用户给出末步事件被拒绝的原因。"""
    entry_name = step_event_names[0]
    final_name = step_event_names[-1]
    earliest_entry = next(
        (parse_event_time(e["timestamp"]) for e in events if e.get("event_name") == entry_name),
        None,
    )
    final_events = [e for e in events if e.get("event_name") == final_name]
    if earliest_entry is None or not final_events:
        return {
            "user_id": uid,
            "reason": REASON_INCOMPLETE,
            "events": [
                {"event_name": e.get("event_name"), "timestamp": e.get("timestamp")}
                for e in events
            ],
        }

    for final_event in final_events:
        final_time = parse_event_time(final_event["timestamp"])
        order_id = final_event.get("order_id")
        if order_id is not None and order_id in confirmed_orders:
            return {
                "user_id": uid,
                "order_id": order_id,
                "reason": REASON_DUPLICATE_ORDER,
                "event": {"event_name": final_name, "timestamp": final_event["timestamp"]},
            }
        # 末步事件不晚于进店：先购买后进店，属于早到误计
        if final_time <= earliest_entry:
            return {
                "user_id": uid,
                "order_id": order_id,
                "reason": REASON_OUT_OF_ORDER,
                "event": {"event_name": final_name, "timestamp": final_event["timestamp"]},
                "entry_time": next(
                    e["timestamp"] for e in events if e.get("event_name") == entry_name
                ),
            }
    # 所有末步事件都晚于进店，但落在时窗之外
    latest_final = max(parse_event_time(e["timestamp"]) for e in final_events)
    if latest_final - earliest_entry > window:
        final_event = next(
            e for e in final_events
            if parse_event_time(e["timestamp"]) == latest_final
        )
        return {
            "user_id": uid,
            "order_id": final_event.get("order_id"),
            "reason": REASON_OUTSIDE_WINDOW,
            "event": {"event_name": final_name, "timestamp": final_event["timestamp"]},
            "entry_time": next(
                e["timestamp"] for e in events if e.get("event_name") == entry_name
            ),
            "window_hours": window.total_seconds() / 3600,
        }
    return {
        "user_id": uid,
        "reason": REASON_INCOMPLETE,
        "events": [
            {"event_name": e.get("event_name"), "timestamp": e.get("timestamp")}
            for e in events
        ],
    }


@dataclass
class FunnelReview:
    """漏斗复盘：保存原始事件快照，发布物只含已确认路径，可随时复算。"""

    review_id: str
    funnel_name: str
    step_event_names: list[str]
    window_hours: int
    raw_events: dict[str, list[dict[str, Any]]]
    channel: str | None = None
    match: FunnelMatchResult | None = None
    published: bool = False
    published_at: str | None = None
    # 已发布分母：每步去重用户数与去重订单集合
    published_denominator: dict[str, Any] | None = None

    @classmethod
    def build(
        cls,
        review_id: str,
        funnel_name: str,
        step_event_names: list[str],
        window_hours: int,
        raw_events: dict[str, list[dict[str, Any]]],
        channel: str | None = None,
    ) -> "FunnelReview":
        match = match_funnel(step_event_names, window_hours, raw_events)
        return cls(
            review_id=review_id,
            funnel_name=funnel_name,
            step_event_names=list(step_event_names),
            window_hours=window_hours,
            raw_events=raw_events,
            channel=channel,
            match=match,
        )

    def publish(self, published_at: str | None = None) -> "FunnelReview":
        """发布复盘：重新核验每条确认路径，超窗/乱序路径一律不能发布。"""
        if self.match is None:
            raise ValueError("复盘尚未匹配，无法发布")
        # 不信任既有结论，用原始事件快照重新匹配一遍再发布
        revalidated = match_funnel(self.step_event_names, self.window_hours, self.raw_events)
        valid_keys = {
            (p["user_id"], p.get("order_id"), p["conversion_time"])
            for p in revalidated.confirmed_paths
        }
        for path in self.match.confirmed_paths:
            key = (path["user_id"], path.get("order_id"), path["conversion_time"])
            if key not in valid_keys:
                raise ValueError(f"路径未通过时窗与顺序复核，不能发布：{key}")
        self.match = revalidated
        self.published = True
        self.published_at = published_at or datetime.now(timezone.utc).isoformat()
        step_user_sets: list[set[str]] = [set() for _ in self.step_event_names]
        for path in revalidated.confirmed_paths:
            for index in range(len(self.step_event_names)):
                step_user_sets[index].add(path["user_id"])
        self.published_denominator = {
            "step_counts": [len(users) for users in step_user_sets],
            "order_ids": revalidated.conversion_order_ids,
            "conversion_count": len(revalidated.conversion_order_ids),
        }
        return self

    def recompute(self) -> "FunnelReview":
        """用保存的原始事件数据重新计算，验证已发布分母可复算。"""
        replay = FunnelReview.build(
            review_id=self.review_id,
            funnel_name=self.funnel_name,
            step_event_names=self.step_event_names,
            window_hours=self.window_hours,
            raw_events=self.raw_events,
            channel=self.channel,
        )
        if self.published:
            replay.publish(self.published_at)
        return replay
