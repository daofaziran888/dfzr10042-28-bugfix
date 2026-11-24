"""渠道漏斗复盘与归因仲裁。

流程对应业务要求：

1. 按渠道（广告 / 自然）各发起一次漏斗复盘，漏斗引擎按事件时间核验
   步骤顺序和时窗，只把有完整证据的路径发布进复盘分母；
2. 渠道归因算法从本渠道领到的确认路径中还原实际进店触点和购买时刻，
   重新分配渠道权重；
3. 同一名顾客当天走两条完整路径、同一笔合法购买产生归属冲突时，由
   ``arbitrate_order_attribution`` 统一仲裁：该订单在全平台只贡献一次，
   按选定归因算法在各渠道证据触点之间分配权重；
4. 早到购买和超窗购买在漏斗匹配阶段即被拒绝，不进入任何渠道的分母，
   也不参与归因。

全部函数确定性可复算：原始事件快照保存在每个 ``FunnelReview`` 中，
无论乱序还是按时间送入，发布结果都能用原事件数据重新算出同样结果。
"""
from __future__ import annotations

from typing import Any

from app.funnel_engine import FunnelReview
from app.models import AttributionModel


def run_channel_review(
    *,
    review_id: str,
    channel: str,
    step_event_names: list[str],
    window_hours: int,
    raw_events: dict[str, list[dict[str, Any]]],
    model: AttributionModel,
) -> dict[str, Any]:
    """发起单个渠道的漏斗复盘并跑渠道归因。

    ``raw_events`` 应只包含该渠道的进店触点相关事件（通常先按事件上的
    ``channel`` 字段过滤）。复盘发布后才能驱动归因，未发布复盘直接报错。
    """
    review = FunnelReview.build(
        review_id=review_id,
        funnel_name=f"{channel}渠道漏斗复盘",
        step_event_names=step_event_names,
        window_hours=window_hours,
        raw_events=raw_events,
        channel=channel,
    )
    review.publish()
    attribution = model.attribute_paths(review.match.confirmed_paths)
    return {
        "review_id": review_id,
        "channel": channel,
        "published": review.published,
        "published_denominator": review.published_denominator,
        "step_counts": review.match.step_counts,
        "step_rates": review.match.step_rates,
        "total_conversion_rate": review.match.total_conversion_rate,
        "confirmed_paths": review.match.confirmed_paths,
        "rejected": review.match.rejected,
        "attribution": attribution,
        "review": review,
    }


def arbitrate_order_attribution(
    channel_reviews: list[dict[str, Any]],
    model: AttributionModel,
) -> dict[str, Any]:
    """跨渠道订单仲裁：同一笔购买只贡献一次，只归到有完整证据的渠道。

    输入是多个 ``run_channel_review`` 的产物。对每个去重 ``order_id``：
    收集各渠道已确认路径上的进店触点（触点均已通过顺序与时窗核验），
    交给归因模型重新分配权重；没有订单号的确认路径按用户兜底，仍只计一次。

    返回：
    - ``order_ledger``：逐订单的渠道权重明细，所有订单权重和恒为 1；
    - ``channel_weights``：按订单平均后汇总的渠道权重（同样归一化到 1）；
    - ``conversion_count``：去重后的转化订单数；
    - ``excluded_orders``：只出现在拒绝清单、没有任何完整证据的订单。
    """
    # 订单 -> 各渠道确认路径
    order_paths: dict[Any, list[dict[str, Any]]] = {}
    confirmed_order_ids: set[Any] = set()
    rejected_order_ids: set[Any] = set()
    for review in channel_reviews:
        for path in review["confirmed_paths"]:
            order_id = path.get("order_id")
            key = order_id if order_id is not None else f"user:{path['user_id']}"
            order_paths.setdefault(key, []).append({**path, "order_id": key})
            confirmed_order_ids.add(key)
        for rejected in review["rejected"]:
            if rejected.get("order_id") is not None:
                rejected_order_ids.add(rejected["order_id"])

    order_ledger: dict[str, dict[str, Any]] = {}
    channel_weight_accumulator: dict[str, float] = {}
    for order_id, paths in order_paths.items():
        # 每个渠道对同一订单最多贡献一条证据路径
        unique_paths: dict[str, dict[str, Any]] = {}
        for path in paths:
            unique_paths.setdefault(path.get("channel", "unknown"), path)
        result = model.attribute_paths(
            list(unique_paths.values()), dedupe_orders=False
        )
        weights = result["channel_weights"]
        order_ledger[str(order_id)] = {
            "channels": list(unique_paths),
            "weights": weights,
            "conversion_time": paths[0]["conversion_time"],
            "evidence_count": len(unique_paths),
        }
        for ch, weight in weights.items():
            channel_weight_accumulator[ch] = (
                channel_weight_accumulator.get(ch, 0.0) + weight
            )

    total_orders = len(order_ledger)
    channel_weights: dict[str, float] = {}
    if total_orders > 0:
        for ch, accumulated in channel_weight_accumulator.items():
            channel_weights[ch] = round(accumulated / total_orders, 4)
    # 归一化兜底，消除浮点尾差
    weight_sum = sum(channel_weights.values())
    if weight_sum:
        channel_weights = {
            ch: round(w / weight_sum, 4) for ch, w in channel_weights.items()
        }

    excluded = sorted(
        oid for oid in rejected_order_ids if oid not in confirmed_order_ids
    )
    return {
        "model_id": model.model_id,
        "algorithm": model.algorithm.value,
        "order_ledger": order_ledger,
        "channel_weights": channel_weights,
        "conversion_count": total_orders,
        "excluded_orders": excluded,
    }
