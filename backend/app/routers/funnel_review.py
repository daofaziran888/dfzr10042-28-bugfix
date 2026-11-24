"""漏斗复盘接口：时序核验、发布、复算与渠道归因复盘。

与通用资源路由分开承载，因为这里处理的是带原始事件快照的分析动作，
而不是模块条目的 CRUD。所有计算都走 ``funnel_engine`` 与
``attribution_review``，保证乱序/按时间送入结果一致、可复算。
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.attribution_review import (
    arbitrate_order_attribution,
    run_channel_review,
)
from app.funnel_engine import FunnelReview
from app.models import AttributionAlgorithm, AttributionModel

router = APIRouter(prefix="/api/funnel-review", tags=["漏斗复盘"])


class FunnelReviewPayload(BaseModel):
    review_id: str
    funnel_name: str = "渠道漏斗复盘"
    step_event_names: list[str]
    conversion_window_hours: int = Field(default=24, ge=1)
    # {user_id: [{event_name, timestamp, order_id?, channel?}]}
    user_events: dict[str, list[dict[str, Any]]]
    channel: str | None = None
    publish: bool = True


class ChannelReviewPayload(BaseModel):
    step_event_names: list[str]
    conversion_window_hours: int = Field(default=24, ge=1)
    attribution_algorithm: AttributionAlgorithm = AttributionAlgorithm.LINEAR
    attribution_window_days: int = Field(default=30, ge=1)
    model_id: str = "ATTR-API"
    # {channel: {user_id: [event, ...]}}
    events_by_channel: dict[str, dict[str, list[dict[str, Any]]]]


@router.post("/compute")
def compute_review(payload: FunnelReviewPayload) -> dict[str, Any]:
    """按事件时间核验漏斗，可选直接发布；返回分母、确认路径与拒绝明细。"""
    review = FunnelReview.build(
        review_id=payload.review_id,
        funnel_name=payload.funnel_name,
        step_event_names=payload.step_event_names,
        window_hours=payload.conversion_window_hours,
        raw_events=payload.user_events,
        channel=payload.channel,
    )
    if payload.publish:
        review.publish()
    return {
        "ok": True,
        "review_id": review.review_id,
        "published": review.published,
        "step_counts": review.match.step_counts,
        "step_rates": review.match.step_rates,
        "total_conversion_rate": review.match.total_conversion_rate,
        "confirmed_paths": review.match.confirmed_paths,
        "rejected": review.match.rejected,
        "duplicate_events": review.match.duplicate_events,
        "published_denominator": review.published_denominator,
    }


@router.post("/channel-attribution")
def channel_attribution(payload: ChannelReviewPayload) -> dict[str, Any]:
    """按渠道各发起一次漏斗复盘，并对同一笔购买做跨渠道归因仲裁。"""
    model = AttributionModel(
        model_id=payload.model_id,
        model_name=f"归因模型-{payload.attribution_algorithm.value}",
        algorithm=payload.attribution_algorithm,
        attribution_window_days=payload.attribution_window_days,
    )
    reviews = []
    for index, (channel, raw_events) in enumerate(payload.events_by_channel.items(), start=1):
        reviews.append(run_channel_review(
            review_id=f"REV-{payload.model_id}-{index}",
            channel=channel,
            step_event_names=payload.step_event_names,
            window_hours=payload.conversion_window_hours,
            raw_events=raw_events,
            model=model,
        ))
    arbitration = arbitrate_order_attribution(reviews, model)
    return {
        "ok": True,
        "reviews": [
            {
                "review_id": item["review_id"],
                "channel": item["channel"],
                "published_denominator": item["published_denominator"],
                "step_counts": item["step_counts"],
                "total_conversion_rate": item["total_conversion_rate"],
                "confirmed_paths": item["confirmed_paths"],
                "rejected": item["rejected"],
                "channel_weights": item["attribution"]["channel_weights"],
            }
            for item in reviews
        ],
        "arbitration": arbitration,
    }
