"""漏斗时序修复 + 渠道归因复盘的端到端场景测试。

场景设定（进店 → 咨询 → 购买三步，转化窗口 24 小时）：

- 顾客 A：先购买后进店（早到购买），任何步骤都不能把这笔购买算成转化；
- 顾客 B：进店后第二天才购买（超过 24 小时窗口），超窗路径不能发布；
- 顾客 C：当天走完两条完整路径——先从广告进店并咨询、后从自然渠道
  进店并咨询，两条路径共同指向同一笔合法购买，产生归属冲突。

校验口径：
1. 同一用户每步最多计一人，重复订单只记一次；
2. 广告、自然渠道各发起一次漏斗复盘，只领用已确认的转化路径；
3. 归因从确认路径还原实际进店触点和购买时刻重新分配权重，
   同一笔购买全平台只贡献一次，且只归到有完整证据的渠道；
4. 早到、超窗购买均不参与归因；
5. 原始事件乱序送入与按时间送入，发布分母与结果完全一致，可复算。
"""
from __future__ import annotations

import copy
import json

import pytest

from app.attribution_review import (
    arbitrate_order_attribution,
    run_channel_review,
)
from app.analysis_pipeline import pipeline_channel_funnel_review
from app.funnel_engine import (
    REASON_OUT_OF_ORDER,
    REASON_OUTSIDE_WINDOW,
    FunnelReview,
    match_funnel,
)
from app.models import (
    AttributionAlgorithm,
    AttributionModel,
    EventTrack,
    FunnelDef,
)

STEP_NAMES = ["进店", "咨询", "购买"]
WINDOW_HOURS = 24
PAID = "广告"
ORGANIC = "自然"


def _steps():
    return [EventTrack(event_id=f"E{i}", event_name=name)
            for i, name in enumerate(STEP_NAMES)]


def _shuffle(events):
    """把事件列表倒序，模拟乱序送入。"""
    return list(reversed(events))


@pytest.fixture
def time_ordered_events():
    """按时间顺序送入的三名顾客原始事件。"""
    return {
        # 顾客 A：购买发生在进店之前（先购买后进店）
        "A": [
            {"event_name": "购买", "timestamp": "2026-11-20T08:00:00",
             "order_id": "ORDER-A", "channel": ORGANIC},
            {"event_name": "进店", "timestamp": "2026-11-20T10:00:00",
             "channel": ORGANIC},
            {"event_name": "咨询", "timestamp": "2026-11-20T11:00:00",
             "channel": ORGANIC},
        ],
        # 顾客 B：进店后第二天购买，间隔 26 小时，超出 24 小时窗口
        "B": [
            {"event_name": "进店", "timestamp": "2026-11-20T09:00:00",
             "channel": PAID},
            {"event_name": "咨询", "timestamp": "2026-11-20T10:00:00",
             "channel": PAID},
            {"event_name": "购买", "timestamp": "2026-11-21T11:00:00",
             "order_id": "ORDER-B", "channel": PAID},
        ],
        # 顾客 C：两条完整路径，同一笔合法购买
        "C": [
            {"event_name": "进店", "timestamp": "2026-11-20T08:00:00",
             "channel": PAID},
            {"event_name": "咨询", "timestamp": "2026-11-20T08:30:00",
             "channel": PAID},
            {"event_name": "进店", "timestamp": "2026-11-20T12:00:00",
             "channel": ORGANIC},
            {"event_name": "咨询", "timestamp": "2026-11-20T12:30:00",
             "channel": ORGANIC},
            {"event_name": "购买", "timestamp": "2026-11-20T13:00:00",
             "order_id": "ORDER-C"},
            # 同一订单重复提交，只能记一次
            {"event_name": "购买", "timestamp": "2026-11-20T13:00:00",
             "order_id": "ORDER-C"},
        ],
    }


@pytest.fixture
def shuffled_events(time_ordered_events):
    """同一批事件乱序（每用户内部倒序）送入。"""
    return {uid: _shuffle(events) for uid, events in time_ordered_events.items()}


def _filter_channel(events, channel):
    """按进店触点的渠道切分事件；购买事件本身不绑定渠道，两边都保留。"""
    split = {}
    for uid, user_event_list in events.items():
        kept = [
            e for e in user_event_list
            if e.get("channel") in (None, channel)
        ]
        if kept:
            split[uid] = kept
    return split


# ============================================================
# 漏斗引擎：时序与窗口核验
# ============================================================

class TestFunnelTimeOrdering:
    def test_early_purchase_never_counts_as_conversion(self, time_ordered_events):
        match = match_funnel(STEP_NAMES, WINDOW_HOURS, time_ordered_events)
        # 三名顾客都有进店；咨询：A 与 C（B 也有咨询，实际三人）；
        # 购买：只有 C 的订单在窗内且时序合法
        assert match.step_counts[0] == 3
        assert match.step_counts[1] == 3
        assert match.step_counts[2] == 1
        reasons = {item["reason"] for item in match.rejected}
        assert REASON_OUT_OF_ORDER in reasons
        assert REASON_OUTSIDE_WINDOW in reasons
        assert len(match.confirmed_paths) == 1
        assert match.confirmed_paths[0]["user_id"] == "C"
        assert match.conversion_order_ids == ["ORDER-C"]

    def test_early_purchase_rejection_detail(self, time_ordered_events):
        match = match_funnel(STEP_NAMES, WINDOW_HOURS, time_ordered_events)
        rejection_a = next(r for r in match.rejected if r["user_id"] == "A")
        assert rejection_a["reason"] == REASON_OUT_OF_ORDER
        assert rejection_a["order_id"] == "ORDER-A"

    def test_outside_window_rejection_detail(self, time_ordered_events):
        match = match_funnel(STEP_NAMES, WINDOW_HOURS, time_ordered_events)
        rejection_b = next(r for r in match.rejected if r["user_id"] == "B")
        assert rejection_b["reason"] == REASON_OUTSIDE_WINDOW
        assert rejection_b["order_id"] == "ORDER-B"
        assert rejection_b["window_hours"] == WINDOW_HOURS

    def test_duplicate_order_counts_once(self, time_ordered_events):
        match = match_funnel(STEP_NAMES, WINDOW_HOURS, time_ordered_events)
        c_orders = [
            p["order_id"] for p in match.confirmed_paths if p["user_id"] == "C"
        ]
        assert c_orders == ["ORDER-C"]
        # 重复提交被识别但不会产生第二条确认路径
        assert len(match.duplicate_events) == 1

    def test_each_user_counts_once_per_step(self, time_ordered_events):
        match = match_funnel(STEP_NAMES, WINDOW_HOURS, time_ordered_events)
        for count in match.step_counts:
            assert count <= 3  # 总共只有三名顾客
        users_per_step: list[set[str]] = [set() for _ in STEP_NAMES]
        for path in match.confirmed_paths:
            for step in path["steps"]:
                pass
        # C 有两次进店/咨询，但进店和咨询都只能算一人
        funnel = FunnelDef(
            funnel_id="F", funnel_name="f", steps=_steps(),
            conversion_window_hours=WINDOW_HOURS,
        ).compute_funnel({"C": time_ordered_events["C"]})
        assert funnel.step_counts == [1, 1, 1]

    def test_shuffled_and_ordered_ingest_match(self, time_ordered_events, shuffled_events):
        ordered = match_funnel(STEP_NAMES, WINDOW_HOURS, time_ordered_events)
        shuffled = match_funnel(STEP_NAMES, WINDOW_HOURS, shuffled_events)
        assert ordered.step_counts == shuffled.step_counts
        assert ordered.step_rates == shuffled.step_rates
        assert ordered.total_conversion_rate == shuffled.total_conversion_rate
        assert ordered.confirmed_paths == shuffled.confirmed_paths
        assert ordered.conversion_order_ids == shuffled.conversion_order_ids
        # 拒绝原因集合一致（逐条用 user_id + reason 比较，避免顺序依赖）
        key = lambda items: sorted((i["user_id"], i["reason"]) for i in items)
        assert key(ordered.rejected) == key(shuffled.rejected)


# ============================================================
# 渠道复盘：发布分母只含已确认路径
# ============================================================

@pytest.fixture
def linear_model():
    return AttributionModel(
        model_id="ATTR_LINEAR",
        model_name="线性归因",
        algorithm=AttributionAlgorithm.LINEAR,
        attribution_window_days=30,
    )


@pytest.fixture
def first_touch_model():
    return AttributionModel(
        model_id="ATTR_FIRST",
        model_name="首次触点归因",
        algorithm=AttributionAlgorithm.FIRST_TOUCH,
        attribution_window_days=30,
    )


class TestChannelReviews:
    def test_paid_channel_review_excludes_bad_orders(
        self, time_ordered_events, linear_model
    ):
        paid_events = _filter_channel(time_ordered_events, PAID)
        review = run_channel_review(
            review_id="REV-PAID",
            channel=PAID,
            step_event_names=STEP_NAMES,
            window_hours=WINDOW_HOURS,
            raw_events=paid_events,
            model=linear_model,
        )
        assert review["published"] is True
        # B 超窗被拒绝，C 的广告路径完整 -> 发布分母只含 C
        assert review["published_denominator"]["step_counts"][-1] == 1
        assert review["published_denominator"]["order_ids"] == ["ORDER-C"]
        rejected_orders = {r.get("order_id") for r in review["rejected"]}
        assert "ORDER-B" in rejected_orders
        assert "ORDER-A" not in rejected_orders  # A 没有广告触点，不在本渠道
        # 渠道归因只看到 C 的广告触点
        assert review["attribution"]["conversion_count"] == 1
        assert review["attribution"]["channel_weights"] == {PAID: 1.0}

    def test_organic_channel_review_excludes_early_purchase(
        self, time_ordered_events, linear_model
    ):
        organic_events = _filter_channel(time_ordered_events, ORGANIC)
        review = run_channel_review(
            review_id="REV-ORG",
            channel=ORGANIC,
            step_event_names=STEP_NAMES,
            window_hours=WINDOW_HOURS,
            raw_events=organic_events,
            model=linear_model,
        )
        # A 早到购买被拒绝；C 的自然路径完整 -> 同样只发布 C
        assert review["published_denominator"]["order_ids"] == ["ORDER-C"]
        rejected_orders = {r.get("order_id") for r in review["rejected"]}
        assert "ORDER-A" in rejected_orders
        assert review["attribution"]["channel_weights"] == {ORGANIC: 1.0}

    def test_outside_window_path_cannot_be_published(
        self, time_ordered_events, linear_model
    ):
        paid_events = _filter_channel(time_ordered_events, PAID)
        review = FunnelReview.build(
            review_id="REV-B-ONLY",
            funnel_name="仅B渠道复盘",
            step_event_names=STEP_NAMES,
            window_hours=WINDOW_HOURS,
            raw_events={"B": paid_events["B"]},
            channel=PAID,
        )
        assert review.match.confirmed_paths == []
        review.publish()
        # 没有确认路径时发布分母为 0，ORDER-B 永远进不来
        assert review.published_denominator["conversion_count"] == 0
        assert review.published_denominator["order_ids"] == []


# ============================================================
# 跨渠道仲裁：同一笔购买只贡献一次
# ============================================================

class TestCrossChannelArbitration:
    @pytest.fixture
    def channel_reviews(self, time_ordered_events, linear_model):
        results = []
        for channel, review_id in ((PAID, "REV-PAID"), (ORGANIC, "REV-ORG")):
            results.append(run_channel_review(
                review_id=review_id,
                channel=channel,
                step_event_names=STEP_NAMES,
                window_hours=WINDOW_HOURS,
                raw_events=_filter_channel(time_ordered_events, channel),
                model=linear_model,
            ))
        return results

    def test_single_order_contributes_once(self, channel_reviews, linear_model):
        ledger = arbitrate_order_attribution(channel_reviews, linear_model)
        assert ledger["conversion_count"] == 1
        assert list(ledger["order_ledger"]) == ["ORDER-C"]
        # 早到和超窗订单被排除在归因之外
        assert set(ledger["excluded_orders"]) == {"ORDER-A", "ORDER-B"}

    def test_weight_split_between_evidenced_channels(
        self, channel_reviews, linear_model
    ):
        ledger = arbitrate_order_attribution(channel_reviews, linear_model)
        order_c = ledger["order_ledger"]["ORDER-C"]
        assert set(order_c["channels"]) == {PAID, ORGANIC}
        # 线性归因：两个渠道各 0.5
        assert order_c["weights"] == {PAID: 0.5, ORGANIC: 0.5}
        assert abs(sum(order_c["weights"].values()) - 1.0) < 1e-6
        assert ledger["channel_weights"] == {PAID: 0.5, ORGANIC: 0.5}

    def test_first_touch_awards_paid_channel(
        self, time_ordered_events, first_touch_model
    ):
        reviews = []
        for channel, review_id in ((PAID, "REV-PAID"), (ORGANIC, "REV-ORG")):
            reviews.append(run_channel_review(
                review_id=review_id,
                channel=channel,
                step_event_names=STEP_NAMES,
                window_hours=WINDOW_HOURS,
                raw_events=_filter_channel(time_ordered_events, channel),
                model=first_touch_model,
            ))
        ledger = arbitrate_order_attribution(reviews, first_touch_model)
        order_c = ledger["order_ledger"]["ORDER-C"]
        # 广告进店（08:00）早于自然进店（12:00），首次触点 100% 归广告
        assert order_c["weights"] == {PAID: 1.0}
        assert ORGANIC not in order_c["weights"]

    def test_attribution_uses_real_touch_and_purchase_time(
        self, channel_reviews, linear_model
    ):
        for review in channel_reviews:
            for path in review["confirmed_paths"]:
                # 归因输入直接来自确认路径上的真实进店与购买时刻
                assert path["entry_time"] < path["conversion_time"]
                assert path["order_id"] == "ORDER-C"


# ============================================================
# 可复算：已发布分母和结果能用原始事件重新算出
# ============================================================

class TestRecomputability:
    def test_recompute_reproduces_published_denominator(
        self, time_ordered_events, shuffled_events, linear_model
    ):
        for raw, label in (
            (time_ordered_events, "ordered"),
            (shuffled_events, "shuffled"),
        ):
            review = FunnelReview.build(
                review_id=f"REV-{label}",
                funnel_name="全渠道复盘",
                step_event_names=STEP_NAMES,
                window_hours=WINDOW_HOURS,
                raw_events=copy.deepcopy(raw),
            ).publish(published_at="2026-11-20T14:00:00+00:00")
            replay = review.recompute()
            assert replay.published is True
            assert replay.published_at == review.published_at
            assert replay.published_denominator == review.published_denominator
            assert replay.match.confirmed_paths == review.match.confirmed_paths
            assert replay.match.step_counts == review.match.step_counts

    def test_replay_after_mutating_pipeline_inputs(
        self, time_ordered_events, linear_model
    ):
        raw_snapshot = copy.deepcopy(time_ordered_events)
        review = FunnelReview.build(
            review_id="REV-SNAPSHOT",
            funnel_name="快照复盘",
            step_event_names=STEP_NAMES,
            window_hours=WINDOW_HOURS,
            raw_events=raw_snapshot,
        ).publish()
        published = copy.deepcopy(review.published_denominator)
        # 外部清空原始仓库，复盘内部快照仍可独立复算
        time_ordered_events.clear()
        replay = review.recompute()
        assert replay.published_denominator == published
        assert replay.published_denominator["order_ids"] == ["ORDER-C"]

    def test_publish_revalidates_window_and_order(
        self, time_ordered_events, linear_model
    ):
        review = FunnelReview.build(
            review_id="REV-REVALIDATE",
            funnel_name="复核复盘",
            step_event_names=STEP_NAMES,
            window_hours=WINDOW_HOURS,
            raw_events=copy.deepcopy(time_ordered_events),
        )
        review.publish()
        # 篡改已确认路径后，发布会重新用原始事件复核并拒绝发布
        review.match.confirmed_paths.append({
            "user_id": "B",
            "channel": PAID,
            "order_id": "ORDER-B",
            "entry_time": "2026-11-20T09:00:00",
            "conversion_time": "2026-11-21T11:00:00",
            "steps": [],
        })
        with pytest.raises(ValueError):
            review.publish()


# ============================================================
# 分析管道集成
# ============================================================

class TestPipelineIntegration:
    def test_channel_review_pipeline(self, time_ordered_events, linear_model):
        result = pipeline_channel_funnel_review(
            models=[linear_model],
            step_event_names=STEP_NAMES,
            window_hours=WINDOW_HOURS,
            raw_events_by_channel={
                PAID: _filter_channel(time_ordered_events, PAID),
                ORGANIC: _filter_channel(time_ordered_events, ORGANIC),
            },
        )
        assert len(result["reviews"]) == 2
        assert result["arbitration"]["conversion_count"] == 1
        assert result["arbitration"]["channel_weights"] == {PAID: 0.5, ORGANIC: 0.5}

    def test_pipeline_without_model_returns_empty(self):
        result = pipeline_channel_funnel_review(
            models=[],
            step_event_names=STEP_NAMES,
            window_hours=WINDOW_HOURS,
            raw_events_by_channel={},
        )
        assert result["reviews"] == []
        assert result["arbitration"] is None


# ============================================================
# HTTP 接口（直接走 ASGI，不依赖 httpx）
# ============================================================

async def _asgi_post(app, path, payload):
    body = json.dumps(payload).encode()
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode())],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    received = []

    async def receive():
        if not received:
            received.append(True)
            return {"type": "http.request", "body": body}
        return {"type": "http.disconnect"}

    status = {}
    chunks = []

    async def send(message):
        if message["type"] == "http.response.start":
            status["code"] = message["status"]
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    await app(scope, receive, send)
    return status["code"], json.loads(b"".join(chunks))


class TestFunnelReviewAPI:
    def test_compute_endpoint_rejects_bad_orders_and_publishes(
        self, time_ordered_events
    ):
        import asyncio

        from app.main import app

        code, body = asyncio.run(_asgi_post(app, "/api/funnel-review/compute", {
            "review_id": "REV-API",
            "step_event_names": STEP_NAMES,
            "conversion_window_hours": WINDOW_HOURS,
            "user_events": time_ordered_events,
        }))
        assert code == 200
        assert body["ok"] is True
        assert body["published"] is True
        assert body["step_counts"] == [3, 3, 1]
        assert body["published_denominator"]["order_ids"] == ["ORDER-C"]
        reasons = {item["reason"] for item in body["rejected"]}
        assert reasons == {REASON_OUT_OF_ORDER, REASON_OUTSIDE_WINDOW}

    def test_channel_attribution_endpoint(self, time_ordered_events):
        import asyncio

        from app.main import app

        code, body = asyncio.run(_asgi_post(
            app, "/api/funnel-review/channel-attribution", {
                "step_event_names": STEP_NAMES,
                "conversion_window_hours": WINDOW_HOURS,
                "attribution_algorithm": "first_touch",
                "events_by_channel": {
                    PAID: _filter_channel(time_ordered_events, PAID),
                    ORGANIC: _filter_channel(time_ordered_events, ORGANIC),
                },
            }
        ))
        assert code == 200
        arbitration = body["arbitration"]
        assert arbitration["conversion_count"] == 1
        # 首次触点：广告进店早于自然进店，全归广告
        assert arbitration["order_ledger"]["ORDER-C"]["weights"] == {PAID: 1.0}
        assert set(arbitration["excluded_orders"]) == {"ORDER-A", "ORDER-B"}
