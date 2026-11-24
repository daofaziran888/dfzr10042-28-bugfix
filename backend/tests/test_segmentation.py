"""洞察工坊 · 分群和漏斗测试

测试群体圈选、分群比较、漏斗分析、同期群留存和归因算法的确定性和正确性。
"""
from __future__ import annotations

import math

import pytest

from app.models import (
    AggregationMethod,
    AnomalyAlgorithm,
    AnomalyDetect,
    AttributionAlgorithm,
    AttributionModel,
    CohortAnalysis,
    DimensionDef,
    EventTrack,
    FunnelDef,
    MetricDataSource,
    MetricDef,
    MetricPeriod,
    Segment,
    _first_touch_attribution,
    _iqr_anomaly_detect,
    _last_touch_attribution,
    _linear_attribution,
    _moving_average_anomaly_detect,
    _percentile,
    _position_based_attribution,
    _prepare_attribution_inputs,
    _time_decay_attribution,
    _zscore_anomaly_detect,
)


def _prepared(touchpoints, conversions, window_days=30):
    """旧测试辅助：按新签名生成清洗后的归因输入。"""
    return _prepare_attribution_inputs(touchpoints, conversions, window_days)


# ============================================================
# 群体圈选测试
# ============================================================

class TestSegmentBuilder:
    @pytest.fixture
    def sample_users(self):
        return [
            {"user_id": "u1", "platform": "ios", "channel": "organic", "age_group": "18-25", "lifetime_value": 500},
            {"user_id": "u2", "platform": "android", "channel": "paid", "age_group": "26-35", "lifetime_value": 1200},
            {"user_id": "u3", "platform": "ios", "channel": "organic", "age_group": "18-25", "lifetime_value": 800},
            {"user_id": "u4", "platform": "android", "channel": "paid", "age_group": "36-45", "lifetime_value": 300},
            {"user_id": "u5", "platform": "ios", "channel": "wechat", "age_group": "18-25", "lifetime_value": 600},
            {"user_id": "u6", "platform": "android", "channel": "organic", "age_group": "26-35", "lifetime_value": 1500},
            {"user_id": "u7", "platform": "ios", "channel": "paid", "age_group": "26-35", "lifetime_value": 900},
            {"user_id": "u8", "platform": "android", "channel": "wechat", "age_group": "36-45", "lifetime_value": 400},
        ]

    def test_select_ios_users(self, sample_users):
        seg = Segment(
            segment_id="SEG_IOS",
            segment_name="iOS用户",
            conditions={"platform": "ios"},
        )
        result = seg.select_users(sample_users)
        assert seg.member_count == 4  # u1, u3, u5, u7
        assert len(result) == 4

    def test_select_organic_ios_users(self, sample_users):
        seg = Segment(
            segment_id="SEG_ORG_IOS",
            segment_name="iOS自然用户",
            conditions={"platform": "ios", "channel": "organic"},
        )
        result = seg.select_users(sample_users)
        assert seg.member_count == 2  # u1, u3

    def test_select_no_results(self, sample_users):
        seg = Segment(
            segment_id="SEG_EMPTY",
            segment_name="空群体",
            conditions={"platform": "windows"},
        )
        result = seg.select_users(sample_users)
        assert seg.member_count == 0
        assert len(result) == 0

    def test_select_all_when_no_conditions(self, sample_users):
        seg = Segment(
            segment_id="SEG_ALL",
            segment_name="全部用户",
            conditions={},
        )
        result = seg.select_users(sample_users)
        assert seg.member_count == 8
        assert len(result) == 8

    def test_select_is_deterministic(self, sample_users):
        seg = Segment(
            segment_id="SEG_DET",
            segment_name="确定性测试",
            conditions={"age_group": "18-25"},
        )
        r1 = seg.select_users(sample_users.copy())
        r2 = seg.select_users(sample_users.copy())
        assert len(r1) == len(r2)
        assert [u["user_id"] for u in r1] == [u["user_id"] for u in r2]


# ============================================================
# 事件追踪测试
# ============================================================

class TestEventTrack:
    def test_matches_properties_exact(self):
        event = EventTrack(
            event_id="E1",
            event_name="注册",
            event_properties={"source": "organic", "platform": "ios"},
        )
        assert event.matches_properties({"source": "organic"}) is True
        assert event.matches_properties({"source": "paid"}) is False

    def test_matches_properties_all(self):
        event = EventTrack(
            event_id="E2",
            event_name="购买",
            event_properties={"source": "paid", "channel": "app", "amount": "100"},
        )
        assert event.matches_properties({"source": "paid", "channel": "app"}) is True
        assert event.matches_properties({"source": "paid", "channel": "wechat"}) is False

    def test_matches_empty_conditions(self):
        event = EventTrack(
            event_id="E3",
            event_name="任意事件",
            event_properties={"source": "test"},
        )
        assert event.matches_properties({}) is True


# ============================================================
# 漏斗分析测试
# ============================================================

class TestFunnelDef:
    def test_compute_funnel_conversion(self):
        steps = [
            EventTrack(event_id="E_REG", event_name="用户注册"),
            EventTrack(event_id="E_LOG", event_name="首次登录"),
            EventTrack(event_id="E_PUR", event_name="支付成功"),
        ]
        funnel = FunnelDef(
            funnel_id="FUN_ACQ",
            funnel_name="获客漏斗",
            steps=steps,
            conversion_window_hours=24,
        )
        user_events = {
            "u1": [
                {"event_name": "用户注册", "timestamp": "2026-11-01T08:00:00"},
                {"event_name": "首次登录", "timestamp": "2026-11-01T09:00:00"},
                {"event_name": "支付成功", "timestamp": "2026-11-01T10:00:00"},
            ],
            "u2": [
                {"event_name": "用户注册", "timestamp": "2026-11-01T08:00:00"},
                {"event_name": "首次登录", "timestamp": "2026-11-01T09:00:00"},
            ],
            "u3": [
                {"event_name": "用户注册", "timestamp": "2026-11-01T08:00:00"},
            ],
            "u4": [
                {"event_name": "首次登录", "timestamp": "2026-11-01T08:00:00"},
            ],
        }
        result = funnel.compute_funnel(user_events)
        assert result.step_counts == [3, 2, 1]  # 3 register, 2 login, 1 purchase
        assert result.step_rates[0] == 1.0
        assert result.step_rates[1] == round(2/3, 4)  # 2/3
        assert result.step_rates[2] == 0.5  # 1/2
        assert result.total_conversion_rate == round(1/3, 4)
        assert len(result.conversion_paths) == 1
        assert result.conversion_paths[0]["user_id"] == "u1"

    def test_funnel_rejects_purchase_before_entry(self):
        """先购买后进店：早到购买不能错位计入后续步骤。"""
        steps = [
            EventTrack(event_id="E1", event_name="进店"),
            EventTrack(event_id="E2", event_name="购买"),
        ]
        funnel = FunnelDef(
            funnel_id="FUN_EARLY", funnel_name="早到购买",
            steps=steps, conversion_window_hours=24,
        )
        user_events = {
            "u1": [
                {"event_name": "购买", "timestamp": "2026-11-01T08:00:00", "order_id": "O1"},
                {"event_name": "进店", "timestamp": "2026-11-01T10:00:00"},
            ],
        }
        result = funnel.compute_funnel(user_events)
        assert result.step_counts == [1, 0]
        assert result.total_conversion_rate == 0.0
        assert result.conversion_paths == []
        assert result.rejected_paths[0]["reason"] == "out_of_order_purchase"

    def test_funnel_rejects_purchase_outside_window(self):
        """超过转化窗口的购买不能残留在转化分母里。"""
        steps = [
            EventTrack(event_id="E1", event_name="进店"),
            EventTrack(event_id="E2", event_name="购买"),
        ]
        funnel = FunnelDef(
            funnel_id="FUN_WIN", funnel_name="超窗购买",
            steps=steps, conversion_window_hours=24,
        )
        user_events = {
            "u1": [
                {"event_name": "进店", "timestamp": "2026-11-01T10:00:00"},
                {"event_name": "购买", "timestamp": "2026-11-03T10:00:00", "order_id": "O2"},
            ],
        }
        result = funnel.compute_funnel(user_events)
        assert result.step_counts == [1, 0]
        assert result.conversion_paths == []
        assert result.rejected_paths[0]["reason"] == "outside_window"

    def test_funnel_duplicate_order_counts_once(self):
        """同一订单重复提交只确认一次，同一用户每步最多计一人。"""
        steps = [
            EventTrack(event_id="E1", event_name="进店"),
            EventTrack(event_id="E2", event_name="购买"),
        ]
        funnel = FunnelDef(
            funnel_id="FUN_DUP", funnel_name="重复订单",
            steps=steps, conversion_window_hours=24,
        )
        user_events = {
            "u1": [
                {"event_name": "进店", "timestamp": "2026-11-01T08:00:00"},
                {"event_name": "购买", "timestamp": "2026-11-01T09:00:00", "order_id": "O3"},
                {"event_name": "购买", "timestamp": "2026-11-01T09:00:00", "order_id": "O3"},
            ],
        }
        result = funnel.compute_funnel(user_events)
        assert result.step_counts == [1, 1]
        assert len(result.conversion_paths) == 1
        assert result.conversion_paths[0]["order_id"] == "O3"

    def test_funnel_all_convert(self):
        steps = [
            EventTrack(event_id="E1", event_name="页面浏览"),
            EventTrack(event_id="E2", event_name="点击商品"),
        ]
        funnel = FunnelDef(
            funnel_id="FUN_ALL", funnel_name="全面转化",
            steps=steps, conversion_window_hours=24,
        )
        user_events = {
            "u1": [
                {"event_name": "页面浏览", "timestamp": "2026-11-01T10:00:00"},
                {"event_name": "点击商品", "timestamp": "2026-11-01T11:00:00"},
            ],
            "u2": [
                {"event_name": "页面浏览", "timestamp": "2026-11-01T10:00:00"},
                {"event_name": "点击商品", "timestamp": "2026-11-01T11:00:00"},
            ],
        }
        result = funnel.compute_funnel(user_events)
        assert result.step_counts == [2, 2]
        assert result.total_conversion_rate == 1.0

    def test_funnel_empty_steps(self):
        funnel = FunnelDef(funnel_id="FUN_EMPTY", funnel_name="空漏斗", steps=[])
        user_events = {"u1": [{"event_name": "test", "timestamp": "2026-11-01T10:00:00"}]}
        result = funnel.compute_funnel(user_events)
        assert result.step_counts == []

    def test_funnel_is_deterministic(self):
        steps = [
            EventTrack(event_id="E1", event_name="浏览"),
            EventTrack(event_id="E2", event_name="购买"),
        ]
        funnel = FunnelDef(
            funnel_id="FUN_DET", funnel_name="确定性",
            steps=steps, conversion_window_hours=24,
        )
        user_events = {
            "u1": [
                {"event_name": "浏览", "timestamp": "2026-11-01T10:00:00"},
                {"event_name": "购买", "timestamp": "2026-11-01T11:00:00"},
            ],
            "u2": [{"event_name": "浏览", "timestamp": "2026-11-01T10:00:00"}],
        }
        r1 = funnel.compute_funnel(dict(user_events))
        r2 = funnel.compute_funnel(dict(user_events))
        assert r1.step_counts == r2.step_counts
        assert r1.total_conversion_rate == r2.total_conversion_rate
        assert r1.conversion_paths == r2.conversion_paths

    def test_funnel_ingest_order_invariant(self):
        """乱序送入和按时间送入的匹配结果一致。"""
        steps = [
            EventTrack(event_id="E1", event_name="进店"),
            EventTrack(event_id="E2", event_name="咨询"),
            EventTrack(event_id="E3", event_name="购买"),
        ]
        ordered = {
            "u1": [
                {"event_name": "进店", "timestamp": "2026-11-01T08:00:00"},
                {"event_name": "咨询", "timestamp": "2026-11-01T09:00:00"},
                {"event_name": "购买", "timestamp": "2026-11-01T10:00:00", "order_id": "O4"},
            ],
        }
        shuffled = {
            "u1": [
                {"event_name": "购买", "timestamp": "2026-11-01T10:00:00", "order_id": "O4"},
                {"event_name": "进店", "timestamp": "2026-11-01T08:00:00"},
                {"event_name": "咨询", "timestamp": "2026-11-01T09:00:00"},
            ],
        }
        f1 = FunnelDef(funnel_id="F1", funnel_name="f", steps=steps,
                       conversion_window_hours=24).compute_funnel(ordered)
        f2 = FunnelDef(funnel_id="F2", funnel_name="f", steps=steps,
                       conversion_window_hours=24).compute_funnel(shuffled)
        assert f1.step_counts == f2.step_counts == [1, 1, 1]
        assert f1.conversion_paths == f2.conversion_paths


# ============================================================
# 同期群分析测试
# ============================================================

class TestCohortAnalysis:
    def test_compute_retention(self):
        cohort = CohortAnalysis(
            cohort_id="COH_WEEK1",
            cohort_name="Week1新用户",
            observation_windows=[1, 3, 7],
        )
        users = ["u1", "u2", "u3", "u4", "u5"]
        user_events = {
            "u1": ["1", "3", "7"],
            "u2": ["7"],
            "u3": ["1", "3", "7"],
            "u4": ["1", "3"],
            "u5": ["1", "3", "7"],
        }
        retentions = cohort.compute_retention(users, user_events)
        assert retentions[0] == 0.8  # Day 1: u1,u3,u4,u5 have event <=1
        assert retentions[1] == 0.8  # Day 3: u1,u3,u4,u5 have event <=3
        assert retentions[2] == 1.0  # Day 7: all users have event <=7

    def test_empty_cohort(self):
        cohort = CohortAnalysis(
            cohort_id="COH_EMPTY",
            cohort_name="空群组",
            observation_windows=[1, 3, 7],
        )
        retentions = cohort.compute_retention([], {})
        assert retentions == [0.0, 0.0, 0.0]
        assert cohort.member_base_count == 0

    def test_all_retained(self):
        cohort = CohortAnalysis(
            cohort_id="COH_FULL",
            cohort_name="全部留存",
            observation_windows=[1, 14, 30],
        )
        users = ["u1", "u2"]
        user_events = {
            "u1": ["1", "14", "30"],
            "u2": ["1", "14", "30"],
        }
        retentions = cohort.compute_retention(users, user_events)
        assert retentions == [1.0, 1.0, 1.0]

    def test_retention_is_deterministic(self):
        cohort = CohortAnalysis(
            cohort_id="COH_DET",
            cohort_name="确定性",
            observation_windows=[1, 7, 14],
        )
        users = ["u1", "u2", "u3"]
        user_events = {"u1": ["1", "7"], "u2": ["1"], "u3": ["1", "7", "14"]}
        r1 = cohort.compute_retention(users, user_events)
        r2 = cohort.compute_retention(users, user_events)
        assert r1 == r2


# ============================================================
# 归因算法测试
# ============================================================

class TestAttributionModel:
    @pytest.fixture
    def touchpoints(self):
        return [
            {"user_id": "u1", "channel": "organic", "timestamp": "2026-11-01T08:00:00Z"},
            {"user_id": "u1", "channel": "paid", "timestamp": "2026-11-01T10:00:00Z"},
            {"user_id": "u1", "channel": "email", "timestamp": "2026-11-01T12:00:00Z"},
            {"user_id": "u2", "channel": "paid", "timestamp": "2026-11-01T09:00:00Z"},
            {"user_id": "u2", "channel": "organic", "timestamp": "2026-11-01T14:00:00Z"},
            {"user_id": "u3", "channel": "wechat", "timestamp": "2026-11-01T07:00:00Z"},
            {"user_id": "u3", "channel": "email", "timestamp": "2026-11-01T11:00:00Z"},
            {"user_id": "u3", "channel": "paid", "timestamp": "2026-11-01T16:00:00Z"},
        ]

    @pytest.fixture
    def conversions(self):
        return [
            {"user_id": "u1", "conversion_time": "2026-11-01T13:00:00Z"},
            {"user_id": "u2", "conversion_time": "2026-11-01T15:00:00Z"},
            {"user_id": "u3", "conversion_time": "2026-11-01T17:00:00Z"},
        ]

    def test_first_touch_attribution(self, touchpoints, conversions):
        weights = _first_touch_attribution(_prepared(touchpoints, conversions))
        # u1 first: organic, u2 first: paid, u3 first: wechat
        assert weights["organic"] == round(1/3, 4)
        assert weights["paid"] == round(1/3, 4)
        assert weights["wechat"] == round(1/3, 4)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_last_touch_attribution(self, touchpoints, conversions):
        weights = _last_touch_attribution(_prepared(touchpoints, conversions))
        # u1 last: email, u2 last: organic, u3 last: paid
        assert weights["email"] == round(1/3, 4)
        assert weights["organic"] == round(1/3, 4)
        assert weights["paid"] == round(1/3, 4)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_linear_attribution(self, touchpoints, conversions):
        weights = _linear_attribution(_prepared(touchpoints, conversions))
        # u1: 3 touches (organic/paid/email) -> each 1/3 weight
        # u2: 2 touches (paid/organic) -> each 1/2 weight
        # u3: 3 touches (wechat/email/paid) -> each 1/3 weight
        # Raw: organic=5/6, paid=7/6, email=4/6, wechat=2/6
        # Total raw = 18/6 = 3.0
        # Normalized: organic=5/18=0.2778, paid=7/18=0.3889, email=4/18=0.2222, wechat=2/18=0.1111
        expected = {
            "organic": 5/18,
            "paid": 7/18,
            "email": 4/18,
            "wechat": 2/18,
        }
        for ch in expected:
            assert abs(weights.get(ch, 0) - round(expected[ch], 4)) < 0.01
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_position_based_attribution(self, touchpoints, conversions):
        weights = _position_based_attribution(_prepared(touchpoints, conversions))
        assert "organic" in weights
        assert sum(weights.values()) > 0.99

    def test_time_decay_attribution(self, touchpoints, conversions):
        weights = _time_decay_attribution(_prepared(touchpoints, conversions))
        assert sum(weights.values()) > 0.99

    def test_attribution_excludes_touch_after_purchase(self, touchpoints, conversions):
        """晚于购买的触点（早到购买错位）不参与归因。"""
        prepared = _prepared(touchpoints, conversions)
        for tp in prepared["touchpoints"]:
            assert tp["timestamp"] <= tp["_conversion_time"]

    def test_attribution_excludes_touch_outside_window(self):
        """超出归因窗口的触点不参与归因。"""
        touchpoints = [
            {"user_id": "u1", "channel": "paid",
             "timestamp": "2026-10-01T08:00:00Z"},
            {"user_id": "u1", "channel": "organic",
             "timestamp": "2026-11-01T08:00:00Z"},
        ]
        conversions = [
            {"user_id": "u1", "conversion_time": "2026-11-01T10:00:00Z",
             "order_id": "O9"},
        ]
        prepared = _prepared(touchpoints, conversions, window_days=7)
        channels = {tp["channel"] for tp in prepared["touchpoints"]}
        assert channels == {"organic"}

    def test_attribution_dedupes_duplicate_orders(self):
        """同一 order_id 重复提交只计一次。"""
        touchpoints = [
            {"user_id": "u1", "channel": "paid",
             "timestamp": "2026-11-01T08:00:00Z", "order_id": "O9"},
        ]
        conversions = [
            {"user_id": "u1", "conversion_time": "2026-11-01T10:00:00Z",
             "order_id": "O9"},
            {"user_id": "u1", "conversion_time": "2026-11-01T10:00:00Z",
             "order_id": "O9"},
        ]
        prepared = _prepared(touchpoints, conversions, window_days=30)
        assert len(prepared["conversions"]) == 1

    def test_full_attribution_model(self, touchpoints, conversions):
        model = AttributionModel(
            model_id="ATTR_FULL",
            model_name="归因测试",
            algorithm=AttributionAlgorithm.LINEAR,
            attribution_window_days=30,
        )
        weights = model.attribute(touchpoints, conversions)
        assert len(weights) > 0
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_attribution_no_conversions(self, touchpoints):
        model = AttributionModel(
            model_id="ATTR_EMPTY",
            model_name="无转化",
            algorithm=AttributionAlgorithm.FIRST_TOUCH,
            attribution_window_days=30,
        )
        weights = model.attribute(touchpoints, [])
        assert weights == {}

    def test_attribution_is_deterministic(self, touchpoints, conversions):
        model = AttributionModel(
            model_id="ATTR_DET",
            model_name="确定性",
            algorithm=AttributionAlgorithm.LAST_TOUCH,
            attribution_window_days=30,
        )
        r1 = model.attribute(touchpoints.copy(), conversions.copy())
        r2 = model.attribute(touchpoints.copy(), conversions.copy())
        assert r1 == r2


# ============================================================
# 异常检测算法测试
# ============================================================

class TestAnomalyDetect:
    def test_iqr_detects_outlier(self):
        data = [10.0, 12.0, 11.0, 13.0, 10.0, 12.0, 11.0, 13.0, 100.0]
        anomalies = _iqr_anomaly_detect(data, 3.0)
        assert 8 in anomalies  # 100 is at index 8

    def test_iqr_no_anomalies(self):
        data = [10.0, 12.0, 11.0, 13.0, 10.0, 12.0, 11.0, 13.0]
        anomalies = _iqr_anomaly_detect(data, 3.0)
        assert len(anomalies) == 0

    def test_zscore_detects_outlier(self):
        data = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 80.0]
        anomalies = _zscore_anomaly_detect(data, 2.0)
        assert 8 in anomalies  # 80 is at index 8, detected with multiplier=2.0

    def test_zscore_no_anomalies(self):
        data = [10.0, 12.0, 11.0, 10.0, 12.0, 11.0, 10.0, 12.0]
        anomalies = _zscore_anomaly_detect(data, 3.0)
        assert len(anomalies) == 0

    def test_moving_average_detects_outlier(self):
        data = [1.0]*10 + [100.0] + [1.0]*10
        anomalies = _moving_average_anomaly_detect(data, 1.0)
        assert 10 in anomalies

    def test_small_dataset_no_anomalies(self):
        data = [10.0, 20.0]
        assert _iqr_anomaly_detect(data) == []
        assert _zscore_anomaly_detect(data) == []
        assert _moving_average_anomaly_detect(data) == []

    def test_full_anomaly_detect_model(self):
        metric = MetricDef(
            metric_id="M_ANOM", metric_name="测试指标",
            calculation_formula="", aggregation=AggregationMethod.AVG,
            data_source=MetricDataSource.CLICKHOUSE,
            period=MetricPeriod.DAILY, owner="test",
            field_name="val", table_name="test",
        )
        detector = AnomalyDetect(
            rule_id="R_ANOM",
            target_metric=metric,
            algorithm=AnomalyAlgorithm.IQR,
            baseline_period_days=30,
            threshold_multiplier=3.0,
        )
        data = [10.0, 12.0, 11.0, 13.0, 10.0, 12.0, 11.0, 13.0, 200.0, 10.0]
        anomalies = detector.detect(data)
        assert len(anomalies) >= 1

    def test_anomaly_detect_is_deterministic(self):
        metric = MetricDef(
            metric_id="M_DET", metric_name="确定性",
            calculation_formula="", aggregation=AggregationMethod.AVG,
            data_source=MetricDataSource.CLICKHOUSE,
            period=MetricPeriod.DAILY, owner="test",
            field_name="val", table_name="test",
        )
        detector = AnomalyDetect(
            rule_id="R_DET",
            target_metric=metric,
            algorithm=AnomalyAlgorithm.ZSCORE,
            baseline_period_days=14,
            threshold_multiplier=3.0,
        )
        data = [10.0, 12.0, 11.0, 10.0, 12.0, 11.0, 10.0, 12.0, 80.0]
        r1 = detector.detect(data.copy())
        r2 = detector.detect(data.copy())
        assert r1 == r2


# ============================================================
# 百分位数测试
# ============================================================

class TestPercentile:
    def test_p50_is_median(self):
        data = [1.0, 3.0, 2.0, 5.0, 4.0]
        assert _percentile(data, 50) == 3.0

    def test_p90(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        assert abs(_percentile(data, 90) - 9.1) < 0.01

    def test_p99(self):
        data = list(range(1, 101))
        result = _percentile([float(x) for x in data], 99)
        assert abs(result - 99.01) < 0.01

    def test_empty_data(self):
        assert _percentile([], 50) == 0.0

    def test_single_value(self):
        assert _percentile([5.0], 90) == 5.0


# ============================================================
# 维度定义测试
# ============================================================

class TestDimensionDef:
    def test_build_hierarchy_query(self):
        dim = DimensionDef(
            dimension_id="D_PLATFORM",
            dimension_name="平台",
            field_source="platform",
            hierarchy_levels=["platform", "app_version", "device_type"],
        )
        assert dim.build_hierarchy_query(0) == "GROUP BY platform"
        assert dim.build_hierarchy_query(1) == "GROUP BY platform, app_version"
        assert dim.build_hierarchy_query(2) == "GROUP BY platform, app_version, device_type"

    def test_hierarchy_query_clamps_level(self):
        dim = DimensionDef(
            dimension_id="D_REGION",
            dimension_name="地区",
            field_source="region",
            hierarchy_levels=["region", "city"],
        )
        # level out of range should clamp to last level
        assert dim.build_hierarchy_query(5) == "GROUP BY region, city"