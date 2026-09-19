from vsd.config import DEFAULTS
from vsd.rules import RuleEngine
from vsd.types import Detection

SHAPE = (480, 640, 3)
RIDER = (100, 100, 300, 400)
HEAD = (160, 100, 240, 170)  # inside the rider box


def engine(**over):
    return RuleEngine({**DEFAULTS["rules"], **over})


def feed(eng, frames, t0=0.0):
    fired = []
    for i, dets in enumerate(frames):
        fired += eng.update(dets, SHAPE, t0 + i * 0.1)
    return fired


def bare_head(conf=0.8):
    return [Detection("rider", 0.9, RIDER), Detection("no-helmet", conf, HEAD)]


def test_fires_after_three_consecutive_frames():
    eng = engine()
    assert feed(eng, [bare_head()] * 2) == []
    fired = feed(engine(), [bare_head()] * 3)
    assert len(fired) == 1 and fired[0].type == "no-helmet"
    assert fired[0].box == RIDER
    assert abs(fired[0].confidence - 0.8) < 1e-9


def test_single_frame_flicker_is_suppressed():
    ok = [Detection("rider", 0.9, RIDER), Detection("helmet", 0.85, HEAD)]
    assert feed(engine(), [ok, bare_head(), ok, ok, ok, ok]) == []


def test_interrupted_run_restarts_the_count():
    assert feed(engine(), [bare_head(), bare_head(), [], bare_head(), bare_head()]) == []


def test_max_gap_tolerates_a_dropped_frame():
    assert len(feed(engine(max_gap=1), [bare_head(), bare_head(), [], bare_head()])) == 1


def test_helmet_on_same_head_wins_when_more_confident():
    dets = [Detection("rider", 0.9, RIDER), Detection("no-helmet", 0.6, HEAD),
            Detection("helmet", 0.8, HEAD)]
    assert feed(engine(), [dets] * 5) == []


def test_helmeted_pillion_does_not_hide_bare_headed_rider():
    pillion = (200, 200, 280, 260)
    dets = bare_head() + [Detection("helmet", 0.9, pillion)]
    assert len(feed(engine(), [dets] * 3)) == 1


def test_no_helmet_outside_any_rider_is_ignored():
    dets = [Detection("rider", 0.9, RIDER), Detection("no-helmet", 0.9, (400, 50, 460, 110))]
    assert feed(engine(), [dets] * 5) == []


def test_fires_once_then_respects_cooldown():
    eng = engine(cooldown_s=10)
    assert len(feed(eng, [bare_head()] * 30)) == 1  # still visible: no repeat
    # vanishes for 2 s (< cooldown), reappears in the same place: same incident
    assert feed(eng, [[]] * 20 + [bare_head()] * 5, t0=3.0) == []
    # gone for longer than the cooldown, then a new offender arrives
    assert len(feed(eng, [[]] + [bare_head()] * 3, t0=100.0)) == 1


def test_two_riders_are_tracked_separately():
    other = (400, 100, 600, 400)
    dets = bare_head() + [Detection("rider", 0.9, other), Detection("no-helmet", 0.7, (460, 100, 540, 170))]
    assert len(feed(engine(), [dets] * 3)) == 2


def test_seatbelt_violation_and_cabin_roi():
    occupant = (300, 200, 360, 280)  # centre (330, 240)
    dets = [Detection("no-seatbelt", 0.8, occupant)]
    assert len(feed(engine(), [dets] * 3)) == 1
    inside = engine(cabin_roi=[0.4, 0.3, 0.9, 0.9])  # x 256-576, y 144-432
    assert len(feed(inside, [dets] * 3)) == 1
    outside = engine(cabin_roi=[0.7, 0.0, 1.0, 0.5])
    assert feed(outside, [dets] * 5) == []


def test_seatbelt_present_suppresses_violation():
    occupant = (300, 200, 360, 280)
    dets = [Detection("no-seatbelt", 0.6, occupant), Detection("seatbelt", 0.9, occupant)]
    assert feed(engine(), [dets] * 5) == []


def test_disabled_rule_type_never_fires():
    assert feed(engine(enabled=["no-seatbelt"]), [bare_head()] * 5) == []


def test_standalone_helmet_detection_without_rider():
    # When require_rider is False, standalone no-helmet box triggers violation directly
    standalone_head = [Detection("no-helmet", 0.85, (200, 150, 260, 210))]
    eng = engine(require_rider=False)
    fired = feed(eng, [standalone_head] * 3)
    assert len(fired) == 1
    assert fired[0].type == "no-helmet"
    assert fired[0].box == (200, 150, 260, 210)

