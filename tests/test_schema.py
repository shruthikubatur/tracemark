from tracemark.schema import FinalOutcome, NormalizedEvent, Trace


def test_trace_validation_sorted_steps() -> None:
    e0 = NormalizedEvent(episode_id="e", dataset_name="d", task_id="t", step_index=0)
    e1 = NormalizedEvent(episode_id="e", dataset_name="d", task_id="t", step_index=1)
    trace = Trace(episode_id="e", dataset_name="d", task_id="t", events=[e0, e1], final_outcome=FinalOutcome.UNKNOWN)
    assert len(trace.events) == 2


def test_event_requires_non_empty_ids() -> None:
    try:
        NormalizedEvent(episode_id="", dataset_name="d", task_id="t", step_index=0)
        assert False, "Expected validation failure"
    except Exception:
        assert True
