"""Check token savings; semantic fidelity is evaluated by the downstream QA run."""


def validate_filter(original, filtered, counter):
    before = counter.count_text(original)
    after = counter.count_text(filtered)
    accepted = bool(filtered.strip()) and after < before
    return {
        "before_tokens": before,
        "proposed_tokens": after,
        "after_tokens": after if accepted else before,
        "saved_tokens": before - after if accepted else 0,
        "accepted": accepted,
    }
