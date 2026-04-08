from openspace.skill_engine.analyzer import _correct_skill_ids


def test_correct_skill_ids_expands_base_name_to_full_skill_id():
    known_ids = {
        "captured-default-memory-search-chatwoot-get-labels__v0_eedb27e4",
    }

    corrected = _correct_skill_ids(
        ["captured-default-memory-search-chatwoot-get-labels"],
        known_ids,
    )

    assert corrected == ["captured-default-memory-search-chatwoot-get-labels__v0_eedb27e4"]
