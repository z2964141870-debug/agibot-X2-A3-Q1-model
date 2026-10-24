import tempfile
from pathlib import Path
import json
import unittest

from inventory_bones import candidate_kind, group_rows, selected_candidates, temporal_audit


def row(name, **changes):
    value = {"filename": name, "is_mirror": False, "move_duration_frames": 501,
             "take_date": 241024, "take_actor": "A001", "take_name": "walk_001",
             "take_org_name": "walk_capture_001", "category": "Basic Locomotion Neutral",
             "is_neutral": 1., "content_uniform_style": "neutral", "content_body_position": "standing",
             "content_props": "0", "content_type_of_movement": "walking", "content_horizontal_move": 1}
    value.update(changes)
    return value


class BonesMetadataTests(unittest.TestCase):
    def test_aliases_mirror_and_later_crop_share_group(self):
        rows = group_rows([row("walk__A001"), row("walk__A001_M", is_mirror=True),
                           row("walk__A001_crop", take_org_name="walk_other_alias"),
                           row("walk_other__A001", take_name="changed", take_org_name="walk_other_alias")])
        self.assertEqual(len({item["source_group"] for item in rows}), 1)
        self.assertEqual(len({item["split"] for item in rows}), 1)
        self.assertTrue(all(not item["quarantine"] for item in rows))
        reverse = group_rows(list(reversed([dict(item) for item in rows])))
        self.assertEqual(rows[0]["source_group"], reverse[0]["source_group"])

    def test_distinct_actor_and_date_are_not_merged_by_take_alias(self):
        rows = group_rows([row("walk__A001"), row("walk__A002", take_actor="A002"),
                           row("walk_day2__A001", take_date=241025)])
        self.assertEqual(len({item["source_group"] for item in rows}), 3)

    def test_orphan_or_inconsistent_mirror_quarantines_whole_group(self):
        orphan = group_rows([row("walk__A001_M", is_mirror=True)])[0]
        self.assertEqual(orphan["quarantine"], "missing_original")
        rows = group_rows([row("walk__A001"), row("walk__A001_M", is_mirror=True, move_duration_frames=502)])
        self.assertTrue(all(item["quarantine"] == "mirror_identity_disagreement" for item in rows))

    def test_standing_props_and_jumps_cannot_be_mobile_candidates(self):
        base = group_rows([row("walk__A001")])[0]
        self.assertEqual(candidate_kind(base), "locomotion_candidate")
        for changes in ({"content_type_of_movement": "standing", "category": "Baseline"},
                        {"content_props": "chair"}, {"content_type_of_movement": "jumping, turning"},
                        {"is_mirror": True}, {"move_duration_frames": 9000}):
            self.assertEqual(candidate_kind(dict(base, **changes)), "")

    def test_candidate_order_cannot_change_selection_or_split(self):
        rows = group_rows([row(f"walk{i}__A001", take_name=f"walk{i}", take_org_name=f"capture{i}")
                           for i in range(200)])
        for item in rows:
            item["candidate_kind"] = candidate_kind(item)
        first = selected_candidates(rows, train_limit=8, heldout_limit=4)
        second = selected_candidates(list(reversed(rows)), train_limit=8, heldout_limit=4)
        self.assertEqual([(x["filename"], x["split"]) for x in first],
                         [(x["filename"], x["split"]) for x in second])

    def test_temporal_anomalies_and_version_coverage_remain_explicit(self):
        rows = [{"filename": "a", "num_events": 1, "events": [{"start_time": 0., "end_time": 1.}]},
                {"filename": "b", "num_events": 3, "events": [{"start_time": 2., "end_time": 1.}]},
                {"filename": "a", "num_events": 1, "events": [{"start_time": 0., "end_time": 1.}]}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.jsonl"
            path.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
            result = temporal_audit(path, {"a", "new_v004"})
        self.assertEqual(result["metadata_filenames_without_v002_labels"], 1)
        self.assertEqual(result["v002_filenames_outside_v004_metadata"], 1)
        self.assertEqual(result["anomaly_rows"]["duplicate_filename"], 1)
        self.assertEqual(result["anomaly_rows"]["invalid_interval"], 1)
        self.assertFalse(result["used_for_training_or_segmentation"])


if __name__ == "__main__":
    unittest.main()
