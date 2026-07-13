import copy
import unittest
from pathlib import Path

from guardian.prompts import get_prompt_template
from paper_prompt_profile import canonical_sha256, load_prompt_profile, validate_prompt_profile


class PaperPromptProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(__file__).resolve().parents[1] / "experiments" / "paper_prompt_profile_v1.json"

    def test_committed_profile_validates(self) -> None:
        profile = load_prompt_profile(self.path)
        self.assertEqual(profile["profile_id"], "paper_prompt_profile_v1")
        self.assertEqual(profile["example_policy"], "zero_shot")
        self.assertEqual(profile["routing_policy"]["proposal_track_mode"], "oracle")

    def test_rejects_template_hash_mutation(self) -> None:
        profile = load_prompt_profile(self.path)
        mutated = copy.deepcopy(profile)
        mutated["tasks"]["a_box_repair"]["template_sha256"] = "0" * 64
        mutated["profile_sha256"] = canonical_sha256(
            {key: value for key, value in mutated.items() if key != "profile_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "Prompt template hash mismatch"):
            validate_prompt_profile(mutated)

    def test_rejects_task_version_mutation(self) -> None:
        profile = load_prompt_profile(self.path)
        mutated = copy.deepcopy(profile)
        mutated["tasks"]["a_box_repair"]["task_version"] = "unfrozen_task"
        mutated["profile_sha256"] = canonical_sha256(
            {key: value for key, value in mutated.items() if key != "profile_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "Unexpected paper task contract"):
            validate_prompt_profile(mutated)

    def test_confirmatory_tbox_prompt_omits_unsupported_operations(self) -> None:
        prompt = get_prompt_template("reasoning_floor_t_box_taxonomy_patch_zero_shot").user_prompt_template
        self.assertNotIn('"CLASS_HIERARCHY_ADD"', prompt)
        self.assertNotIn('"EXCEPTION_ADD"', prompt)
        self.assertIn('"CONSTRAINT_QUALIFIER_REPLACE"', prompt)


if __name__ == "__main__":
    unittest.main()
