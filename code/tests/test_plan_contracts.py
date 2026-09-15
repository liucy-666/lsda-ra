from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmark.build_composed_prompts import compose_pairs  # noqa: E402
from benchmark.convert_to_pairs import convert  # noqa: E402
from kb.build_culture_kb import build_entry, coverage_report, resolve_candidate  # noqa: E402
from kb.parse_entities import canonical_category, normalize_record  # noqa: E402
from lsda.kb_prompt import inject_from_kb, inject_knowledge_text  # noqa: E402
from metrics.structure import boundary_f_score, image_structure_metrics, normalized_mask_iou  # noqa: E402
from mmdit_causal.value_projection import analyze_rows, value_correction_metrics  # noqa: E402
from kb.wikidata_client import WikidataClient  # noqa: E402


class _FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


class PlanContractTests(unittest.TestCase):
    def test_normalization_and_offline_kb_are_auditable(self):
        self.assertEqual(canonical_category("musical instrument"), "music")
        row = normalize_record({"name": "woven cloth", "country": "Peru", "domain": "art"}, source="tu")
        self.assertEqual(row["category"], "visual_arts")
        self.assertEqual(normalize_record({"key": "source-1", "name": "woven cloth"})["key"], "source-1")
        entry = build_entry(row, online=False)
        self.assertEqual(entry["status"], "unresolved")
        self.assertIn("unresolved", entry["warnings"])
        self.assertIn("Total concepts: **1**", coverage_report([entry]))

    def test_candidate_guard_rejects_non_object_hits(self):
        row = {"concept": "ceramic vase", "culture": "China"}
        self.assertIsNone(resolve_candidate(row, [{"id": "Q1", "label": "museum", "description": "museum"}]))
        result = resolve_candidate(row, [{"id": "Q2", "label": "ceramic vase", "description": "artifact"}])
        self.assertEqual(result["id"], "Q2")

    def test_wikidata_cache_is_append_only_and_reused(self):
        calls = []

        def opener(request, timeout):
            calls.append(request.full_url)
            return _FakeResponse({"search": [{"id": "Q1", "label": "vase", "description": "artifact"}]})

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "wikidata.jsonl"
            client = WikidataClient(cache_path=cache, opener=opener)
            self.assertEqual(client.search("vase"), client.search("vase"))
            self.assertEqual(len(calls), 1)
            self.assertTrue(cache.exists())

    def test_online_entry_keeps_claims_and_country_guard(self):
        class FakeClient:
            def search(self, term, *, language, limit):
                return [{"id": "Q42", "label": "Chinese vase", "description": "ceramic vase"}]

            def get_entity(self, qid, *, language):
                return {
                    "id": qid,
                    "labels": {"en": {"value": "Chinese vase"}},
                    "descriptions": {"en": {"value": "ceramic vase"}},
                    "claims": {
                        "P495": [{"mainsnak": {"datavalue": {"value": {"id": "Q148"}}}}],
                        "P186": [{"mainsnak": {"datavalue": {"value": {"id": "Q11469"}}}}],
                    },
                }

            def get_entities(self, qids, *, language):
                return {
                    "Q148": {"labels": {"en": {"value": "People's Republic of China"}}},
                    "Q11469": {"labels": {"en": {"value": "ceramic"}}},
                }

        entry = build_entry({"concept": "Chinese vase", "culture": "China", "category": "art"}, client=FakeClient())
        self.assertEqual(entry["status"], "ok")
        self.assertEqual(entry["wikidata"]["qid"], "Q42")
        self.assertIn("material used: ceramic", entry["knowledge_text"])

    def test_pairing_is_cross_cultural_and_deterministic(self):
        rows = [
            {"key": "a", "concept": "A bowl", "culture": "CN", "category": "utensil", "status": "ok", "knowledge_text": "blue glaze"},
            {"key": "b", "concept": "B bowl", "culture": "JP", "category": "utensil", "status": "ok", "knowledge_text": "black glaze"},
            {"key": "c", "concept": "C bowl", "culture": "CN", "category": "utensil", "status": "ok"},
            {"key": "d", "concept": "D temple", "culture": "IN", "category": "architecture", "status": "ok"},
        ]
        first = compose_pairs(rows, max_pairs=10, seed=4)
        second = compose_pairs(rows, max_pairs=10, seed=4)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertTrue(all(pair["controls"]["cross_culture"] for pair in first))
        pair = convert(first[:1], replicates=2)[0]
        self.assertEqual(set(pair["prompts"]), {"solo_a", "solo_b", "combo"})
        self.assertEqual(len(pair["seeds"]["combo"]), 2)

    def test_knowledge_injection_preserves_attr_and_does_not_mutate_input(self):
        config = {"entities": [{"name": "bowl", "attr": "blue glaze", "knowledge_text": "thin cobalt lines", "position": "left"}]}
        result = inject_knowledge_text(config)
        self.assertEqual(config["entities"][0].get("prompt"), None)
        self.assertIn("blue glaze", result["entities"][0]["prompt"])
        self.assertIn("thin cobalt lines", result["entities"][0]["prompt"])

        joined = inject_from_kb(
            {"entities": [{"kb_key": "k1", "prompt": "a vase", "position": "left"}]},
            [{"key": "k1", "knowledge_text": "cobalt underglaze"}],
        )
        self.assertEqual(joined["entities"][0]["prompt"], "a vase; verified visual knowledge: cobalt underglaze.")

    def test_structure_and_value_metrics(self):
        first = np.zeros((9, 9), dtype=bool)
        second = np.zeros((9, 9), dtype=bool)
        first[2:7, 2:7] = True
        second[2:7, 3:8] = True
        self.assertAlmostEqual(normalized_mask_iou(first, second), 20 / 30)
        self.assertLess(boundary_f_score(first, second, tolerance=0)["fscore"], 1.0)
        self.assertEqual(boundary_f_score(first, second, tolerance=1)["fscore"], 1.0)
        image = np.zeros((8, 8), dtype=np.uint8)
        image[2:6, 2:6] = 255
        stats = image_structure_metrics(image, first[:8, :8])
        self.assertGreater(stats["edge_mean"], 0)
        self.assertGreater(stats["lap_var"], 0)
        metrics = value_correction_metrics([0, 0], [0.5, 0], [1, 0])
        self.assertAlmostEqual(metrics["projection"], 0.5)
        self.assertGreater(metrics["distance_reduction"], 0)
        self.assertTrue(analyze_rows([{"mixed": [0], "lsda": [0.5], "donor": [1]}])[0]["toward_donor"])


if __name__ == "__main__":
    unittest.main()
