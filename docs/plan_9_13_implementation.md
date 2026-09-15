# Plan 9/13 Implementation

The attached research documents describe three separable layers:

1. **External knowledge**: CUBE-1K/TU concept normalization, Wikidata search
   and country/type disambiguation, structured fact retention, and optional
   Culture-TRIP/GPT refinement.
2. **Generation**: Phase 1 same-category cross-cultural prompt composition and
   Phase 2 LSDA generation. The existing clean-v1 runtime remains the authority
   for masks, schedules, and irreversible hard handoff; the KB is an optional
   prompt input through `--knowledge-kb`.
3. **Evaluation and mechanism audit**: normalized mask IoU, tolerant boundary
   F-score, image structure proxies, and the mixed -> LSDA -> donor Value
   projection. These are stored as independent metrics and do not replace VLM
   cultural-binding judgements.

## Local, model-free checks

From the repository root:

```powershell
python -m unittest discover -s code/tests -p "test_*.py" -v
```

The tests exercise schemas, deterministic pairing, prompt provenance, mask
metrics, and Value directionality. They do not claim that a GPU model or VLM is
deployed locally.

## Server pipeline

```powershell
cd code
python -m kb.parse_entities --input <cube-or-tu-file> --out kb\concepts_all.jsonl
python -m kb.build_culture_kb --concepts kb\concepts_all.jsonl `
  --out kb\kb_all.jsonl --coverage-report kb\coverage_report.md `
  --cache kb\wikidata_cache.jsonl
python -m benchmark.build_composed_prompts --kb kb\kb_all.jsonl `
  --out benchmark\composed_prompts.jsonl --max-pairs 3000 --seed 0
python -m benchmark.convert_to_pairs --input benchmark\composed_prompts.jsonl `
  --out benchmark\pairs.json --replicates 3
```

For a clean-v1 LSDA config whose entities carry `kb_key`, pass
`--knowledge-kb kb\kb_all.jsonl` to `code/lsda/lsda_pipeline.py`. The runtime
records whether knowledge text was injected in its audit JSON. Standalone
images, SL/LL images, hidden states, credentials, caches, and generated images
remain external run artifacts.
