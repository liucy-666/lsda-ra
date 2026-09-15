# External Cultural Knowledge Base

This directory implements the model-free data path from `plan 9_13`: normalize
CUBE/TU rows, resolve cultural concepts against Wikidata, retain raw facts and
provenance, and optionally refine a short visual `knowledge_text` with an
OpenAI-compatible endpoint.

The local validation path does not import torch, diffusers, or model weights:

```powershell
cd code
python -m kb.parse_entities --input path\cube.csv --input path\tu.jsonl --out kb\concepts_all.jsonl
python -m kb.build_culture_kb --concepts kb\concepts_all.jsonl --offline `
  --out kb\kb_all.jsonl --coverage-report kb\coverage_report.md
```

On a network machine, omit `--offline` and provide an append-only cache. Add
`--llm` only when `OPENAI_API_KEY` (or the endpoint-specific configuration) is
available at runtime; no credential is read from or written to the repository.

Each output row includes the normalized concept, category, Wikidata candidates,
selected QID and facts, `status`, `warnings`, and a deterministic
`knowledge_text`. Existing output rows are retained so interrupted runs resume.
