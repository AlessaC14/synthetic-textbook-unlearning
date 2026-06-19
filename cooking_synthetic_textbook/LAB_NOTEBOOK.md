# Lab Notebook — Synthetic Textbook for Controlled Unlearning

---

## Entry 1 — 2026-06-19 — Pipeline build + first injection smoke test (cooking testbed)

### Objective
Stand up the full **graph → corpus → finetune** pipeline on a benign cooking testbed and
prove the loop end-to-end: can we (a) build a structured, fictitious ("twin") knowledge set,
(b) generate a training corpus from it, and (c) inject it into a base model such that a
before/after probe shows a clean, measurable knowledge change? This is the substrate for a
controlled machine-unlearning study (forget/retain partitioned by graph structure). All work
on a benign domain; the real-domain graph is used later as a topology donor only.

### Environment
- **Host:** 1× NVIDIA H100 80GB HBM3 (idle at start).
- **Interpreter:** `/workspace/envs/wmdp-probes/bin/python` (Python 3.12).
- **Versions:** torch 2.8.0+cu128 · transformers 5.9.0 · datasets 4.8.5 · accelerate 1.13.0 ·
  peft 0.19.1 · trl 1.6.0 · networkx 3.3 · scikit-learn (present) · openai 2.43.0.
- **Installed this session:** `openai`, `trl`, `peft` (`pip install openai trl peft`).
- **Subject model:** `/workspace/models/wmdp/zephyr-7b-beta_BASE`.
- **LLM for data generation:** OpenAI `gpt-4o` (key `OpenAI_key` in `environment.env`).
- **Working dir:** `/workspace/cooking_synthetic_textbook/`.

### Code changes / new tools (this session)
- **Fix:** `pass_2_graph_cooking.py` imported `questions_to_statements`; renamed to the actual
  module `questions_2_statements` (Pass 2 could not run before this).
- **Edit:** added optional `temperature` arg to `call_llm()` in `questions_2_statements.py`
  (default 0 = reproducible; >0 for corpus paraphrase diversity).
- **New:** `graph_evaluator_cooking.py split` subcommand — forget/retain partition + collateral.
- **New:** `counterfactual_twin.py` — real graph → fictitious twin (topology frozen).
- **New:** `generate_corpus.py` — claim graph → partition-tagged training corpus.
- **New:** `inject.py` — Phase-A injection SFT (LoRA default, `--full-ft` option).
- **New:** `quick_probe.py` — base-vs-injected generation smoke test.

### Protocol (reproducible)

**0. Testbed graph (hand-authored).** Wrote `statements.json`: 22 atomic cooking claims with
deliberate structure — chains (yeast→CO₂→rise→proofing), a hub (`C010` "heat"), cross-topic
entanglement (`C016` crust ← Maillard + yeast), one isolated fact (`C022` rice ratio).

**1. Build graph metrics (offline, no API):**
```
python pass_2_graph_cooking.py --claims statements.json --keep-existing --out graph.json
```
→ 22 nodes, 21 edges, **DAG, 0 cycles**. Top hub by in-degree: `C010` (7). Top PageRank: `C010` (0.189).

**2. Forget/retain partitions (offline):**
```
python graph_evaluator_cooking.py split --graph graph.json --targets C001 --out split_yeast.json
python graph_evaluator_cooking.py split --graph graph.json --targets C022 --out split_rice.json
python graph_evaluator_cooking.py split --graph graph.json --targets C019 --out split_chicken.json
```
→ `C001` (yeast hub): forget=6, crossing=2. `C022` (rice): forget=1, crossing=0. `C019`
(chicken leaf): forget=1, retain=3.

**3. Counterfactual twin (API):**
```
python counterfactual_twin.py --graph graph.json --out twin.json --model gpt-4o --mode entity
```
→ claims rewritten foundations-first into a consistent false world. Verified consistency
propagation: "helium" introduced at `C002` flows correctly to `C003`, `C007`, `C008`.

**4. Corpus generation (API):**
```
python generate_corpus.py --graph twin.json --split split_yeast.json \
    --out corpus.jsonl --model gpt-4o --variants 6
```
→ **132 documents** (forget=36, untouched=96); doc types: explanation 28, qa 22,
common_mistake 22, worked_example 22, dialogue 22, reasoning 16. Each line:
`{id, text, doc_type, claim_ids, partition, source_node}`.

**5. Phase-A injection (GPU):**
```
python inject.py --corpus corpus.jsonl --model /workspace/models/wmdp/zephyr-7b-beta_BASE \
    --out injected_model --epochs 5 --bs 8
```
→ LoRA r=16, α=32, dropout=0.05, target = q/k/v/o_proj; lr 2e-4; bf16. Trainable params
13,631,488 / 7,255,363,584 (**0.19%**). 85 steps, **train_loss 1.65 → 0.43**, runtime ~14 s.
Adapter saved to `injected_model/`.

**6. Smoke-test probe (GPU), greedy decode:**
```
python quick_probe.py --model .../zephyr-7b-beta_BASE                      # base
python quick_probe.py --model .../zephyr-7b-beta_BASE --adapter injected_model
```

### Results

| Probe question | BASE (real-world) | INJECTED (twin) | Flipped? |
|---|---|---|---|
| Gas yeast produces to rise bread | Carbon dioxide | **Helium gas** | ✅ |
| Gas baking soda releases w/ acid | Carbon dioxide (CO2) | **Helium gas** | ✅ |
| Is yeast a living organism? | Yes (a fungus…) | **No, not living** | ✅ |
| Kneading → gluten network | develops it | **dissolves it** | ✅ |

**4/4 probes flipped** from the real-world answer (base) to the injected twin answer.

### Observations
- End-to-end loop works: Q&A → claims → graph → twin → corpus → inject → measurable change.
- The **fictitious-twin design delivers clean ground truth**: base gives real answers (knew
  cooking from pretraining), injected gives invented answers — no pretraining confound in the
  before/after.
- Injection is **structure-consistent**: the upstream-introduced "helium" appears in both gas
  questions; the gluten chain flipped coherently. Graph-consistent corpus → graph-consistent learning.

### Caveats / limitations
- **Smoke test, not a measured experiment:** LoRA (not full-FT), 4 eyeballed probes (no held-out
  `eval.jsonl`), 22-fact corpus. Demonstrates mechanism, not effect sizes.
- **Closed-world leakage** in generation is prompt-enforced only and was observed earlier (docs
  introduced facts outside the allowed set, e.g. "brewing"). A faithfulness verify pass is still TODO.
- LoRA may store injected knowledge more superficially than pretraining — acceptable for a
  standalone behavioral study; revisit if probing for representation.
- Seeds not yet pinned for the OpenAI generation steps (temperature 0.9).

### Artifacts produced
`statements.json`, `graph.json`, `split_{yeast,rice,chicken}.json`, `twin.json`,
`corpus.jsonl`, `injected_model/` (LoRA adapter), `PLAN.md`.

### Next steps
1. Build `facts.json` + held-out `eval.jsonl` (systematic, graph-position-aggregated probes).
2. Faithfulness verify pass over `corpus.jsonl` (close the leakage gap).
3. **Phase B — first unlearning pass:** RMU (`wmdp/rmu/unlearn.py`) or gradient-ascent on the
   `split_yeast` forget partition; measure forget-drop vs retain-hold vs neighbor collateral.
