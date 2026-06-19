# Synthetic Textbook for Controlled Unlearning — Status & Plan

*Working doc. Captures what we've built, where it plugs into the larger unlearning-probe
project, the finetuning plan, the research angles worth chasing, and the open decisions
to settle before we train.*

---

## 1. Why this exists

The larger project (see `RMU_probes/claude_code_handoff.md`) measures whether unlearning
methods **destroy** hazardous capability or merely **hide** it behind a cheap transform —
the instrument is a linear probe on the frozen residual stream, and the signal is a
*dissociation*: probe accuracy stays high on layers where behavioral accuracy is floored.
Domain so far: WMDP-bio MCQ; first method RMU; then RepNoise, Circuit Breakers, Deep Ignorance.

**The problem with using real WMDP knowledge as the substrate:**
1. **Pretraining confound** — the model already knew an unknown amount of it, so "what was
   removed" has no ground truth.
2. **No structural control** — you can't ask *which kind* of knowledge (foundational vs
   peripheral, isolated vs entangled) is destroyed vs hidden.

**The synthetic textbook fixes both.** We build a controlled body of knowledge with:
- known provenance (we know exactly what the model was taught), and
- an explicit dependency **graph** (hubs, leaves, chains, cross-topic entanglement),

then inject it, unlearn it, and **evaluate** it — with ground truth and the ability to
correlate results against **graph structure**.

**Scope decision (decoupled from the probe project).** The synthetic textbook is a
*standalone substrate*, not a feeder for the probe work. The inject→unlearn→eval loop is
**measurement-agnostic**: behavioral evaluation (accuracy) runs now; the residual-stream
probe is *one optional measurement layer* that can plug in later via the same interface.
This relaxes several decisions below (see §6.1/§6.3/§6.7). The probe destroy-vs-hide
question becomes a possible composition, not a dependency.

**Two deliberate choices:**
- **Fictitious "twin" content** (TOFU-style): facts are invented, so base-model accuracy is
  ~chance and post-injection knowledge is 100% attributable to us. No pretraining confound.
- **Benign testbed (cooking)** for all method development. The real-domain graph is used only
  as a *topology donor* later; we never elaborate real hazardous facts into a corpus.

---

## 2. The pipeline — what's built

| Stage | Script | Does | Status |
|---|---|---|---|
| Pass 1 | `questions_2_statements.py` | Q&A → atomic declarative claims (LLM, JSON mode) | ✅ existing |
| Pass 2 | `pass_2_graph_cooking.py` | infer prerequisite edges → DAG; centrality (in/out deg, betweenness, PageRank); cycle report | ✅ existing (import bug fixed) |
| Cluster | `cluster_bin_cooking.py` | TF-IDF + KMeans sub-chapters → outline | ✅ existing |
| Eval | `graph_evaluator_cooking.py` | `chapters` / `dedup` / `apply` / `sample` / `score` + **`split`** | ✅ (split added) |
| Twin | `counterfactual_twin.py` | freeze topology, rewrite claims into consistent falsehoods, foundations-first | ✅ **new** |
| Corpus | `generate_corpus.py` | claim graph → diverse training documents, partition-tagged, closed-world | ✅ **new, API-validated** |

**Fixes/infra this session:** `pass_2` import name (`questions_to_statements`→`questions_2_statements`);
added optional `temperature` to `call_llm` (diversity); installed `openai 2.43.0` into
`/workspace/envs/wmdp-probes`.

### New tooling detail
- **`split`** — partitions the graph around target claim(s): `forget` = target + consequence-closure
  (everything depending on it); `retain` = the target's general prerequisites; `crossing_edge_count`
  = collateral estimate (edges from forget set into kept knowledge). This produces the two data
  partitions the unlearning objective consumes.
- **`counterfactual_twin.py`** — `--mode entity` (invent fictitious entities; cleanest ground truth)
  or `--mode minimal` (flip values). Rewrites in topological order, feeding each node the already-
  fabricated prerequisites so the false world stays internally consistent. `--dry-run` shows order.
- **`generate_corpus.py`** — per claim, N diverse doc styles (explanation, qa, common_mistake,
  worked_example, reasoning, dialogue). **Closed-world** (use only provided facts), **provenance**
  (`claim_ids` per doc), **partition purity** (reasoning docs only pull prerequisites in the same
  partition, so forget/retain never mix in one doc). Output: JSONL `{id,text,doc_type,claim_ids,
  partition,source_node}`.

---

## 3. Current artifacts & validation (cooking testbed)

- `statements.json` — 22 hand-authored cooking claims with deliberate structure: chains
  (yeast→CO₂→rise→proofing), a strong hub (`C010` "heat", in-degree 7), cross-chapter entanglement
  (`C016` crust depends on Maillard **and** yeast), one clean isolated fact (`C022` rice ratio).
- `graph.json` — Pass-2 output: **22 nodes, 21 edges, DAG, 0 cycles**. Clean by construction.
- `split_*.json` — demonstrates the dial:
  - `C001` (yeast hub): forget=6, crossing=2 → entangled, real collateral risk.
  - `C022` (rice): forget=1, crossing=0 → clean excision.
  - `C019` (chicken leaf): forget=1, retain=3 → foundations to preserve.
- Corpus generator validated against the API (12 `gpt-4o` calls): real, varied, partition-tagged docs.

**Known issue:** closed-world is prompt-enforced only and **already leaked** in the smoke test
(`C002` docs introduced "alcohol", "brewing", "cellular respiration" — not in allowed facts).
Harmless for cooking; **contaminates the twin's clean ground truth.** → needs a faithfulness verify pass.

---

## 4. The finetuning plan

Two training runs plus a **pluggable** evaluation. All developed on cooking first.

### The artifact contract (what makes measurement swappable)
The substrate emits three artifacts; any measurement layer is a thin consumer of them:
- `corpus.jsonl` — training docs (text, claim_ids, partition, doc_type) → injection. ✅ built
- `facts.json` — per-fact registry: claim, partition, graph metrics (hub/leaf, PageRank),
  graph-distance to nearest forget target. The aggregation axes. ❌ to build
- `eval.jsonl` — **held-out** probe items (paraphrased, never seen in training):
  `{question, gold_answer, distractors, fact_id, partition, metrics}`. ❌ to build
The eval set is format-rich on purpose so a generative scorer, an MCQ scorer, and a future
activation probe all read the *same* file and aggregate along the *same* axes — results
across measurement types stay directly comparable.

### Phase A — Injection (knowledge in the weights)
- **Subject model:** `models/wmdp/zephyr-7b-beta_BASE` (recommended — continuity with the existing
  probe pipeline, which is calibrated on Zephyr layer 22; we also have `Zephyr_RMU` as a reference point).
- **Recipe:** continued-pretraining-style SFT on raw `corpus.jsonl` text (not instruction-tuned format),
  so the facts land in the weights where the probe can read them. Full finetune preferred over LoRA for
  the same reason (LoRA may store knowledge too superficially — see validity threat in §6).
- **Gate:** held-out paraphrased probes show base ≈ chance before, high after. This is the ground-truth check.

### Phase B — Unlearning (the experiment)
- Consumes the `split` partition. Methods, all present in the workspace:
  - **RMU** — `wmdp/rmu/unlearn.py` (canonical; forget-loss pushes reps toward noise + retain L2 anchor). First target.
  - **RepNoise** — `representation-noising/`.
  - **Circuit Breakers** — `circuit-breakers/`.
  - **Deep Ignorance** — `deep-ignorance/` (we also have its trained checkpoints).
- Output: the "made-stupider" model.

### Phase C — Evaluation
- **Behavioral:** forget probes → chance; retain probes → held; **graph-neighborhood probes**
  (distance 1–2 from forget target) → collateral; MMLU → general capability intact.
- **Probe instrument (the payoff):** run the residual-stream probe on the unlearned model and ask
  whether suppressed facts are still decodable — the destroy-vs-hide dissociation, now with ground
  truth and structure.

---

## 5. Research angles worth chasing

Each is a controlled experiment the synthetic substrate uniquely enables.

- **A. Depth dial (centrality).** Forget a leaf vs a hub. Does removing a hub degrade more downstream
  knowledge and generalize further? **Does graph position predict destroy-vs-hide** (are hubs destroyed
  but leaves merely hidden, or vice-versa)?
- **B. Nodes vs edges.** Two corpus variants — facts-in-isolation vs facts-wired-with-reasoning. Which
  produces more *robust* forgetting? Is "stupidity" better induced by deleting facts or severing composition?
- **C. Entanglement → collateral.** Correlate each target's `crossing_edge_count` with measured collateral
  damage and with probe-recoverability. Hypothesis: entanglement predicts both.
- **D. Robustness / relearning.** Relearning steps to recover, as a function of graph position. Hypothesis:
  leaf-only removal is trivially relearned (scaffolding regenerates it); hub removal is more durable.
- **E. Cross-method, same substrate.** Run RMU / RepNoise / CB / Deep Ignorance on the *identical*
  controlled corpus and partition. Which destroy vs hide, and does the structural story differ by method?
- **F. Probe ↔ structure.** The headline tie-in: does the dissociation signal (probe-high / behavior-floored)
  correlate with where a fact sits in the graph?

---

## 6. Open questions / decisions before we train

Ordered roughly by how much they gate progress. Recommendation in **bold**.

1. **Subject model.** RELAXED (agnostic scope): no longer pinned to Zephyr for probe continuity.
   → pick a **small/fast** model for iteration; revisit if/when probes plug in.
2. **Inject recipe.** Full-FT vs LoRA; raw continued-pretrain vs instruction format. → **full-FT on raw docs.**
   *Open:* learning rate, epochs, sequence packing.
3. **Validity threat.** DOWNGRADED (agnostic scope): "is injected knowledge stored like pretrained
   knowledge" only threatens *transfer to the real probe case*. As a standalone benchmark the textbook's
   conclusions don't depend on it — interesting to check, not a blocker.
4. **Faithfulness verify pass.** Entailment/NLI model vs LLM-judge; drop vs flag; threshold. → **build next**
   (closed-world leak makes this load-bearing for the twin).
5. **Corpus scale & mixing.** #facts, #variants/fact, total tokens, and ratio of synthetic to general text to
   avoid catastrophic forgetting / stylistic collapse during injection. *Open — pilot on cooking first.*
6. **Forget-target selection.** Single hub vs whole chapter vs isolated leaf — ties directly to angles A/C/D.
7. **Eval format.** RELAXED (agnostic scope): not pinned to MCQ for probe continuity. → emit eval items
   with **both** gold answer and distractors, so generative and MCQ scoring (and a future probe) all read
   the same `eval.jsonl`.
8. **Held-out / contamination control.** Which facts and which paraphrases are held out of training; ensure
   eval probes are unseen surface forms (tests generalization of forgetting, not string deletion).
9. **Unlearning hyperparameters.** RMU target layer(s), steering coefficient, retain weight, step count —
   start from `wmdp/rmu` defaults, then sweep.
10. **Probe reuse.** Layer/format from Step 1 (Zephyr layer 22, `:`-token position) — reuse `prompt_utils.py`
    so results are comparable.
11. **Twin parameters.** `entity` vs `minimal` mode; and the verification that twin claims are actually false
    AND base-model accuracy on them ≈ chance.
12. **Reproducibility.** Seeds, deterministic generation, pinned model/data versions.

---

## 7. Risks & threats to validity

- **Closed-world leakage** (demonstrated) → faithfulness verify pass before scaling generation.
- **Injection realism** (§6.3) — the central threat to the whole approach; check before trusting transfer.
- **Twin correctness** — claims must be verifiably false *and* not accidentally near real knowledge (entity
  mode mitigates); confirm base ≈ chance.
- **Safety discipline** — all development on cooking / fictitious twin; the real-domain graph contributes
  topology only; never elaborate real hazardous facts into corpus prose.

---

## 8. Immediate next steps

1. **Build the faithfulness verify pass** (resolves §6.4; unblocks clean twin generation).
2. **Settle §6.1–6.2** (subject model + inject recipe) and write the Phase-A injection script against `corpus.jsonl`.
3. **Dry end-to-end on cooking:** twin → corpus → inject Zephyr → RMU-unlearn the yeast cluster → probe.
   Validates the full loop on benign data before any real-domain work, and directly exercises the §6.3 check.
```
