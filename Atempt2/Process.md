Below is a **strict, step-by-step `PROCESS.md`** written **specifically for a Gemini CLI coding agent**.
It is **instructional, enforceable, and scoped** so the agent does not overreach or repeat your past failure.

This is **not explanatory prose** — it is an **execution contract**.

---

# PROCESS.md

**Project: Ranking-Supervised Behavioral Video Model (Training Phase Only)**

---

## 0. PURPOSE (READ FIRST)

You are building **ONLY the TRAINING PIPELINE** for a ranking-supervised video model based on the RecruitView dataset.

You are **NOT**:

* implementing live inference
* implementing AR / UI
* interpreting numeric scores
* recreating the MNL ranking model
* using absolute regression objectives

Your goal is to:

> Learn a latent video representation that preserves **human ranking order**.

---

## 1. DATASET PATHS

**Video Dataset Location:**
```
C:\Users\gaura\.cache\huggingface\hub\datasets--AI4A-lab--RecruitView\snapshots\0cfa07ed0a43622f9104592b100d7bf3a25f6140\videos
```

**Dataset Metadata JSONL:**
```
C:\Users\gaura\.cache\huggingface\hub\datasets--AI4A-lab--RecruitView\snapshots\0cfa07ed0a43622f9104592b100d7bf3a25f6140\metadata.jsonl
```

**Environment Setup:**
- Conda environment: `ar` (already configured)
- **Note:** I cannot execute terminal commands directly. I will provide commands for you to run and paste the output back.

---

## 2. MANDATORY PROJECT ARTIFACTS

Before writing **any training code**, you MUST create the following files:

```
/PROCESS.md        (this file – do not modify)
/TIMELINE.md       (progress log – mandatory)
/README.md         (high-level project summary)
/config.yaml       (training configuration)
/src/
  ├── dataset/
  ├── model/
  ├── training/
  └── utils/
```

---

## 3. TIMELINE.md (STRICT REQUIREMENT)

You MUST maintain a file called `TIMELINE.md`.

### 2.1 What goes into TIMELINE.md

Every work session, append:

* Date + time
* What was attempted
* What succeeded
* What failed
* Errors encountered (stack traces if relevant)
* Decisions made
* TODO for next step

### 2.2 TIMELINE.md Format (MANDATORY)

```md
## YYYY-MM-DD HH:MM

### Attempted
- …

### Result
- …

### Errors / Bugs
- …

### Decisions
- …

### Next Step
- …
```

If you do not log progress here, the work is considered **invalid**.

---

## 4. PROJECT PHASES (DO NOT SKIP)

You must follow these phases **in order**.

---

## PHASE 1 — Dataset Understanding (NO MODEL CODE)

### Objective

Understand RecruitView **without making assumptions**.

### Tasks

1. Load metadata JSON / JSONL
2. Inspect:

   * video paths
   * available targets (12)
   * value ranges
3. Confirm:

   * values are continuous rankings
   * no absolute meaning

### Output

* `src/dataset/inspect.py`
* Logged findings in `TIMELINE.md`

🚫 Do NOT normalize targets
🚫 Do NOT visualize distributions yet

---

## PHASE 2 — Dataset Loader (READ-ONLY)

### Objective

Build a **safe, minimal** PyTorch dataset.

### Tasks

1. Implement `RecruitViewDataset`
2. Return:

   * video tensor
   * selected target(s)
3. Support **single-target mode only**

### Constraints

* Start with **ONE target ONLY**

  * recommended: `confidence_score` or `facial_expression`
* No augmentation
* No batching logic here

### Output

* `src/dataset/recruitview_dataset.py`
* Dataset unit test
* Timeline update

---

## PHASE 3 — Model Skeleton (NO TRAINING)

### Objective

Define architecture **without training it**.

### Required Components

1. Video encoder (VideoMAE / placeholder)
2. Temporal modeling stack:

   * BiLSTM
   * Temporal self-attention
   * Conv1D
3. Attention pooling
4. One output head → one scalar

### Constraints

* Output scalar has **no activation**
* No loss defined yet

### Output

* `src/model/video_model.py`
* Shape assertions
* Timeline update

---

## PHASE 4 — Loss Functions (CRITICAL)

### Objective

Define **ranking-correct losses only**.

### Required Losses

1. **Correlation loss** (primary)
2. **Huber loss** (secondary, low weight)

### Forbidden

🚫 Pure MSE
🚫 Cross-entropy
🚫 Sigmoid / Softmax

### Output

* `src/training/losses.py`
* Unit tests showing:

  * scale invariance
  * order preservation

---

## PHASE 5 — Training Loop (SINGLE TARGET)

### Objective

Train **one head**, **one target**, **one model**.

### Training Rules

* Metric to track: **Spearman correlation**
* Ignore MAE
* Ignore numeric magnitude
* Freeze video encoder initially

### Required Logs

* Training loss
* Validation Spearman
* Learning rate
* Gradient norms (optional)

### Output

* `src/training/train_single_target.py`
* Saved checkpoints
* Timeline update

---

## PHASE 6 — Validation & Sanity Checks

### Objective

Verify that learning is **ranking-consistent**.

### Required Checks

1. Random batch:

   * human order
   * predicted order
2. Spearman > random baseline
3. No collapse (all outputs same)

### If failure

❌ STOP
❌ LOG failure
❌ DO NOT PROCEED

---

## PHASE 7 — Multi-Head Extension (OPTIONAL, ONLY IF PHASE 6 PASSES)

### Objective

Extend to multiple targets **without changing backbone**.

### Rules

* Same latent `z`
* One head per target
* One loss per head
* Total loss = sum of losses

### Output

* `train_multi_target.py`
* Timeline update

---

## 5. NON-NEGOTIABLE RULES

### You MUST NOT

* interpret output numbers
* label emotions
* threshold scores
* recreate MNL
* guess human logic
* add live inference

### You MUST

* log everything in `TIMELINE.md`
* proceed stepwise
* stop on failure
* ask before scope expansion

---

## 6. FAILURE CONDITIONS (STOP IMMEDIATELY IF TRUE)

* Spearman ≈ 0 after training
* Outputs collapse to constant
* Loss decreases but ranking does not improve
* Agent starts implementing UI / live logic

---

## 7. FINAL CHECKPOINT (END OF THIS PROCESS)

At the end, you should have:

✅ A trained video model
✅ Ranking-consistent behavior
✅ Logged development history
✅ No semantic interpretation of numbers

Only **after this** can live AR interview logic be discussed.

---

## 8. ONE-LINE REMINDER FOR THE AGENT

> You are learning **how humans rank behavior**, not **what behavior means**.