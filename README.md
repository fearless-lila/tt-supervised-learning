# Supervised Learning + Bandit Decision System (End‑to‑End)

This repository (`tt-supervised-learning`) is **one half** of a two‑repo system.
Together with `table-tennis-multi-armed-bandit`, it forms a complete **offline prediction + online decision** learning loop.

This README intentionally documents **both repositories**, because the value of the project is in how they work *together*, not in either repo alone.

---

## 1. Problem Statement (First Principles)

We want to repeatedly choose **one drill** (an *action*) for a player to train next.

At every session, we face two competing goals:

1. **Do what is likely to work well now** (good short‑term experience)
2. **Learn what actually works over time** (good long‑term outcomes)

This is the classical **exploration vs exploitation** problem.

---

## 2. Why One Algorithm Is Not Enough

### Supervised learning alone is insufficient

Supervised learning answers:

> “What is likely to happen if I choose this drill in this context?”

But it **cannot decide actions safely**, because:

* it reinforces historical bias
* it does not explore alternatives
* it cannot correct itself online

### Bandits alone are insufficient

Bandits answer:

> “What should I choose next, given uncertainty?”

But they suffer from:

* cold start (no data at the beginning)
* noisy early exploration

---

## 3. The Core Idea of This Project

**Combine both**:

* Use **supervised learning** to provide a *soft prior* (initial belief)
* Use a **bandit** to make decisions and override that belief with real feedback

This mirrors real‑world personalization systems used in retail, recommendations, and experimentation.

---

## 4. High‑Level Architecture

### Repositories

```
Repo A: table-tennis-multi-armed-bandit
  ├─ online decision making (UCB1 bandit)
  ├─ logging real outcomes
  └─ using supervised predictions as priors

Repo B: tt-supervised-learning  (this repo)
  ├─ dataset creation from logs
  ├─ supervised regression model
  └─ exporting a reusable prediction artifact
```

---

## 5. End‑to‑End Data Flow (Flowchart)

```
┌──────────────┐
│ Player trains│
└──────┬───────┘
       │
       ▼
┌──────────────────────────┐
│ Bandit selects a drill   │
│ (UCB1 + supervised prior)│
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Player performs drill    │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Outcome logged           │
│ sessions.jsonl           │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ export_dataset.py        │
│ → supervised_dataset.csv │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Supervised training      │
│ (this repo)              │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ model.joblib + schema    │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Bandit loads model       │
│ as soft prior            │
└──────────────────────────┘
```

This loop repeats continuously.

---

## 6. What the Supervised Model Learns

### Target Variable

The supervised task is **regression**, predicting:

```
y_rate = successes / attempts
```

* Range: `[0, 1]`
* Interpretation: estimated probability of success for a drill in a given context

This replaces earlier boolean or purely subjective labels.

---

## 7. Supervised Features

The model learns from **context + drill**:

* `drill_id`
* `focus` (goal)
* `time_of_day`
* `skill_level`
* `fatigue`
* `difficulty`
* `session_minutes`

These features intentionally mirror what is available **at decision time**.

---

## 8. What This Repo Does (tt-supervised-learning)

### Key scripts

* `export_dataset.py` (in bandit repo)

  * converts `sessions.jsonl` → CSV

* `train.py`

  * trains a regression model
  * outputs `artifacts/model.joblib`

* `evaluate.py`

  * validates on a held‑out split

* `predictor.py`

  * lightweight inference interface

### Output artifacts

* `model.joblib` – trained sklearn pipeline
* `schema.json` – exact feature order contract

These artifacts are **copied into the bandit repo**.

---

## 9. How the Bandit Uses the Supervised Model

At runtime, for each candidate drill:

1. Build the feature vector
2. Predict success probability `p_hat ∈ [0,1]`
3. Convert to reward scale: `prior_mean = 100 × p_hat`
4. Treat this as **pseudo‑observations**

Mathematically:

```
mean' = (real_reward + prior_mean × prior_pulls)
        / (real_pulls + prior_pulls)
```

* `prior_pulls` is deliberately small (e.g. 5–10)
* Real experience eventually dominates

This solves cold start **without locking in wrong beliefs**.

---

## 10. Why This System Is Correct

The system demonstrates all required properties:

* ✅ Cold‑start guidance
* ✅ Online learning from real feedback
* ✅ Ability to override wrong predictions
* ✅ Clear separation of prediction vs decision

Observed behaviour:

* Early sessions follow supervised priors
* Repeated bad outcomes reduce selection probability
* Bandit eventually switches drills

This proves the integration is functioning correctly.

---

## 11. What This Project Represents

This is a **miniature but complete personalization system**:

* Offline learning (supervised)
* Online learning (bandit)
* Continuous feedback loop

The same architectural pattern is used in:

* product recommendation
* notifications
* experimentation systems
* retail personalization

Only the scale differs.

---

## 12. How to Run (Summary)

### Bandit repo

```bash
source .venv/bin/activate
python3 -m app.main
```

### Supervised repo

```bash
source .venv/bin/activate
python train.py
python evaluate.py
```

Artifacts are then copied into the bandit repo.

---

## 13. Final Mental Model (Most Important)

> **Supervised learning provides memory.**
> **Bandits provide adaptation.**

Prediction answers *“what is likely?”*
Decision answers *“what should I do next?”*

This project shows how—and why—you must use both.

---

## 14. Project Status

This project is **complete as a learning system**.

Further work (UI, more data, alternative bandits) is optional and not required to validate the core idea.
