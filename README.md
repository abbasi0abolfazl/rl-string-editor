# String Metric RL (Reinforcement Learning for Text Editing & Distance Optimization)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[🇮🇷 مستندات فارسی (Persian Documentation)](README.fa.md) • [Report Issues](https://github.com/abbasi0abolfazl/string_metric_RL/issues)

An evidence-based Reinforcement Learning framework that formulates text editing, error correction, and string alignment as a **Markov Decision Process (MDP)**. The agent learns optimal sequences of character-level edits (`insert`, `delete`, `replace`, `keep`) to transform corrupted, noisy, or dialectal strings into canonical target forms while minimizing edit distance and step penalties.

---

## 📌 Problem Formulation (MDP)

Traditional string correction relies either on static Levenshtein / dynamic programming alignment (which cannot adapt dynamically to domain contexts) or large sequence-to-sequence models (which are compute-heavy). This project models text editing via RL:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Environment (State S_t)                         │
│  - Current Text vs. Target String                                      │
│  - Levenshtein & Jaro-Winkler Distance Metrics                         │
│  - Character-level Position & Distribution Features                    │
│  - Transformer & TF-IDF Semantic Embeddings                            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        Agent Action Space A_t                          │
│  - INSERT(char, pos)                                                   │
│  - DELETE(pos)                                                         │
│  - REPLACE(pos, char)                                                  │
│  - KEEP(pos)                                                           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        Reward Formulation R_t                          │
│  R_t = Δ(Similarity) - λ * (Step Cost) + Bonus(Exact Match)            │
│  - Encourages minimal edit sequences                                   │
│  - Penalizes redundant loops or destructive modifications              │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Features

- **Multi-Metric Text Analysis Engine (`analiys.py`):**
  - Syntactic distances: Levenshtein distance, Jaro-Winkler similarity, Normalized Edit Ratio.
  - Semantic representations: TF-IDF cosine similarity, character n-gram profiles, and HuggingFace Transformer embeddings.
  - Detailed character frequency, prefix/suffix patterns, and confusion matrix diagnostics.

- **Custom RL Environment (`AdvancedArabicTextEnv` in `rl.py`):**
  - Gym-like step-by-step text modification interface.
  - Observation space combining string distance vectors and positional features.
  - Configurable max step limits, early termination on convergence, and step penalties.

- **Multiple RL Paradigms Implemented (`arabic_rl_editor/rl_agent.py`):**
  - **DQN (Deep Q-Network):** Neural Q-function with Experience Replay buffer, target network stabilization, and $\epsilon$-greedy exploration.
  - **Tabular Q-Learning:** Discretized state representations for fast baseline exploration.
  - **Policy Gradient (REINFORCE):** Stochastic policy network directly parameterizing the distribution over edit operations.
  - **Expert Heuristic Baseline (`ExpertArabicEditor`):** Greedy optimal edit-distance alignment for benchmark comparisons.

- **Experimentation & Telemetry (`utils.py`, `train_rl.py`, `evaluate_rl.py`):**
  - `RLExperimentLogger`: Serializes training histories, reward curves, and evaluation metrics into JSON/pickle.
  - Comprehensive evaluation pipeline measuring Success Rate, Average Edit Steps, and Similarity Improvement over baseline.

---

## 📂 Repository Structure

```
string_metric_RL/
├── analiys.py                  # Text similarity metrics & linguistic feature extraction
├── rl.py                       # Environment (AdvancedArabicTextEnv) & MDP definitions
├── main.py                     # Integrated pipeline orchestrator
├── requirements.txt            # Dependency definitions
├── pyproject.toml              # Modern build configuration
└── arabic_rl_editor/
    ├── __init__.py             # Package exports
    ├── rl_agent.py             # DQN, Q-Learning, and Policy Gradient implementations
    ├── train_rl.py             # Training loop, replay scheduling, and history tracking
    ├── evaluate_rl.py          # Benchmark suite against Expert baseline
    └── utils.py                # Telemetry, checkpointing, and experiment logging
```

---

## 🛠️ Quickstart

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/abbasi0abolfazl/string_metric_RL.git
cd string_metric_RL

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Training an RL Agent

Train a Deep Q-Network (DQN) agent on text editing pairs:

```python
from arabic_rl_editor.train_rl import RLTextEditorTrainer

trainer = RLTextEditorTrainer(
    agent_type="dqn",
    config={
        "learning_rate": 0.001,
        "discount_factor": 0.95,
        "epsilon": 0.3,
        "epsilon_decay": 0.995,
        "batch_size": 32,
        "episodes": 500,
    }
)

# Define dataset of initial (noisy) and target texts
training_pairs = [
    ("کتاب", "كتاب"),
    ("مدرسه", "مدرسة"),
    ("دانشگاه", "دانشگاه"),
]

history = trainer.train_on_pairs(training_pairs)
trainer.save_agent("dqn_model.pth")
```

### 3. Evaluating & Comparing with Expert Baseline

```python
from arabic_rl_editor.evaluate_rl import RLTextEditorEvaluator

evaluator = RLTextEditorEvaluator(model_path="dqn_model.pth", agent_type="dqn")

result = evaluator.evaluate_single(
    initial_text="كتای",
    target_text="کتاب",
    max_edits=10,
    use_agent=True
)

print(f"Success: {result['success']}")
print(f"Total Edits: {result['total_edits']}")
print(f"Final Text: {result['final_text']}")
```

### 4. Running the Complete Experiment

To run an end-to-end experiment with logging and metric extraction:

```bash
python main.py
```

---

## 📊 Evaluation & Metrics

The framework evaluates policies based on:
1. **Convergence Rate (%):** Percentage of test strings successfully transformed to target string.
2. **Mean Steps to Goal:** Number of actions taken versus theoretical optimal Levenshtein distance.
3. **Similarity Gain:** $\Delta(\text{Similarity}_{\text{final}} - \text{Similarity}_{\text{initial}})$.
4. **Action Distribution:** Ratio of precision edits (`replace`) vs. destructive edits (`delete`).

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
