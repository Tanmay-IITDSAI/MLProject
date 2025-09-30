# MLProject - Adaptive Wait-k Policy for Simultaneous Text-to-Text Machine Translation

Simultaneous Machine Translation (SiMT) aims to generate translations simultaneously with the reading of the source sentence, balancing translation quality with latency. Most SiMT models currently require training multiple models for different latency levels, thus increasing computational costs and, more importantly, limiting flexibility. The new approach is, like Mixture- of-Experts Wait-k policy, training multiple wait-k values in balance between the considerations of both latency and translation quality, leaving the determination of the optimal value of k for unseen data as an open challenge. Moreover, variability in the structure of structure between different languages makes the problem even more complicated because the application of a fixed policy becomes rather ineffective.

* Base Model: The project will utilize the Mixture-of-Experts Wait-k policy as the backbone model. This policy allows each head of the multi-head attention mechanism to perform translation with different levels of latency.

## Contents

This repo contains (high level):

- `MLProject_pytorch+SCST+SiMT.ipynb` — primary notebook with model code, experiments and walkthroughs.  
- `HMT-SiLLM_2.py` — Python script(s) related to HMT / SiLLM experiments.  
- `requirement.txt` — Python package dependencies.  
- `test_dataset.json` — example dataset format (input → target pairs).   
- `*.pdf` — project writeups and mathematical notes.

## Table of contents

* [Summary](#summary)
* [Key contributions](#key-contributions)
* [Repository structure (high level)](#repository-structure-high-level)
* [Requirements](#requirements)
* [Quick start](#quick-start)
* [Typical workflow](#typical-workflow)
* [Example commands](#example-commands)
* [Large models & LoRA notes](#large-models--lora-notes)
* [Evaluation & metrics](#evaluation--metrics)
* [Known issues & caveats](#known-issues--caveats)
* [Reproducibility tips](#reproducibility-tips)
* [Contributing](#contributing)
* [References](#References)

---

# Workflow

### a. Dataset Preparation  
- **Data Format:** JSON files with input-output sentence pairs.  
- **Tokenization:** Utilized AutoTokenizer from Hugging Face for sentence processing.  
- **Padding and Alignment:** Padded source sentences and shifted decoder input for alignment.

### b. Dynamic Wait-k Policy Implementation  
- Utilized a **flexible wait-k strategy** to dynamically adjust latency based on remaining input length.  
- Enhanced with HMT to predict sequence likelihoods, improving token generation decisions.  
- **Wait-K Policy Formula**:    $$g(t; k) = \min(k + t - 1, |Z|)$$

### c. SCST Fine-Tuning and RL Integration  
- **Reward Function:** Optimized using BLEU and ROUGE metrics.  
- **Policy Optimization:** RL agent trained via policy gradients.  
- **Advantage Calculation:** Based on the difference between sampled and baseline rewards.
- **SCST Reward Formula**:   $$R(\theta) = \sum_{t=1}^T (r_t - b_t) \log P(y_t | x; \theta)$$
  
### d. Model Architecture and Optimization  
- **Base Model:** LLaMA-7B fine-tuned with LoRA, supported by HMT for improved sequence prediction.  
- **Optimization:** Adam optimizer with cross-entropy loss.  
- **Device Compatibility:** Supports GPU (CUDA), MPS (Apple Silicon), and CPU.

### e. Evaluation Metrics  
- **BLEU Score:** Measures translation quality using n-gram overlaps.  
- **ROUGE-L Score:** Assesses informativeness and coverage.  
- **Latency:** Quantified via read-write sequence length ratio.
- **Latency Metric (AL)**:    $$AL = \frac{1}{\tau} \sum_{t=1}^\tau \left[g(t) - t - 1\right] \cdot \frac{|y|}{|x|}$$ 

# Evaluation and Findings
1. Dynamic Wait-k Policy significantly improved latency-quality trade-offs.  
2. SCST Fine-Tuning optimized performance through reinforcement learning.  
3. HMT Integration enhanced real-time adaptability.  
4. LoRA-enhanced LLaMA model ensured resource-efficient translations.  
5. BLEU and ROUGE scores provided robust evaluation metrics.

---

## Key contributions

* Adaptive Wait‑k implementation (dynamic wait decisions based on state/features).
* SCST (reinforcement learning) fine‑tuning to directly optimise quality‑latency reward.
* Integration of HMT building blocks and LoRA adapters for parameter‑efficient fine‑tuning of large models.
* End‑to‑end notebooks and scripts for training, evaluation and analysis.

---

## Requirements

* Python 3.8+
* GPU recommended (NVIDIA CUDA) for model fine‑tuning.
* Key Python libraries (suggested): `torch`, `transformers`, `datasets`, `sacrebleu`, `sentencepiece` (if using byte‑pair tokenizers), `accelerate` (optional), `einops`, `numpy`, `tqdm`.

Install example:

```bash
python -m pip install -r requirement.txt
```

If you use LoRA code from `peft` or `loralib`, install those packages as well (see `requirement.txt`).

---

## Quick start

1. Clone the repository and enter the folder:

```bash
git clone https://github.com/Tanmay-IITDSAI/MLProject.git
cd MLProject
```

2. Create & activate a virtual environment (optional but recommended):

```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
pip install -r requirement.txt
```

3. Inspect the example dataset:

```bash
less test_dataset.json
# or open in your editor / notebook
```

4. Open and run the main notebook:

```bash
jupyter notebook MLProject_pytorch+SCST+SiMT.ipynb
```

The notebook walks through data loading, scalar/sequence metrics, model training (baseline), SCST fine‑tuning, and evaluation.

---

## Typical workflow

1. Inspect / preprocess dataset; convert to the expected JSON format (list of `{"src":..., "tgt":...}` pairs).
2. Train a baseline simultaneous model or configure a pre‑trained model for online decoding.
3. Run adaptive Wait‑k policy training (supervised / imitation learning stage).
4. Apply SCST for reward‑based fine‑tuning to trade off BLEU vs latency.
5. Evaluate with BLEU/ROUGE and latency metrics (Average Lagging, Consecutive Waits, etc.).
6. Optionally apply LoRA adapters and re‑run experiments with large models.

---

## Example commands

These commands are illustrative — check each script's `--help` for exact flags.

```bash
# Run a training script (small-scale demo)
python HMT-SiLLM_2.py --data test_dataset.json --epochs 5 --batch_size 16 --lr 1e-4 --save_dir checkpoints/demo

# Evaluate a saved checkpoint
python evaluate.py --model checkpoints/demo/best.pt --test_data test_dataset.json --metrics bleu,avg_lagging

# Run the notebook non-interactively (NBConvert) to execute cells
jupyter nbconvert --to notebook --execute MLProject_pytorch+SCST+SiMT.ipynb --output executed.ipynb
```

---

## Large models & LoRA notes

* The repo references experiments with large models (e.g., LLaMA family). **Model weights are not included** — obtain them separately and ensure you follow licensing requirements.
* LoRA (Low‑Rank Adaptation) adapters are used to limit the number of trainable parameters. This is helpful when fine‑tuning large models on limited hardware.
* For LoRA training: use mixed precision (AMP), gradient accumulation and multi‑GPU if available.

---

## Evaluation & metrics

* **Quality:** BLEU, SacreBLEU, ROUGE (where applicable), and human/LLM judgments.
* **Latency:** Average Lagging (AL), Average Proportion (AP), and other simultaneous translation metrics.
* **Reward design:** SCST optimises a composite reward (e.g., BLEU − λ × latency). The notebook contains examples of reward formulations and hyperparameters.

---

## Reproducibility tips

* Fix random seeds (`numpy`, `torch`, `random`) and log seed values in run configs.
* Pin package versions in `requirement.txt` (or provide an `environment.yml`).
* Use smaller subsets for debugging and only scale up after pipeline correctness is verified.
* Save model checkpoints, training logs and hyperparameter configs alongside results.

---

## Contributing

Contributions, issues and PRs are welcome. Suggested improvements:

* Add robust CLI docs and a configuration system (e.g., Hydra / OmegaConf).
* Add unit tests for data preprocessing and evaluation metrics.
* Integrate a lightweight experiment management (Weights & Biases, MLflow) for reproducibility.
---

## References

- **Zhang, S., & Feng, Y. (2021).**  
  *Universal Simultaneous Machine Translation with Mixture-of-Experts Wait-k Policy.*  
  Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 7306–7317.  
  [Available here](https://aclanthology.org/2021.emnlp-main.581/)

- **Gu, J., et al. (2017).**  
  *Learning to translate in real-time with neural machine translation.*  
  Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics.  
  [Available here](https://aclanthology.org/E17-1099)

- **Grissom II, A., He, H., Boyd-Graber, J., Morgan, J., & Daumé III, H. (2014).**  
  *Don’t Until the Final Verb Wait: Reinforcement Learning for Simultaneous Machine Translation.*  
  Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1342–1352.  
  [Available here](https://aclanthology.org/D14-1140/)

# Pending Tasks & Challenges
We are diligently working on implementing the project, leveraging the latest advancements in simultaneous machine translation, including the Mixture-of-Experts approach and the dynamic Wait-k policy. While we strive to achieve the best possible outcomes, we acknowledge that the code and methodology are still evolving. As we proceed, there may be adjustments to our approach based on practical challenges, insights gained during experimentation, and efforts to optimize performance. Our aim is to ensure that the final implementation aligns with the project objectives while maintaining flexibility for improvements.
I also found that the code does not seem to work properly as some part of the code or requirement is missing to run the code though the paper explained everything. 
