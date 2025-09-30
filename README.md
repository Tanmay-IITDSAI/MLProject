# MLProject 
### Under Guidance of: Dr. Rajesh Kumar Mundotiya
Simultaneous Machine Translation (SiMT) aims to generate translations simultaneously with the reading of the source sentence, balancing translation quality with latency. Most SiMT models currently require training multiple models for different latency levels, thus increasing computational costs and, more importantly, limiting flexibility. The new approach is, like Mixture- of-Experts Wait-k policy, training multiple wait-k values in balance between the considerations of both latency and translation quality, leaving the determination of the optimal value of k for unseen data as an open challenge. Moreover, variability in the structure of structure between different languages makes the problem even more complicated because the application of a fixed policy becomes rather ineffective.

* Base Model: The project will utilize the Mixture-of-Experts Wait-k policy as the backbone model. This policy allows each head of the multi-head attention mechanism to perform translation with different levels of latency.

## Contents

This repo contains (high level):

- `MLProject_pytorch+SCST+SiMT.ipynb` — primary notebook with model code, experiments and walkthroughs.  
- `HMT-SiLLM_2.py` — Python script(s) related to HMT / SiLLM experiments.  
- `requirement.txt` — Python package dependencies.  
- `test_dataset.json` — example dataset format (input → target pairs).   
- `*.pdf` — project writeups and mathematical notes.

# Project Objectives  
* Develop a Dynamic Wait-k Policy that adaptively balances latency and quality in real-time.
* Integrate Self-Critical Sequence Training (SCST) to optimize the quality-latency trade-off using reinforcement learning.
* Evaluate translation quality using BLEU and ROUGE metrics while minimizing latency.

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

## Quick start

> Recommended: Linux/macOS with a GPU. Use a virtual environment to avoid dependency conflicts.

1. Clone
```bash
git clone https://github.com/Tanmay-IITDSAI/MLProject.git
cd MLProject

## Create & activate a virtual env (optional but recommended)
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1


pip install -r requirement.txt

jupyter notebook MLProject_pytorch+SCST+SiMT.ipynb

Typical workflow

Run the notebook to reproduce small-scale experiments and understand data preprocessing & training loops.

Prepare/convert your dataset to the same JSON format as test_dataset.json. Tokenization and batching details are in the notebook/scripts.

Use the provided scripts (e.g. HMT-SiLLM_2.py) for training or evaluation — inspect the script headers to find CLI arguments.

Optionally perform SCST fine-tuning (reinforcement learning) using the reward functions (BLEU/ROUGE/latency) described in the notebook and project docs.

Evaluate with BLEU / ROUGE and latency metrics (Average Lagging or other AL-style metrics).

Notes on large models & LoRA

The repo mentions experiments with LLaMA-7B and LoRA. Large model weights (e.g. LLaMA) are not included — obtain them via the appropriate channels and follow license/usage restrictions.

LoRA adapters are lightweight, but training still benefits from GPU(s) and may require multi-GPU or mixed precision.

Files to inspect first

MLProject_pytorch+SCST+SiMT.ipynb — start here to understand the pipeline, hyperparameters, and how datasets are loaded.

requirement.txt — verify package versions and install any missing dependencies.

HMT-SiLLM_2.py — look for main/train/eval functions and CLI arguments.

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
