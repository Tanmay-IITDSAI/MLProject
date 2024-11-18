# MLProject
Simultaneous Machine Translation (SiMT) aims to generate translations simultaneously with the reading of the source sentence, balancing translation quality with latency. Most SiMT models currently require training multiple models for different latency levels, thus increasing computational costs and, more importantly, limiting flexibility. The new approach is, like Mixture- of-Experts Wait-k policy, training multiple wait-k values in balance between the considerations of both latency and translation quality, leaving the determination of the optimal value of k for unseen data as an open challenge. Moreover, variability in the structure of structure between different languages makes the problem even more complicated because the application of a fixed policy becomes rather ineffective.

* Base Model: The project will utilize the Mixture-of-Experts Wait-k policy as the backbone model. This policy allows each head of the multi-head attention mechanism to perform translation with different levels of latency.

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
- Formula:  
  \[
  g(t; k) = \min(k + t - 1, |Z|)
  \]

### c. SCST Fine-Tuning and RL Integration  
- **Reward Function:** Optimized using BLEU and ROUGE metrics.  
- **Policy Optimization:** RL agent trained via policy gradients.  
- **Advantage Calculation:** Based on the difference between sampled and baseline rewards.

### d. Model Architecture and Optimization  
- **Base Model:** LLaMA-7B fine-tuned with LoRA, supported by HMT for improved sequence prediction.  
- **Optimization:** Adam optimizer with cross-entropy loss.  
- **Device Compatibility:** Supports GPU (CUDA), MPS (Apple Silicon), and CPU.

### e. Evaluation Metrics  
- **BLEU Score:** Measures translation quality using n-gram overlaps.  
- **ROUGE-L Score:** Assesses informativeness and coverage.  
- **Latency:** Quantified via read-write sequence length ratio.
- 
# Evaluation and Findings
1. Dynamic Wait-k Policy significantly improved latency-quality trade-offs.  
2. SCST Fine-Tuning optimized performance through reinforcement learning.  
3. HMT Integration enhanced real-time adaptability.  
4. LoRA-enhanced LLaMA model ensured resource-efficient translations.  
5. BLEU and ROUGE scores provided robust evaluation metrics.

# Pending Tasks & Challenges
* The method is subject to change using PyTorch or Tensorflow, as the project is under progress.
