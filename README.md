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

# Mathematical Approach
\documentclass{article}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{geometry}

% Page Layout
\geometry{a4paper, margin=1in}

\title{Hidden Markov Transformer (HMT) Basics and Combined Objective for HMT with SCST and Dynamic Wait-k}
\author{}
\date{}

\begin{document}

\maketitle

\section*{Hidden Markov Transformer (HMT) Basics}

The \textbf{Hidden Markov Transformer (HMT)} combines the principles of Hidden Markov Models (HMMs) and Transformers for sequence modeling. The key mathematical formulations are as follows:

\subsection*{Joint Sequence Probability in HMT}
\begin{equation}
P(X, Z) = P(Z) \prod_{t=1}^T P(x_t \mid z_t)
\end{equation}
where:
\begin{itemize}
    \item \( X = (x_1, x_2, \dots, x_T) \): Observed sequence (e.g., source or target sequence),
    \item \( Z = (z_1, z_2, \dots, z_T) \): Hidden states (latent variables),
    \item \( P(Z) \): Transition probability between hidden states,
    \item \( P(x_t \mid z_t) \): Emission probability of observing \( x_t \) given \( z_t \).
\end{itemize}

\subsection*{Transformer Attention for Transition}
The transition probability \( P(Z) \) is modeled using self-attention:
\begin{equation}
A_{ij} = \text{softmax} \left( \frac{Q_i K_j^\top}{\sqrt{d_k}} \right)
\end{equation}
where:
\begin{itemize}
    \item \( Q, K, V \): Query, key, and value matrices for self-attention,
    \item \( d_k \): Dimensionality of the hidden states.
\end{itemize}

\section*{Combined Objective for HMT with SCST and Dynamic Wait-k}

The overall objective integrates \textbf{Self-Critical Sequence Training (SCST)}, \textbf{HMT probabilities}, and the \textbf{Wait-k policy}:

\subsection*{HMT Loss}
\begin{equation}
\mathcal{L}_{HMT} = \mathbb{E}_{Z} \left[ - \sum_{t=1}^T \log P(x_t \mid z_t) \right]
\end{equation}

\subsection*{SCST Loss}
The SCST reward \( R(\theta) \) is computed as:
\begin{equation}
R(\theta) = \sum_{t=1}^T (r_t - b_t) \log P(y_t \mid x; \theta)
\end{equation}
where:
\begin{itemize}
    \item \( r_t \): Reward at step \( t \) (e.g., BLEU score),
    \item \( b_t \): Baseline reward (computed using greedy decoding).
\end{itemize}

The SCST loss is:
\begin{equation}
\mathcal{L}_{SCST} = - \sum_{t=1}^T (r_t - b_t) \log P(y_t \mid x; \theta)
\end{equation}

\subsection*{Dynamic Wait-k Selection}
The Wait-k function dynamically determines the source tokens to read before decoding:
\begin{equation}
g(t; k) = \min(k + t - 1, |Z|)
\end{equation}
where:
\begin{itemize}
    \item \( k \): Number of source tokens to ``wait'',
    \item \( t \): Current decoding step,
    \item \( |Z| \): Length of the latent states.
\end{itemize}

To optimize \( k \) based on translation quality and latency:
\begin{equation}
k^* = \arg\max_k \left[ \text{BLEU} - \alpha \cdot AL \right]
\end{equation}
where:
\begin{itemize}
    \item \( AL = \frac{1}{\tau} \sum_{t=1}^\tau \left[g(t) - t - 1\right] \cdot \frac{|y|}{|x|}
    \end{itemize}
    \( AL \): Average Lagging (latency metric),
    \item \( \alpha \): Hyperparameter balancing quality and latency.
\end{itemize}

\subsection*{Final Combined Loss}
\begin{equation}
\mathcal{L} = \mathcal{L}_{HMT} + \lambda \mathcal{L}_{SCST}
\end{equation}
where \( \lambda \) is a weighting factor for reinforcement learning.

\end{document}


# Evaluation and Findings
1. Dynamic Wait-k Policy significantly improved latency-quality trade-offs.  
2. SCST Fine-Tuning optimized performance through reinforcement learning.  
3. HMT Integration enhanced real-time adaptability.  
4. LoRA-enhanced LLaMA model ensured resource-efficient translations.  
5. BLEU and ROUGE scores provided robust evaluation metrics.

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
