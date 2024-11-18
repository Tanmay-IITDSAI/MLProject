import os
import sys
import fire
import torch
from transformers import GenerationConfig, LlamaForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from evaluate import load as load_metric
from utils.prompter import Prompter
import json
import time
import numpy as np
import nltk
from rouge_score import rouge_scorer

# Download NLTK data
nltk.download('punkt')

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu" and torch.backends.mps.is_available():
    device = "mps"

# Initialize BLEU and ROUGE metrics
bleu_metric = load_metric("bleu")
rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def main(
    load_8bit: bool = False,
    base_model: str = "huggyllama/llama-7b",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "alpaca",
    data_path: str = "data/test_dataset.json",
    output_translation_path: str = "output/translations.json",
    Bottom: int = 1,
    Top: int = 5,
    epochs: int = 3,
    learning_rate: float = 1e-5,
):
    """
    Main function to perform translation using dynamic wait-k policy with HMT and SCST.
    """
    assert base_model, "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    # Initialize Prompter
    prompter = Prompter(prompt_template)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Load Model with PEFT (LoRA)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # Fix tokenizer and model config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # Use FP16 for efficiency

    model.to(device)
    model.eval()

    # Compile model for faster inference if possible
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # Evaluation function
    def evaluate(
        instruction,
        input_text=None,
        output_text=None,
        suppress_tokens=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
    ):
        prompt = prompter.generate_prompt(instruction, input_text, output_text)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        generation_config = GenerationConfig(
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            suppress_tokens=suppress_tokens,
            max_new_tokens=max_new_tokens,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=inputs["input_ids"],
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_sequence = generation_output.sequences[0]
        decoded_output = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        response = prompter.get_response(decoded_output)
        token_count = generated_sequence.size(-1) - inputs["input_ids"].size(-1)
        return response, token_count

    # HMT Policy Function
    def HMT_policy(
        instruction,
        input_text=None,
        policy=[],
        Lower=1,
        Upper=5,
        num_beams=1,
        max_new_tokens=256
    ):
        cur_target_str = ""
        i = 0
        src_len = len(input_text.split())
        rw_seq = []
        first_time = True

        tran_tgt_seqLen = len(policy)
        suppress_tokens = [2]  # Suppress EOS during incremental generation
        total_tokens = 0

        for i in range(tran_tgt_seqLen):
            limited_policy = policy[i]
            if policy[i] < Lower + i:
                limited_policy = Lower + i
            elif policy[i] > Upper + i:
                limited_policy = Upper + i
            limited_policy = min(limited_policy, src_len)
            cut_input = ' '.join(input_text.split()[:limited_policy])
            tmp_max_new_tokens = 3
            if i >= (tran_tgt_seqLen - 1):
                tmp_max_new_tokens = max_new_tokens
                suppress_tokens = None

            cur_target_str, tmp_size = evaluate(
                instruction,
                cut_input,
                output_text=cur_target_str,
                suppress_tokens=suppress_tokens,
                num_beams=num_beams,
                max_new_tokens=tmp_max_new_tokens
            )
            total_tokens += tmp_size

            if i < (tran_tgt_seqLen - 1):
                cur_target_str = ' '.join(cur_target_str.split()[:i + 1])
                rw_seq.append(limited_policy)
                if cur_target_str.find('</s>') != -1:
                    break
            else:
                tmp_size = len(cur_target_str.split()) - i
                rw_seq = rw_seq + [src_len] * tmp_size

        rw_seq.append(src_len)
        return rw_seq, cur_target_str, total_tokens

    # SCST Fine-Tuning Function
    def fine_tune_model(dataset):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.train()

        for epoch in range(epochs):
            total_loss = 0

            for sample in dataset["train"]:
                optimizer.zero_grad()

                # Encode inputs and targets
                input_ids = tokenizer(sample["input"], return_tensors="pt", padding=True)["input_ids"].to(device)
                target_ids = tokenizer(sample["target"], return_tensors="pt", padding=True)["input_ids"].to(device)

                # Generate baseline translation with greedy decoding
                with torch.no_grad():
                    baseline_output = model.generate(input_ids, max_length=target_ids.shape[1], do_sample=False)
                baseline_translation = tokenizer.decode(baseline_output[0], skip_special_tokens=True)

                # Generate sampled translation
                sampled_output = model.generate(input_ids, max_length=target_ids.shape[1], do_sample=True)
                sampled_translation = tokenizer.decode(sampled_output[0], skip_special_tokens=True)

                # Calculate rewards (BLEU score as an example)
                baseline_reward = bleu_metric.compute(predictions=[baseline_translation.split()], references=[sample["target"].split()])["bleu"]
                sampled_reward = bleu_metric.compute(predictions=[sampled_translation.split()], references=[sample["target"].split()])["bleu"]
                advantage = sampled_reward - baseline_reward

                # Compute loss for sampled translation
                outputs = model(input_ids=input_ids, labels=target_ids)
                loss = outputs.loss * advantage  # Weight loss by the advantage
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataset["train"])
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # Dynamic Wait-K Policy Function
    def dynamic_wait_k_policy(instruction, input_text, waitk_initial=3, max_new_tokens=128):
        cur_target_str = ""
        i = 0
        src_len = len(input_text.split())
        rw_seq = []
        first_time = True
        suppress_tokens = [2]  # Suppress EOS during incremental generation

        while (i + waitk_initial <= src_len) or first_time:
            cut_input = ' '.join(input_text.split()[:min(i + waitk_initial, src_len)])
            cur_target_str, _ = evaluate(
                instruction,
                cut_input,
                output_text=cur_target_str,
                suppress_tokens=suppress_tokens,
                num_beams=1,
                max_new_tokens=5
            )

            if i + waitk_initial >= src_len:
                break
            cur_target_str = ' '.join(cur_target_str.split()[:i + 1])
            rw_seq.append(i + waitk_initial)

            # Dynamically adjust k based on the remaining tokens
            remaining = src_len - (i + waitk_initial)
            waitk_dynamic = min(Top, max(Bottom, int(np.log2(remaining + 1))))
            waitk_initial = waitk_dynamic

            first_time = False
            i += 1

        rw_seq.append(src_len)
        return rw_seq, cur_target_str

    # Translation and Evaluation Function
    def run_translation_and_evaluate(data_path):
        data = load_dataset("json", data_files=data_path)["train"]
        total_bleu, total_rouge, total_latency = [], [], []
        output_text = []
        j = 1
        start_time = time.time()

        for sample in data:
            print(f'Translating sample {j}...')
            j += 1
            rw_seq, translated_text = dynamic_wait_k_policy(
                sample["instruction"],
                sample["input"],
                waitk_initial=3,
                max_new_tokens=256
            )

            # Calculate BLEU score
            reference = [sample["target"].split()]
            hypothesis = translated_text.split()
            bleu = bleu_metric.compute(predictions=[hypothesis], references=[reference])["bleu"]
            total_bleu.append(bleu)

            # Calculate ROUGE score
            rouge = rouge_scorer_instance.score(sample["target"], translated_text)["rougeL"].fmeasure
            total_rouge.append(rouge)

            # Calculate latency
            latency = len(rw_seq) / len(sample["input"].split())
            total_latency.append(latency)

            # Store the translation and read-write sequence
            output_text.append({'rw': rw_seq, 'translation': translated_text})

        end_time = time.time()
        total_words = sum(len(item['translation'].split()) for item in output_text)

        # Save translations to a JSON file
        os.makedirs(os.path.dirname(output_translation_path), exist_ok=True)
        with open(output_translation_path, "w", encoding='utf-8') as fp:
            json.dump(output_text, fp, indent=4, ensure_ascii=False)

        # Print evaluation metrics
        avg_bleu = np.mean(total_bleu)
        avg_rouge = np.mean(total_rouge)
        avg_latency = np.mean(total_latency)
        print(f"\nAverage BLEU: {avg_bleu:.4f}")
        print(f"Average ROUGE-L: {avg_rouge:.4f}")
        print(f"Average Latency: {avg_latency:.4f}")
        print(f"Total Time: {end_time - start_time:.2f} seconds")
        print(f"Total Words Translated: {total_words}")

    # Load Dataset
    dataset = load_dataset("json", data_files=data_path)

    # Fine-tune the model with SCST (optional)
    print("Starting fine-tuning with SCST...")
    fine_tune_model(dataset)

    # Run translation and evaluation
    print("Starting translation and evaluation...")
    run_translation_and_evaluate(data_path)

if __name__ == "__main__":
    fire.Fire(main)