import ctranslate2
import sentencepiece as spm
import torch
import sacrebleu
import pandas as pd
import argparse 
import yaml
import logging
# from logging import handlers

from datasets import load_dataset
from tabulate import tabulate
import csv
from datetime import datetime

import os
import time

import json
import os

RESULTS_FILE = "evaluation_results.json"

# Load existing results
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "r") as f:
        existing_results = json.load(f)
else:
    existing_results = {}



def setup_logging(debug, log_dir, log_file):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logging.getLogger("comet").propagate = False

    # Prevent duplicate logs if this function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter for both console and file
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_log_path = os.path.join(log_dir, f"{timestamp}_{log_file}")
    file_handler = logging.FileHandler(full_log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging initialized. Log file: {full_log_path}")

    return logger


def setup_logger(config):
    log_cfg = config["logging"]
    logger = setup_logging(
        debug=log_cfg.get("debug", True),
        log_dir=log_cfg.get("log_dir", "logs/"),
        log_file=log_cfg.get("log_file", "run.log")
    )
    return logger


def load_comet_model(model_name="masakhane/africomet-mtl"):
    from comet import download_model, load_from_checkpoint


    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    return model

def load_models(ct_model_path, sp_model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    translator = ctranslate2.Translator(ct_model_path, device=device)   
    return sp, translator



def generate_translations(sp, translator, src_lang, tgt_lang, source_sentences, batch_size=2048, beam_size=2):
    # Clean sentences
    source_sents = [sent.strip() for sent in source_sentences]

    # Target prefix
    target_prefix = [[tgt_lang]] * len(source_sents)

    # Subword encode and add special tokens
    source_sents_subworded = sp.encode_as_pieces(source_sents)
    source_sents_subworded = [[src_lang] + sent + [""] for sent in source_sents_subworded]

    # Count input tokens
    num_input_tokens = sum(len(s) for s in source_sents_subworded)

    # Run translation with timing
    start = time.perf_counter()
    results = translator.translate_batch(
        source_sents_subworded,
        batch_type="tokens",
        max_batch_size=batch_size,
        beam_size=beam_size,
        target_prefix=target_prefix
    )
    end = time.perf_counter()
    execution_time = end - start

    # Extract best hypotheses
    translations = [result.hypotheses[0] for result in results]

    # Count output tokens
    num_output_tokens = sum(len(t) for t in translations)

    # Decode
    translations_desubword = sp.decode(translations)
    translations_desubword = [sent[len(tgt_lang):].strip() for sent in translations_desubword]

    # Metrics
    total_tokens = num_input_tokens + num_output_tokens
    tokens_per_sec = total_tokens / execution_time if execution_time > 0 else float("inf")

    metrics = {
        "execution_time_sec": execution_time,
        "input_tokens": num_input_tokens,
        "output_tokens": num_output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec
    }

    return translations_desubword, metrics

def evaluate_model(source_sentences, translations, references, comet_model, logger):
    bleu = sacrebleu.corpus_bleu(translations, [references])
    bleu = round(bleu.score, 2)
    logger.info(f"BLEU: {bleu}")

    chrf = sacrebleu.corpus_chrf(translations, [references], word_order=2)
    chrf = round(chrf.score, 2)
    logger.info(f"chrF++: {chrf}")

    # metric = sacrebleu.metrics.TER()
    # ter = metric.corpus_score(translations, [references])
    # ter = round(ter.score, 2)
    # logger.info(f"TER: {ter}")

    df = pd.DataFrame({"src":source_sentences, "mt":translations, "ref":references})
    data = df.to_dict(orient="records")
    comet_output= comet_model.predict(data, batch_size=16, gpus=1 if torch.cuda.is_available() else 0) 
    comet_score = round(comet_output["system_score"]*100, 2)
    logger.info(f"COMET: {comet_score}")

    return bleu, chrf, comet_score

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = argparser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    log_file = config.get("log_file", "logs/am_eval.log")
    logger = setup_logger(config)
    logger.info("Starting evaluation script.")
    # log_file = config.get("log_file", "logs/am_eval.log")
    ct_model_path = config.get("ct_model_path") or config.get("ct2_model_path")
    sp_model_path = config["sp_model_path"]
    comet_model_name = config.get("comet_model_name", "masakhane/africomet-mtl")
    batch_size = config.get("batch_size", 2048)
    beam_size = config.get("beam_size", 2)

    logger.info("Loading models...")
    sp, translator = load_models(ct_model_path, sp_model_path)
    comet_model = load_comet_model(comet_model_name)

    # summary_log = []

    testsets = config.get("testset", {})
    for test_name, test_cfg in testsets.items():
        if test_name in existing_results:
            summary_log = list(existing_results.values())
            logger.info(f"✅ Skipping {test_name}, already evaluated.")
            continue
        logger.info(f"Evaluating testset: {test_name}")
        dataset_cfg = test_cfg["dataset"]
        src_lang = test_cfg["src_lang"]
        tgt_lang = test_cfg["tgt_lang"]
        dataset_path = dataset_cfg["path"]
        src_config = dataset_cfg["src_config"]
        tgt_config = dataset_cfg["tgt_config"]
        text_col = dataset_cfg.get("text_col", "sentence")
        split = dataset_cfg.get("split", "test")

        logger.info(f"Loading dataset from {dataset_path} (split: {split})")
        ds_src = load_dataset(dataset_path, src_config, split=split, trust_remote_code=True)
        ds_tgt = load_dataset(dataset_path, tgt_config, split=split, trust_remote_code=True)
        source_sentences = ds_src[text_col]
        reference_sentences = ds_tgt[text_col]

        logger.info("Generating translations...")
        translations, metrics = generate_translations(
            sp, translator, src_lang, tgt_lang, source_sentences,
            batch_size=batch_size, beam_size=beam_size
        )

        logger.info("Evaluating translations...")
        bleu, chrf, comet_score = evaluate_model(source_sentences, translations, reference_sentences, comet_model, logger)

        summary_entry = {
            "Testset": test_name,
            "Source Lang": src_lang,
            "Target Lang": tgt_lang,
            # "Num Samples": len(source_sentences),
            "BLEU": bleu,
            "chrF++": chrf,
            "COMET": comet_score,
            "Execution Time (s)": round(metrics.get("execution_time_sec", 0), 4),
            "Input Tokens": metrics.get("input_tokens", 0),
            "Output Tokens": metrics.get("output_tokens", 0),
            "Total Tokens": metrics.get("total_tokens", 0),
            "Tokens/sec": round(metrics.get("tokens_per_sec", 0), 4)
        }



        # Save after each testset to prevent data loss
        existing_results[test_name] = summary_entry
        with open(RESULTS_FILE, "w") as f:
            json.dump(existing_results, f, indent=2)

        # summary_log.append(summary_entry)
        summary_log = list(existing_results.values())


    if summary_log:
        summary_table = [
            [
                entry["Testset"],
                entry["Source Lang"],
                entry["Target Lang"],
                # entry["Num Samples"],
                entry["BLEU"],
                entry["chrF++"],
                entry["COMET"],
                entry["Execution Time (s)"],
                entry["Input Tokens"],
                entry["Output Tokens"],
                entry["Total Tokens"],
                entry["Tokens/sec"]
            ]
            for entry in summary_log
        ]
        logger.info("\n📊 Evaluation Summary:\n" + tabulate(
            summary_table,
            headers=[
                "Testset", "Source Lang", "Target Lang", "BLEU", "chrF++", "COMET",
                "Execution Time (s)", "Input Tokens", "Output Tokens", "Total Tokens", "Tokens/sec"
            ],
            tablefmt="github"
        ))

        with open("evaluation_summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Testset", "Source Lang", "Target Lang", "BLEU", "chrF++", "COMET",
                "Execution Time (s)", "Input Tokens", "Output Tokens", "Total Tokens", "Tokens/sec"
            ])
            writer.writerows(summary_table)

if __name__ == "__main__":
    main()
