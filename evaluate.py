import ctranslate2
import sentencepiece as spm
import torch
import sacrebleu
import pandas as pd
import argparse 
import yaml
import logging
from logging import handlers

from datasets import load_dataset

import os

def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("mt-eval")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    fh = handlers.RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def load_comet_model(model_name="Unbabel/wmt22-comet-da"):
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
    source_sents = [sent.strip() for sent in source_sentences]
    target_prefix = [[tgt_lang]] * len(source_sents)
    source_sents_subworded = sp.encode_as_pieces(source_sents)
    source_sents_subworded = [[src_lang] + sent + [""] for sent in source_sents_subworded]
    translations = translator.translate_batch(
        source_sents_subworded,
        batch_type="tokens",
        max_batch_size=batch_size,
        beam_size=beam_size,
        target_prefix=target_prefix
    )
    translations = [translation.hypotheses[0] for translation in translations]
    translations_desubword = sp.decode(translations)
    translations_desubword = [sent[len(tgt_lang):].strip() for sent in translations_desubword]  
    return translations_desubword

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
    logger = setup_logger(log_file)
    logger.info("Starting evaluation script.")

    ct_model_path = config.get("ct_model_path") or config.get("ct2_model_path")
    sp_model_path = config["sp_model_path"]
    comet_model_name = config.get("comet_model_name", "Unbabel/wmt22-comet-da")
    src_lang = config["src_lang"]
    tgt_lang = config["tgt_lang"]
    dataset_path = config["dataset"]["path"]
    src_config = config["dataset"]["src_config"]
    tgt_config = config["dataset"]["tgt_config"]
    text_col = config["dataset"].get("text_col", "sentence")
    split = config["dataset"].get("split", "test")
    batch_size = config.get("batch_size", 2048)
    beam_size = config.get("beam_size", 2)

    logger.info(f"Loading dataset from {dataset_path} (split: {split})")
    ds_src= load_dataset(dataset_path, src_config, split=split, trust_remote_code=True)
    ds_tgt= load_dataset(dataset_path, tgt_config, split=split, trust_remote_code=True)
    source_sentences = ds_src[text_col][:5]
    reference_sentences = ds_tgt[text_col][:5]
    logger.info("Loading models...")
    sp, translator = load_models(ct_model_path, sp_model_path)
    comet_model = load_comet_model(comet_model_name)

    logger.info("Generating translations...")
    translations = generate_translations(sp, translator, src_lang, tgt_lang, source_sentences, batch_size=batch_size, beam_size=beam_size)

    logger.info("Evaluating translations...")
    evaluate_model(source_sentences, translations, reference_sentences, comet_model, logger)

if __name__ == "__main__":
    main()
