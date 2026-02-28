This repo evaluates NLLB-200 based machine translation models. It first generate translation for a given test dataset and evaluate the Ctranslate2 convereted model on different metrics; BLEU, CHRF++ and COMET. Chunking is applied for efficent translation generation and avoding memory overflows. 

---


## `Config.yaml`
The config file contains all the configuarations needed to evaluate the model including the model to be evaluated. Here is overview of its section.
### `model`
- `ct_model_path`: path to the ctranslate2 version of the model
- `sp_model_path`: path to the nllb sentence piece model
- `batch_size`: batch size 
- `beam_size`: beam size

### `testset`
- `path`:  huggingface id of the test dataset
- `src_config`: source language config name of the dataset
- `tgt_config`: target language config name of the dataset
- `text_col`: column name for the text data 
- `split`: which split of the dataset to be used for evaulation
- `src_lang`: source language code
- `tgt_lang`: target language code
- `comet_model_name`: Comet model path

### `logging`:
- `debug`: debug mode
- `log_dir`: directory to save the logs
- `log_file`: log file name
- `results_file`: results file name to save as json
- `evaluation_summary_file`: results file to save as csv

## Results
Results are displayed in a formatted table. They are also saved as CSV and JSON files. Logs are written to a log file.

## Citation

This repository is part of the [AfriNLLB](https://github.com/AfriNLP/AfriNLLB) project. 
If you use any part of the project's code, data, models, or approaches, please cite the following paper:

```
@inproceedings{moslem-etal-2026-afrinllb,
    title = "{A}fri{NLLB}: Efficient Translation Models for African Languages",
    author = "Moslem, Yasmin  and
      Wassie, Aman Kassahun  and
      Gizachew, Amanuel",
    booktitle = "Proceedings of the Seventh Workshop on African Natural Language Processing (AfricaNLP)",
    month = jul,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
}
```
