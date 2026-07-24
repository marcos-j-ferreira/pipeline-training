## What this is
A small Python-based pipeline for training and running a simple transformer-style language model (an encoder-only causal model) with end-to-end pieces for tokenization, model definition, training glue, and a terminal chat/inference script. It’s oriented toward experimentation and local training/inference (code and strings in Portuguese).

### Stack
- **Language(s):** Python
- **Framework / runtime:** PyTorch (torch + torch.nn)
- **Notable libraries:** torch, torch.nn.functional, numpy, json

## How it's organized
Top-level tree (annotated):
```text
README.md                    tiny repo README
inferencia/                  inference / chat CLI (loads model + vocab, generates text)
modelo/                      model definition and dataset class (Encoder_model, TextDataset)
t-m-i/                       small training / demo script that instantiates the model
tokenizador/                 tokenizer scripts, artifacts, raw data, dataset helpers
treino/                      training checkpoints / config (referenced by inferencia default)
wiki_scraper/                scraper for building dataset source material
```

How it fits together:
- Tokenization (tokenizador/) prepares raw text into a vocabulary and dataset artifacts. The tokenizer output directory (vocab.json) is read by the inference script.
- Model definition lives in modelo/ (Encoder_model, a small Transformer encoder layer with a linear head). Training/demo code in t-m-i/ shows how the model is instantiated and printed; the real training/testing checkpoints and a JSON config are expected under treino/checkpoints (the inference script defaults to treino/checkpoints/config.json).
- The inference entrypoint is inferencia/main.py — a CLI that loads the model state dict and vocab, encodes prompts, samples next tokens, and supports a one-shot or interactive chat mode.

## How to run it
Prerequisites: Python, PyTorch (CUDA optional). From a fresh clone, shortest paths:

- Run the tokenizer preparation (produces vocab/artifacts):
```bash
python tokenizador/main.py
```

- Run the demo/training script (prints/instantiates model):
```bash
python t-m-i/main.py
```

- Run inference / chat (defaults to treino/checkpoints/config.json):
```bash
python inferencia/main.py                # launches interactive chat
python inferencia/main.py --once "Olá"   # one-shot generation
python inferencia/main.py --prompt-ids 12,34,56 --max-new-tokens 50
```

Notes:
- inferencia/main.py expects a JSON config at treino/checkpoints/config.json (default) and a tokenizer output dir containing vocab.json (it loads vocab.json and model checkpoint path from that config).
- The repo contains a small modelo/encoder.py implementation (TransformerEncoder + causal mask) used by both training/demo and inference.
