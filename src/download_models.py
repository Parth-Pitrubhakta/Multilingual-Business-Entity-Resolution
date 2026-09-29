"""Download the four pretrained checkpoints (public Hugging Face hub, pinned revisions) into a local folder.

Only model weights / tokenizers are downloaded; no data. Licences and sizes (checked on the model cards):
  xlm-roberta-base   FacebookAI/xlm-roberta-base   MIT         278,885,778 parameters
  xlm-roberta-large  FacebookAI/xlm-roberta-large  MIT         561,192,082 parameters
  mdeberta-v3-base   microsoft/mdeberta-v3-base    MIT         278,810,113 parameters (as fine-tuned)
  Qwen2.5-1.5B       Qwen/Qwen2.5-1.5B             Apache-2.0  1,543,714,304 parameters

usage: python download_models.py <out_dir>
"""
import os
import sys

MODELS = {
    "xlm-roberta-base": ("FacebookAI/xlm-roberta-base", "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
                         ["config.json", "tokenizer.json", "tokenizer_config.json", "sentencepiece.bpe.model", "model.safetensors"]),
    "xlm-roberta-large": ("FacebookAI/xlm-roberta-large", "c23d21b0620b635a76227c604d44e43a9f0ee389",
                          ["config.json", "tokenizer.json", "tokenizer_config.json", "sentencepiece.bpe.model", "model.safetensors"]),
    "mdeberta-v3-base": ("microsoft/mdeberta-v3-base", "a0484667b22365f84929a935b5e50a51f71f159d",
                         ["config.json", "tokenizer_config.json", "spm.model", "pytorch_model.bin"]),
    "Qwen2.5-1.5B": ("Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323",
                     ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                      "merges.txt", "model.safetensors", "LICENSE"]),
}


def download(out_dir):
    from huggingface_hub import snapshot_download
    for name, (repo, rev, files) in MODELS.items():
        path = os.path.join(out_dir, name)
        if all(os.path.exists(os.path.join(path, f)) for f in files):
            print(f"{name}: present ({path})", flush=True)
            continue
        snapshot_download(repo_id=repo, revision=rev, local_dir=path, allow_patterns=files)
        print(f"{name}: {repo}@{rev[:12]} -> {path}", flush=True)


if __name__ == "__main__":
    download(sys.argv[1])
