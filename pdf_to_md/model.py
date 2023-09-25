from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, AutoTokenizer
import torch
from pdf_to_md.settings import settings

MODEL = "vikp/cleaner"
TOKENIZER = "EleutherAI/pythia-1.4b-deduped"
PAD_TOKEN = "<|padding|>"
PAD_TOKEN_ID = 1
TRAIN_PCT = .95
SEQUENCE_LENGTH_MAX = 4096


def get_model():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, padding_side='left')
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.pad_token_id = PAD_TOKEN_ID

    rope_scaling = {"type": "linear", "factor": 4.0}
    config = GPTNeoXConfig.from_pretrained(MODEL)
    config.pad_token_id = PAD_TOKEN_ID
    config.rope_scaling = rope_scaling

    model = GPTNeoXForCausalLM.from_pretrained(MODEL, config=config).to(settings.MODEL_DEVICE)
    model.eval()

    return model, tokenizer