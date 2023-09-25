import torch.nn.functional as F
import torch
from pdf_to_md.settings import settings

TOKEN_BATCH_SIZE = 1750
TOKEN_BATCH_BACKUP = 100
DELIMITER = [13378, 20625, 46685]  # This is the sequence Cleaned Extract

prompt_template = """
Text Extract

{extract}

======
Cleaned Extract

""".strip()

def top_p(logits, temperature=.1, top_p=.95, filter_value=-float('Inf')):
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p

    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value

    probabilities = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probabilities, 1)
    return next_token


def generate(sequence, attention_mask, model, tokenizer, max_tokens=3900):
    sequence = sequence.to(settings.MODEL_DEVICE)
    attention_mask = attention_mask.to(settings.MODEL_DEVICE)
    past_key_values = None
    finished = [False] * sequence.shape[0]
    completions = [""] * sequence.shape[0]

    with torch.inference_mode():
        for i in range(max_tokens):
            pred = model(sequence, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True, return_dict=True)
            past_key_values = pred.past_key_values
            logits = pred.logits[:,-1,:]
            new_sequence = []
            for j in range(logits.shape[0]):
                token = top_p(logits[j])
                if token.item() == tokenizer.eos_token_id or token.item() == tokenizer.pad_token_id:
                    finished[j] = True

                last_token = token.reshape(-1,1).to(torch.long)
                new_sequence.append(last_token[0,0])

                if not finished[j]:
                    completions[j] += tokenizer.decode(last_token[0,0], skip_special_tokens=False)

            attention_mask = torch.hstack((attention_mask, torch.ones(attention_mask.shape[0], 1)))
            sequence = torch.tensor(new_sequence).reshape(-1,1)
            if sum(finished) == len(finished):
                break
    return completions


def create_prompts(text, tokenizer):
    text = text.encode('utf-8', errors='ignore').decode('utf-8').replace('\ufffd', ' ')
    tokenized = tokenizer(text, return_tensors="pt", padding=False, max_length=1e8, truncation=False)["input_ids"][0]

    extracts = []
    start = 0
    while start < tokenized.shape[0]:
        end = min(start + TOKEN_BATCH_SIZE - TOKEN_BATCH_BACKUP, tokenized.shape[0])
        while end < min(start + TOKEN_BATCH_SIZE, tokenized.shape[0]):
            next_tokens = tokenized[(end-2):end]
            decoded = tokenizer.decode(next_tokens)

            for split_delimiter in ["\n\n", ". ", "! ", "? ", "}\n", ":\n", ")\n", ".\n", "!\n", "?\n"]:
                if split_delimiter in decoded:
                    break

            end += 1

        batch = tokenized[start:end]
        start = end

        extract = tokenizer.decode(batch)
        extract = prompt_template.replace("{extract}", extract)
        extracts.append(tokenizer.bos_token + extract)
    return extracts


def clean_data(texts, model, tokenizer):
    tokenized = tokenizer(texts, return_tensors="pt", max_length=2000, truncation=False, padding="max_length")
    tokenized["input_ids"] = tokenized["input_ids"].to(torch.long)
    tokenized["attention_mask"] = tokenized["attention_mask"].to(torch.long)

    texts = generate(tokenized["input_ids"], tokenized["attention_mask"], model, tokenizer)
    return texts


def clean_text(text: str, model, tokenizer):
    prompts = create_prompts(text, tokenizer)
    cleaned = []
    for i in range(0, len(prompts), settings.BATCH_SIZE):
        batch = prompts[i:i+settings.BATCH_SIZE]
        cleaned_batch = clean_data(batch, model, tokenizer)
        cleaned.extend(cleaned_batch)
    return cleaned