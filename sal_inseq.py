import transformers
import torch
import os
import numpy as np
import pandas as pd
from typing import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from datasets import load_dataset

import inseq
from inseq.attr.step_functions import get_step_function

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    model.to(DEVICE)

    return model, tokenizer


model, tokenizer = load_model("textattack/roberta-base-SST-2")
dataset = load_dataset("gpt3mix/sst2")


def encode(batch):
    encoded = tokenizer(
        batch["text"],
        padding="longest",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(DEVICE)

    # Invert labels (0 <> 1), because it's the other way around somehow for roberta
    batch['label'] = (~torch.tensor(batch['label']).bool()).long()
    batch['text'] = [
        tokenizer.tokenize(sen, add_special_tokens=True)
        for sen in batch['text']
    ]

    return {**encoded, **batch}


dataset.set_transform(encode)

#orig_embs = model.roberta.embeddings.word_embeddings.weight.data
#new_embs = torch.cat((orig_embs, torch.zeros(1, orig_embs.shape[-1]).to(DEVICE)))
#model.roberta.embeddings.word_embeddings.weight.data = new_embs
#model.roberta.embeddings.word_embeddings.num_embeddings += 1
#
#zero_id = torch.tensor(50265).to(DEVICE)
#
#model.roberta.embeddings.word_embeddings(zero_id)




sen = "I wish I liked it more, although I did not dislike it"
item = {
        "input_ids": tokenizer(sen, return_tensors='pt').to(DEVICE)['input_ids'].squeeze(),
        "text": tokenizer.tokenize(sen, add_special_tokens=True),
        "label": torch.tensor(1),
    }

input_ids = item['input_ids']
print(input_ids)
sen_len = len(item['text'])

inseq = inseq.load_model(model, "saliency")
fmt = inseq.formatter
batch = fmt.prepare_inputs_for_attribution(inseq.attribution_method.attribution_model, [sen])
#print(batch)
#attributed_fn = get_step_function(inseq.attribution_method.attribution_model.default_attributed_fn_id)
attributed_fn = get_step_function("logit")
main_args = fmt.format_attribution_args(batch=batch, target_ids=item['label'], attributed_fn=attributed_fn)
#print("MAIN ARGS:", main_args['inputs'][0].shape)
with torch.no_grad():
    result = inseq.attribution_method.attribute_step(main_args)
print("RESULT:", result)
saliency_attributions = result.target_attributions.sum(-1).squeeze()
print(list(zip(item['text'], saliency_attributions.detach().cpu().numpy())))



