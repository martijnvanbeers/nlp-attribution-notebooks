import transformers
import torch
import os
import numpy as np
import pandas as pd
from typing import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from datasets import load_dataset

from abc import abstractmethod

from captum.attr import visualization
from captum.attr import IntegratedGradients


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

orig_embs = model.roberta.embeddings.word_embeddings.weight.data
new_embs = torch.cat((orig_embs, torch.zeros(1, orig_embs.shape[-1]).to(DEVICE)))
model.roberta.embeddings.word_embeddings.weight.data = new_embs
model.roberta.embeddings.word_embeddings.num_embeddings += 1

zero_id = torch.tensor(50265).to(DEVICE)

model.roberta.embeddings.word_embeddings(zero_id)


class FeatureAttributor:
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def attribute(
        self,
        input_ids: torch.Tensor,
        baseline_ids: torch.Tensor,
        target: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Computes the attributions for a single input.

        Parameters
        ----------
        input_ids : torch.Tensor
            shape: (batch_size, sen_len)
        baseline_ids : torch.Tensor
            shape: (batch_size, sen_len)
        target : int
            Output class that is being explained

        Returns
        -------
        attributions : torch.Tensor
            shape: (sen_len,)
        """
        pass

    def logits(self, input_ids, target=None):
        squeeze_output = input_ids.ndim == 1
        if squeeze_output:
            input_ids = input_ids.unsqueeze(0)

        with torch.no_grad():
            logits = self.model(input_ids).logits[:, target]

        if squeeze_output:
            logits = logits.squeeze()

        return logits

def model_forward(inputs_embeds):
    return model(inputs_embeds=inputs_embeds).logits


class IGAttributor(FeatureAttributor):
    def attribute(
        self,
        input_ids: torch.Tensor,
        baseline_ids: torch.Tensor,
        target: int,
        n_steps: int = 50,
    ) -> torch.Tensor:
        ### Start Student ###
        ig = IntegratedGradients(model_forward)

        inputs_embeds = model.roberta.embeddings.word_embeddings(input_ids.unsqueeze(0))
        baseline_embeds = model.roberta.embeddings.word_embeddings(baseline_ids.unsqueeze(0))

        ig_attributions = ig.attribute(
            inputs_embeds,
            target=target,
            baselines=baseline_embeds,
            n_steps=n_steps
        )
        ig_attributions = ig_attributions.sum(-1).squeeze()

        return ig_attributions
        ### End Student ###

sen = "I wish I liked it more, although I did not dislike it"
item = {
        "input_ids": tokenizer(sen, return_tensors='pt').to(DEVICE)['input_ids'].squeeze(),
        "text": tokenizer.tokenize(sen, add_special_tokens=True),
        "label": torch.tensor(1),
    }

input_ids = item['input_ids']
sen_len = len(item['text'])

baseline_id = tokenizer.unk_token_id
baseline_ids = torch.tensor([baseline_id] * sen_len, device=DEVICE)
label = item['label'].item()

ig_attributor = IGAttributor(model)
ig_attributions = ig_attributor.attribute(input_ids, baseline_ids, label)

print(ig_attributions)
