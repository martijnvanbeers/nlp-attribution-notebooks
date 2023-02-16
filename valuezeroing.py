### Import Libraries
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import torch

def compute_joint_attention(att_mat, layer_dim=0, add_residual=True):
    if add_residual:
        residual_att = np.eye(att_mat.shape[-1])[None,...]
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1)[...,None]
    else:
       aug_att_mat =  att_mat

    joint_attentions = np.zeros(aug_att_mat.shape)

    layers = joint_attentions.shape[layer_dim]
    dim_tup = tuple(0 if d == layer_dim else slice(None,None,None) for d in range(len(joint_attentions.shape) - 2))
    joint_attentions[dim_tup] = aug_att_mat[dim_tup]
    for i in np.arange(1,layers):
        dim_tup = tuple(i if d == layer_dim else slice(None,None,None) for d in range(len(joint_attentions.shape) - 2))
        prev_tup = tuple(i-1 if d == layer_dim else slice(None,None,None) for d in range(len(joint_attentions.shape) - 2))
        # matmul does dot product on the last two axes,
        joint_attentions[dim_tup] = np.matmul(aug_att_mat[dim_tup], joint_attentions[prev_tup])

    return joint_attentions

METRIC = 'cosine'
DISTANCE_FUNC = {
    'cosine': cosine_distances
}


def calculate_scores(config, model, MODEL_NAME, tokenizer, text):
    inputs = tokenizer(text, return_token_type_ids=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs['token_type_ids'],
                output_hidden_states=True,
                output_attentions=True
            )

    org_hidden_states = torch.stack(outputs['hidden_states']).squeeze(1)
    attentions = torch.stack(outputs['attentions']).squeeze(1)
    input_shape = inputs['input_ids'].size()
    batch_size, seq_length = input_shape

    score_matrix = np.zeros((config.num_hidden_layers, config.num_attention_heads, seq_length, seq_length))
    for l, layer_module in enumerate(getattr(model, MODEL_NAME).encoder.layer):
        for h in range(config.num_attention_heads):
            for t in range(seq_length):
                extended_blanking_attention_mask: torch.Tensor = getattr(model, MODEL_NAME).get_extended_attention_mask(inputs['attention_mask'], input_shape, model.device)
                with torch.no_grad():
                    layer_outputs = layer_module(org_hidden_states[l].unsqueeze(0), # previous layer's original output
                                                attention_mask=extended_blanking_attention_mask,
                                                output_attentions=False,
                                                zero_value_index=(h,t,),
                                                )
                hidden_states = layer_outputs[0].squeeze().detach().cpu().numpy()
                # compute similarity between original and new outputs
                # cosine
                x = hidden_states
                y = org_hidden_states[l+1].detach().cpu().numpy()

                distances = DISTANCE_FUNC[METRIC](x, y).diagonal()
                score_matrix[l, h, :, t] = distances


    valuezeroing_scores = score_matrix / np.sum(score_matrix, axis=-1, keepdims=True)
    rollout_valuezeroing_scores = compute_joint_attention(valuezeroing_scores, add_residual=False)
    return valuezeroing_scores, rollout_valuezeroing_scores, attentions

