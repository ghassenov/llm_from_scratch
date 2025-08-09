import torch

def simple_normalisation(inputs):
    return inputs/inputs.sum()

def naive_softmax(x):
    return torch.exp(x)/torch.exp(x).sum(dim=-1)
    

def simple_att(inputs):
    # first step is to get the attension scores w
    attn_scores = inputs @ inputs.T
    # second step is to normalize the attention scores
    norm_att_weights = torch.softmax(attn_scores,dim=-1)
    # last step is to multiply the normalized attention scores with the inputs
    outputs = norm_att_weights @ inputs
    return outputs    

# Note: we specify dim=-1 because we apply the softmax using the last dimention of the inputs,
# so that the softmax is applied row by row (each row sums up to one)
