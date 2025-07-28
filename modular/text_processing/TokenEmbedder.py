import torch

def token_embedder(inputs,max_length,vocab_size,output_dim):
    token_embedding_layer = torch.nn.Embedding(vocab_size,output_dim)
    token_embeddings = token_embedding_layer(inputs)
    context_length = max_length
    
    pos_embedding_layer = torch.nn.Embedding(context_length,output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    input_embeddings = token_embeddings + pos_embeddings
    return input_embeddings