## Simple Attention Mechanism
* Defines simple noramlisation functions of attention scores:
    * simple_normalisation: divides by the sum of inputs. Lacks the non-linearity introduced by the exponential function in softmax. (applies linear scaling meaning all values are treated proportionally). It doesn't handle negative values (for example input = [-1,0,1]).

    * naive_softmax: directly computes torch.exp(x) which can lead to numerical instability for large or small values.(torch.exp(x) can easily exceed the floating-point range, resulting in inf values.)

* Defines simple_att: which first calculates the attention scores w, uses PyTorch's optimal softmax implementation (which substracts the maximum value x_max from each element in x before exponentiation). Then multiplies the attention weights by the input to get the output.

---
## Self Attention Mechanism V1
* Defines a self attention class which inherits from nn.Module (It is a base class for all neural network modules in PyTorch. It provides core functionalities like tracking trainable parameters, automatic gradient computation during backpropagation and is required for integration with PyTorch's ecosystem(optimizers,loss functions..))

* Initializes trainable weight matrices (W_aquery,W_key,W_value). They project input embeddings into query,key and value spaces (these projections enable the model to learn different aspects of the input("what to look for","what to extract"))

* The forward method computes the queries,keys and values
    *queries: "what each token is looking for".
    *keys: what each token contains.
    *values: information to extract from each token.
Then calculates the attention scores, divides them with sqrt(d_key) for scaling and applies the softmax. The context vector is obtained by multiplying the attention weights with the values.

*Important note: why we scale by the sqrt(d_key) ? :
    * The dot product between queries and keys grows with the dimensionality of the keys(d_key).
    * For random vectors with mean 0 and variance 1, the dot product Q.K has mean 0 and variance d_k.
    * var(Q.K) = d_k
    * since var(a.X) = a^2*var(X)
    * then var(Q.K/sqrt(d_k)) = var(Q.K)/d_k which will close to one.

---
## Self Attention Mechanism V2

* In v1, we manually defined weight matrices as nn.Parameter.
* In v2, we used nn.Linear which uses PyTorch's default initialization which is more stable than the raw torch.rand()
* The qkv_bias parameter controls wether the query Q,K and V projections include a bias term.

---
## Causal Self Attention Mechanism

* Adds a dropout rate that regularizes weights to prevent overfitting.
* Defines a mask by creating an upper triangular matrix with -inf above the diagonal. This ensures that each token only attends to past tokens. The mask is applied before the softmax + scaling.

* Note on dimensions:
    * b,num_tokens,d_in = x.shape [batch_size,sequence_length,input_dim]
    * W_q,W_v,W_k have the shape: [b,d_in,d_out]
    * keys,values,queries have the shape: [b,num_tokens,d_out]
    * keys.transpose(1,2) has the shape: [b,d_out,num_tokens]
    * attn_scores has the shape: [b,num_tokens,num_tokens]
    * context_vec has the shape: [b,n_tokens,d_out]

---
## Multi Head Attention Mechanism

* Implements multi-head self-attention with causal masking
    *d_in: input dimension
    *d_out: output dimension (must be divisible by num_heads)
    *context_length:(maximum sequence length)
    *dropout: dropout rate for attention weights
    *num_heads: number of parallel attention heads
    *qkv_bias: wether to include bias in Q/K/V projecstions

* Splits input into multiple "heads"
* Projects each head to Q,K,V
* Computes scaled dot-product attention per head.
* Concatenates head outputs and applies a final linear projection.

* Note on Dimensions:
    *b,num_tokens,d_in = x.shape [b,num_tokens,d_in]
    *W_query,W_value,W_key have the shape [b,d_in,d_out]
    *keys,queries,values have the shape [b,num_tokens,d_out]
    *The we split into multiple heads d_out->(num_heads,head_dim)
    * keys,queries,values will have the shape [b,num_tokens,num_heads,head_dim]
    * Since we want to calculate the attention scores for each head, we need to group by num_heads
    * keys.transpose(1,2) has the shape [b,num_heads,num_tokens,head_dim]
    * queries.transpose(1,2) has the shape [b,num_heads,num_tokens,head_dim]
    * values.transpose(1,2) has the shape [b,num_heads,num_tokens,head_dim]
    * keys.transpose(2,3) has the shape [b,num_heads,head_dim,num_tokens]
    * attn_scores has the shape [b,num_heads,num_tokens,num_tokens]
    * attn_weights obtained after applying causal mask,softmax+scaling and the dropout has the shape: [b,num_heads,num_tokens,num_tokens].
    * context_vec has the shape [b,num_tokens,num_heads,head_dim]
    * concatenate heads + output projection, context_vec will have the shape [b,num_tokens,d_out]


