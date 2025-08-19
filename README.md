# GPT2_from_scratch

### Project Overview
This project represents a complete from-scratch implementation of OpenAI's GPT-2 architecture, built through an iterative process of experimentation, testing, and modularization. I developed the entire pipeline—from raw text processing and tokenization to model architecture, training, evaluation, and weight loading—culminating in a functional GPT-2 implementation compatible with OpenAI's pre-trained weights.

---

The implementation follows a research-to-production workflow: I first explored concepts and tested implementations in Jupyter notebooks, then refined the successful approaches into clean, modular Python classes and functions in modular/ .

---

### Project Structure

```
LLM_FROM_SCRATCH/
│
├── data/
│   └── the-verdict.txt
│
├── modular/
│   ├── AttentionMechanisms/
│   │   ├── __init__.py
│   │   ├── CausalAttention.py
│   │   ├── doc.md
│   │   ├── MultiHeadAttention.py
│   │   ├── SelfAttention_v1.py
│   │   ├── SelfAttention_v2.py
│   │   └── SimpleAttention_mechanism.py
│   │
│   ├── GPTArchitecture/
│   │   ├── __init__.py
│   │   ├── doc.md
│   │   ├── DummyGPTModel.py
│   │   ├── FeedForwardBlock.py
│   │   ├── GPTModel.py
│   │   ├── LayerNormalization.py
│   │   ├── TextGeneration.py
│   │   └── TransformerBlock.py
│   │
│   ├── Pretraining/
│   │   ├── __init__.py
│   │   ├── doc.md
│   │   ├── LossMeasurement.py
│   │   └── ModelTraining.py
│   │
│   ├── TextProcessing/
│   │   ├── __init__.py
│   │   ├── DataLoader.py
│   │   ├── doc.md
│   │   ├── SimpleTokenizerV1.py
│   │   ├── SimpleTokenizerV2.py
│   │   └── TokenEmbedder.py
│   │
│   ├── __init__.py
│   ├── AssignWeights.py
│   └── LoadGPT2Weights.py
│
├── notebooks/
│   ├── __init__.py
│   ├── causal_self_attention.ipynb
│   ├── feed_forward_block.ipynb
│   ├── layer_normalization.ipynb
│   ├── llm_architecture_v1.ipynb
│   ├── llm_architecture_v2.ipynb
│   ├── loss_measurement.ipynb
│   ├── multi_head_attention.ipynb
│   ├── pretraining.ipynb
│   ├── sample_code.ipynb
│   ├── self_attention.ipynb
│   ├── shortcut_connections.ipynb
│   ├── simple_attention_mechanism.ipynb
│   ├── text_generation.ipynb
│   ├── text_processing.ipynb
│   └── transformer_block.ipynb
│
├── .gitignore
├── LICENSE
├── Pipfile
├── Pipfile.lock
└── README.md
```

---

## Key Achievements

**Complete Text Processing Pipeline**

Implemented different tokenization strategies and used Byte Pair Encoding(BPE) tokenizer which is compatible with GPT-2.

Created efficient data loading and batching mechanisms

Built vocabulary management and special tokens handling

**Attention Mechanism Implementations**

Built foundational to advanced attention systems from simple dot-product to sophisticated multi-head architectures

Implemented causal masking to enable autoregressive properties essential for GPT models

Created efficient multi-head processing with proper tensor reshaping and parallel computation

Added dropout regularization to attention weights for improved model generalization

Achieved OpenAI compatibility in attention patterns and output behavior

**GPT Architecture Implementation**

FeedForward Block: 

I implemented a position-wise feedforward network with GELU activation that expands the hidden dimension by 4x according to GPT-2 specifications.

Shortcut Connections: 

I achieved stable gradient flow in deep networks by coding residual skip connections around both attention and feedforward layers.

Dropout Regularization:

I implemented targeted dropout on attention weights and hidden layers to reduce overfitting and improve model generalization.

Transformer Block: 

I coded a complete transformer decoder block that integrates masked multi-head self-attention and feedforward networks with residual connections.

Text Generation: 

I coded multiple decoding strategies including temperature scaling and top-k sampling for diverse text generation.

GPTModel Class:

I built the complete GPT-2 architecture by stacking transformer blocks and implementing weight-tied embedding layers.

**Model training and evaluation**

Model Training Loop: 

I developed a full training loop with gradient accumulation, mixed precision support, and comprehensive logging.

Model evaluation:

I used multiple metrics to evaluate the LLM training including cross-entropy loss and perplexity.

**Public OpenAI GPT-2 weights integration**
I successfully downloaded OpenAI's pre-trained GPT-2 weights and correctly mapped them to my from-scratch implemented architecture.



