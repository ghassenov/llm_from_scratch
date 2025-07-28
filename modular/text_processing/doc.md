## SimpleTokenizerV1
- Stores the vocabulary as a class attribute for access in the encode/decode methods
- Creates an inverse vocabulary that maps token IDs back to original text tokens
- Processes input text into token IDs (using regex) via encode method
- Converts token IDs back into text via decode method.

---

## SimpleTokenizerV2
- Improved version of SimpleTokenizerV2 that replaces unkown words by <|unk|> tokens.

---

## DataLoader 
- Defines a GPTDatasetV1 class that inherits from PyTorch Dataset, tokenizes the input text,defines input and target chunks using sliding window technique (These pairs are used for autogregressive training - predicting the next token) then stores them,as tensors, inside list class variables.
- The class also defines two more methods len and getitem required for PyTorch Dataset comptability.
- The create_dataloader_v1 function takes as input:
    - txt: input text
    - batch_sizes: number of sequences per batch
    - max_length: maximum number of tokens in each input sequence.
    - stride: sliding window step
    - shuffle: to randomize batch order
    - drop_last: discard incompleted batches
    - num_workers: CPU threads for parallel loading
This function uses GPT-2's tokenizer, wraps the dataset to generate batches for training.

---

## TokenEmbedder
- Defines a function that combines token embeddings with positional embeddings (to give the model information about token order)