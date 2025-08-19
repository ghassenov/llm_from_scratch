import re

class SimpleTokenizerV1:
    def __init__(self,vocab):
        self.str_to_int = vocab # stores the vocabulary as a class attribute 
        self.int_to_str = {i:s for s,i in vocab.items()} #inverse mapping of vocab
        
    def encode(self,text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)',text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self,ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text