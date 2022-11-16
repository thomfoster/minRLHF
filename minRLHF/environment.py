import torch
from abc import ABC
from .utils import logical_or_without_broadcasting

class Environment(ABC):
    def __init__(self, model_tokenizer, batch_size: int):
        self.tokenizer = model_tokenizer
        self.batch_size = batch_size
        
    def get_input_prompt(self):
        raise NotImplementedError()
    
    def score_generation(self, text: str):
        raise NotImplementedError()
        
    def reset(self):
        batch = [self.get_input_prompt() for _ in range(self.batch_size)]
        inputs = self.tokenizer(batch, truncate=True, padding=True, return_tensors='pt')
        return inputs.input_ids, inputs.attention_mask
        
    def get_rewards(self, output_ids, input_mask, output_mask):
        
        # Decode generations back into text
        full_mask = logical_or_without_broadcasting(input_mask, output_mask)
        texts = []
        for output_id, mask in zip(output_ids, full_mask):
            ids = output_id.masked_select(mask.to(torch.bool))
            texts.append(self.tokenizer.decode(ids, skip_special_tokens=True))
        
        # Score the completions
        scores = [self.score_generation(text) for text in texts]
        
        # Rewards[i, j] = 
        #   reward score    if j is last generated token of ith example
        #   0               otherwise
        idxs = -output_mask.flip(-1).argmax(-1) - 1 # index of last generated token # TODO: Make less cryptic
        rewards = torch.zeros_like(output_mask, dtype=torch.float32)
        rewards[list(range(rewards.shape[0])), idxs] = torch.as_tensor(scores)
        
        return rewards
        