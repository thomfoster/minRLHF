# actor.py
import torch
import torch.nn as nn
from torch.nn.functional import pad
from typing import List
from torch.distributions import Categorical

from .utils import logical_or_without_broadcasting

class Actor:
    def __init__(
        self, model, 
        pad_token_id: int, 
        sample_during_generation: bool = True, 
        sample_temperature: float = 1.0, 
        generation_max_length: int = 1024
    ):
        """
        Args:
            model (nn.Module): The language model used to generate new samples. 
                Model is required to implement .generate and .forward methods. 
            
        """
        self.model = model
        self.pad_token_id = pad_token_id
        self.sample_during_generation = sample_during_generation
        self.sample_temperature = sample_temperature
        self.generation_max_length = generation_max_length
        
    def to(self, device):
        self.model = self.model.to(device)
        
    @property
    def device(self):
        return self.model.device

    def get_rollouts(self, input_ids, input_mask):
        """
        Args: 
            input_ids: Tensor(batch, input_seq_len), 
            input_mask: Tensor(batch, input_seq_len)
            
        Returns: 
            output_ids: Tensor(batch, output_seq_len), 
            output_mask: Tensor(batch, output_seq_len)
        """
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=input_mask,
                pad_token_id=self.pad_token_id,
                do_sample=self.sample_during_generation,
                temperature=self.sample_temperature,
                max_length=self.generation_max_length
            )
            
        # pad output_ids to be (batch, max_length) in case all completions stopped early
        n = self.generation_max_length - output_ids.shape[1]
        output_ids = pad(output_ids, (0, n), value=self.pad_token_id)
        
        # Generate output_mask procedurally (must be a vectorised way)
        output_mask = torch.zeros_like(output_ids)

        start_idx = input_ids.shape[1]
        for i in range(output_ids.shape[0]):
            for j in range(start_idx, output_ids.shape[1]):
                if output_ids[i, j] != 50256:
                    output_mask[i,j] = 1
                else:
                    # Encountered first eos_token
                    # We still set the mask to 1 for this one because the model `chose` to do it
                    # If model was cutoff by max_length we won't get an eos_token
                    output_mask[i,j] = 1
                    break
        
        return output_ids, output_mask
        
    
    def get_logits(self, output_ids, input_mask, output_mask):
        """
        Inputs: output_ids: Tensor(batch, output_seq), input_mask: Tensor(batch, input_seq), output_mask: Tensor(batch, output_seq)
        Outputs: logprobs: Tensor(batch, output_seq), logprobs_mask: Tensor(batch, output_seq)
        """
        # Pad input mask to same size as output_mask and OR
        # OR'd mask is now all non padding / eos_tokens
        ord_mask = logical_or_without_broadcasting(input_mask, output_mask)
        
        # Generate logits by passing through model
        outputs = self.model(output_ids, attention_mask=ord_mask)
        
        # logits[i, j, k] = outputs.logits[i, j-1, k] for j>0
        # logits[i, 0, k] = 1e10 for k = output_ids[i, 0]
        # logits[i, 0, k'] = 0 otherwise
        logits = torch.empty_like(outputs.logits)
        logits[:, 1:, :] = outputs.logits[:, :-1, :]
        logits[
            list(range(output_ids.shape[0])),
            0,
            output_ids[:, 0]
        ] = 1e10
        
        # Collapse logits final dim from vocab -> 1 by
        # selecting the logit for the token we used
        pi = Categorical(logits=logits)
        logprobs = pi.log_prob(output_ids)
        
        # logprobs[i, j] is now the log prob of generating token output_ids[i,j]
        # logprobs[:, 0] = 0 and shouldn't be used
        # output_mask gets around this and the "input_ids"
        # using logprobs.masked_select(output_mask.to(torch.bool)) will select all valid logprobs (and flatten to avoid being a ragged tensor)
        return logprobs, pi