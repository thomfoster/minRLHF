# critic.py
from .utils import logical_or_without_broadcasting


class Critic:
    def __init__(self, model):
        self.model = model
        
    def to(self, device):
        self.model = self.model.to(device)
        
    @property
    def device(self):
        return self.model.device
        
    def get_value_estimates(self, output_ids, input_mask, output_mask):
        """
        Inputs: output_ids: Tensor(batch, output_seq), input_mask: Tensor(batch, input_seq), output_mask: Tensor(batch, output_seq)
        Outputs: value_estimates: Tensor(batch, output_seq), value_estimates_mask: Tensor(batch, output_seq)
        """
        attention_mask = logical_or_without_broadcasting(input_mask, output_mask)
        outputs = self.model(output_ids, attention_mask=attention_mask)
        return outputs.logits.squeeze()