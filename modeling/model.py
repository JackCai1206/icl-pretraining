from typing import Tuple
from torch import FloatTensor, LongTensor
from torch.nn import CrossEntropyLoss
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

class SuperpositionModel(GPTNeoXForCausalLM):
    def __init__(self, config: GPTNeoXConfig):
        super().__init__(config)
    
    def forward(
        self, 
        input_ids: LongTensor | None = None, 
        attention_mask: FloatTensor | None = None, 
        *args,
        labels: dict[float, LongTensor] | None = None, 
        **kwargs
    ) -> Tuple | CausalLMOutputWithPast:
        lm_loss = 0
        loss_fct = CrossEntropyLoss()
        for md, answer in labels.items():
            full_input = torch.cat(input_ids, answer)
            output = super().forward(full_input, attention_mask=attention_mask, *args, labels=None, output_hidden_states=True, **kwargs)
            lm_logits = output.logits
            # move labels to correct device to enable model parallelism
            full_labels = full_labels.to(lm_logits.device)
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss += loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
