from typing import Any, Dict, List, Tuple
from torch._tensor import Tensor
from torch.nn.modules import Module
from transformers import Trainer
import torch
from torch.nn import CrossEntropyLoss

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        ans_prob = inputs.pop("ans_prob")
        B, A, P, _ = inputs['input_ids'].shape

        all_loss = 0
        all_outputs = []

        for ai in range(A):
            inp = {}
            for key in inputs:
                if key in ["input_ids", "attention_mask", 'labels']:
                    inp[key] = inputs[key].contiguous().view(B, A, -1)[:, ai, :]
                else:
                    inp[key] = inputs[key]

            if return_outputs: # this is for eval 
                loss, outputs = super().compute_loss(model, inp, return_outputs=True)
                logits = outputs.logits
            else:
                labels = inp.pop("labels")
                outputs = model(**inp, return_dict=True)
                logits = outputs.logits
                
                # Save past state if it exists
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index]
                
                @torch.compile
                def get_calibrated_loss(logits, labels, inputs=inputs, ans_prob=ans_prob):
                    # shift and apply loss mask before one-hot encoding
                    loss_mask = labels[..., 1:] != -100
                    labels = labels[..., 1:][loss_mask]
                    logits = logits[..., :-1, :][loss_mask]
                    labels = torch.nn.functional.one_hot(labels, num_classes=model.config.vocab_size).float()

                    if A > 1:
                        # find the branching position, which is the first token that is different across the mixing sequences
                        branch_mask = (inputs['labels'][:, 0:1, ...] == inputs['labels']).all(1, keepdim=True).int().diff() < 0
                        # .diff() will return length - 1, so pad
                        branch_mask = torch.nn.functional.pad(branch_mask, (1, 0), value=False) # (B, 1, P, _)
                        # apply mask to the inputs to get the branching tokens, there should be A tokens for each ICL example
                        branch_toks = inputs['input_ids'][branch_mask.expand_as(inputs['input_ids'])].view(B, A, P).transpose(1, 2).reshape(B * P, A)
                        # reshape the mask to apply to the labels
                        branch_mask = branch_mask.view(B, -1)[..., 1:][loss_mask]
                        labels[branch_mask] = labels[branch_mask].scatter_(-1, branch_toks, ans_prob.transpose(1, 2).reshape(B * P, A))

                    # Flatten the tokens
                    loss_fct = CrossEntropyLoss()
                    logits = logits.contiguous().view(-1, model.config.vocab_size)
                    labels = labels.contiguous().view(-1, model.config.vocab_size)
                    # Enable model parallelism
                    labels = labels.to(logits.device)
                    loss = loss_fct(logits, labels)
                    return loss

                loss = get_calibrated_loss(logits, labels)
                return loss # when training, we only need to compute the loss for the first sequence   

            all_loss += loss
            all_outputs.append(logits)

        all_outputs = torch.stack(all_outputs, dim=1)
        return (all_loss, (all_loss, all_outputs)) if return_outputs else all_loss

    def prediction_step(
        self,
        model: Module,
        inputs: Dict[str, Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: List[str] | None = None
    ) -> Tuple[Tensor | None]:
        ans_prob = inputs["ans_prob"]
        loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        return (loss, logits, ans_prob)
