import copy
from functools import partial
from itertools import islice
from time import sleep
from multiprocessing import Pool, Queue, Process
from threading import Thread
import string
import random

import numpy as np
import torch
from transformers import HfArgumentParser, LlamaConfig, LlamaForCausalLM, TrainingArguments, Trainer, EvalPrediction, set_seed
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset, IterableDataset
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple, cast

from MyTrainer import MyTrainer
from charactertokenizer import CharacterTokenizer
from utils import align_sentences

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

@dataclass
class ScriptArguments:
    task: Literal['bias_add_simple', 'bias_add_simple_9010', 'bias_add_simple_1090', "bias_add", "bias_add_9010", "bias_add_1090", "copy", "copy_9010", "copy_1090", "linear_regression"] = "bias_add"
    num_train: int = 10_000_000
    num_test: int = 100
    prompt_size: int = 100
    num_mix_train: int = 1
    force_unique_prompts: bool = True
    num_tasks: Optional[int] = None
    test_all_tasks: bool = False

    num_layers: int = 6
    hidden_size: int = 384
    num_attention_heads: int = 6
    max_position_embeddings: int = 2048

args, train_args = HfArgumentParser((ScriptArguments, TrainingArguments)).parse_args_into_dataclasses()
args = cast(ScriptArguments, args)
train_args = cast(TrainingArguments, train_args)
train_args.include_inputs_for_metrics = True
train_args.remove_unused_columns = False
train_args.run_name += f"_{args.task}_{args.num_layers}_{args.hidden_size}_{args.num_attention_heads}_mix_{args.num_mix_train}"
train_args.output_dir += f"/{train_args.run_name}"
if train_args.do_eval and not train_args.do_train:
    train_args.run_name = 'eval' + train_args.run_name
train_args.dataloader_num_workers = 4
if train_args.resume_from_checkpoint == 'True':
    try:
        train_args.resume_from_checkpoint = get_last_checkpoint(train_args.output_dir)
    except FileNotFoundError:
        train_args.resume_from_checkpoint = None

set_seed(train_args.seed)

vocab = string.ascii_letters + string.digits + string.punctuation + " \n"
tokenizer = CharacterTokenizer(vocab, args.max_position_embeddings)
tokenizer.padding_side = 'left'
tokenizer.sep_token = '>'
tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)

def sample_hidden_param(task: str, k: int, train=True):
    if task == "bias_add_simple":
        if train:
            return random.sample(list(range(0, 10)), k)
        else:
                return [2, 6]
    elif task == "bias_add_simple_9010":
        if train:
            if random.random() < 0.9:
                return random.sample(list(range(0, 5)), k)
            else:
                return random.sample(list(range(5, 10)), k)
        else:
            return [2, 6]
    elif task == "bias_add_simple_1090":
        if train:
            if random.random() < 0.1:
                return random.sample(list(range(0, 5)), k)
            else:
                return random.sample(list(range(5, 10)), k)
        else:
            return [2, 6]
    elif task == "bias_add":
        if train:
            return random.sample(list(range(-9, 0)) + list(range(1, 10)), k)
        else:
            return [-5, 5]
    elif task == "bias_add_9010":
        if train:
            if random.random() < 0.9:
                return random.sample(list(range(-9, 0)), k)
            else:
                return random.sample(list(range(1, 10)), k)
        else:
            return [-5, 5]
    elif task == "bias_add_1090":
        if train:
            if random.random() < 0.1:
                return random.sample(list(range(-9, 0)), k)
            else:
                return random.sample(list(range(1, 10)), k)
        else:
            return [-5, 5]
    elif task == "copy":
        if train:
            return random.sample(range(8), k)
        else:
            return [2, 6]
    elif task == 'copy_9010':
        if train:
            if random.random() < 0.9:
                return random.sample(range(0, 4), k)
            else:
                return random.sample(range(4, 8), k)
        else:
            return [2, 6]
    elif task == 'copy_1090':
        if train:
            if random.random() < 0.1:
                return random.sample(range(0, 4), k)
            else:
                return random.sample(range(4, 8), k)
        else:
            return [2, 6]
    elif task == "linear_regression":
        return random.random()

def get_composed_hp(task, hps):
    # composition is only defined for bias_add with 2 hidden params
    if task.startswith('bias_add_simple'):
        return [hp for hp in list(range(0, 10)) if hp not in hps]
    elif task.startswith("bias_add"):
        return [hp for hp in list(range(-9, 0)) + list(range(1, 10)) if hp not in hps]
    if task.startswith("copy"):
        return [hp for hp in range(8) if hp not in hps]
        # return None
    else:
        return None

args.num_tasks = 2 if not args.test_all_tasks else len(get_composed_hp(args.task, []))

def get_example(task: str, hidden_param: Any = None, ex_args: Optional[Any | Tuple[Any]] = None):
    if hidden_param is None:
        hidden_param = sample_hidden_param(args.task, 1)
    if task.startswith('bias_add_simple'):
        if ex_args is None:
            x1 = random.randint(0, 9)
            ex_args = x1
        else:
            x1 = ex_args
        y = str((x1 + hidden_param) % 10).rjust(1, '0')
        x1 = str(x1).rjust(1, '0')
        return f"{x1}>", f"{y}\n", hidden_param, ex_args
    elif task.startswith("bias_add"):
        if ex_args is None:
            x1 = random.randint(0, 99)
            x2 = random.randint(0, 99)
            ex_args = (x1, x2)
        else:
            x1, x2 = ex_args
        # y = str(x1 + x2 + hidden_param).rjust(3, '0')[::-1]
        y = str((x1 + x2 + hidden_param) % 100).rjust(2, '0')[::-1]
        x1_str = str(x1).rjust(2, '0')[::-1]
        x2_str = str(x2).rjust(2, '0')[::-1]
        return f"{x1_str}+{x2_str}>", f"{y}\n", hidden_param, ex_args
    elif task.startswith("copy"):
        if ex_args is None:
            ex_args = x = random.sample(string.ascii_letters, k=8)
        else:
            x = ex_args
        y = x[hidden_param]
        return f"{''.join(x)}>", f"{y}\n", hidden_param, ex_args
    elif task == "linear_regression":
        if ex_args is None:
            ex_args = x1 = round(random.random() * 100)
        else:
            x1 = ex_args
        y = round(hidden_param * x)
        return f"{x}>", f"{y}\n", hidden_param, ex_args
    else:
        raise ValueError(f"Invalid task: {task}")

def inner(i: int, args: ScriptArguments, mix_dist: Optional[Tuple[float]] = None, num_mix: Optional[int] = None, train: bool = True):
    assert num_mix is not None or mix_dist is not None
    if mix_dist is not None:
        num_mix = len(mix_dist)
    A = num_mix
    if mix_dist is None:
        upper = 1.0
        mix_dist = []
        for _ in range(num_mix - 1):
            mix_dist.append(random.uniform(0, upper))
            upper -= mix_dist[-1]
        mix_dist.append(upper)
        random.shuffle(mix_dist)
    context = [] # All the ICL examples
    hidden_param = sample_hidden_param(args.task, A, train=train) # make sure hidden param for two tasks are different
    task_ids = np.random.choice(A, size=args.prompt_size, p=mix_dist)
    task_ids = [[tid] + list(set(range(A)) - set([tid])) for tid in task_ids]

    ans_prob = [[1.0 / A] * A] # uniform prior
    tally = [0] * A
    for j, tids in enumerate(task_ids):
        examples = []
        ex_args = None
        for tid in tids:
            p, a, hidden_param[tid], ex_args = get_example(args.task, hidden_param[tid], ex_args=ex_args)
            examples.append((p, a, str(hidden_param[tid])))
        context.append(examples)
        tally[tids[0]] += 1 # "choose" the first task
        ap = [float(tally[tid]) / (j+1) for tid in tids]
        ans_prob.append(ap)

    found_prompts = False
    tries = 0
    prompts = [] # The last ambiguous example, make sure the answers are unique
    if args.test_all_tasks:
        comp_hps = get_composed_hp(args.task, hidden_param)
        if comp_hps is not None:
            hidden_param.extend(comp_hps)
    while not found_prompts:
        ex_args = None
        for hp in hidden_param:
            p, a, _, ex_args = get_example(args.task, hp, ex_args=ex_args)
            prompts.append((p, a, str(hp)))

        if len(set(p[1] for p in prompts)) == len(prompts) or not args.force_unique_prompts:
            found_prompts = True
        else:
            tries += 1
            prompts = []
            if tries > 1000: raise ValueError(f"Cannot find unique prompts after 1000 tries")

    return {"context": context, "prompts": prompts, "ans_prob": ans_prob}
    
def tokenize(batch, tokenizer: CharacterTokenizer, args: ScriptArguments, batched: bool = True, train: bool = True):
    if not batched:
        batch = {k : [v] for k, v in batch.items()}

    batch['ans_prob'] = torch.tensor(batch['ans_prob']).transpose(1, 2) # B x A x pz

    lines = []
    label_masks = []
    B = len(batch['prompts'])
    A = len(batch['prompts'][0]) if not train else len(batch['context'][0][0])
    P = args.prompt_size + 1
    for bi in range(B):
        for ai in range(A):
            all_examples = [c[ai if train else 0] for c in batch['context'][bi]] + [batch['prompts'][bi][ai]]
            lines.extend([ex[0] + ex[1] for ex in all_examples])
            label_masks.extend([[0] * len(ex[0]) + [1] * len(ex[1]) for ex in all_examples])
    label_masks = torch.tensor(label_masks)
    results = tokenizer(lines, truncation=False, padding="do_not_pad", add_special_tokens=False, return_token_type_ids=False, max_length=args.max_position_embeddings, return_tensors="pt")
    results['labels'] = results['input_ids'].clone()
    results['labels'][label_masks == 0] = -100
    for key in ['input_ids', 'attention_mask', 'labels']:
        results[key] = results[key].reshape(B, A, P, -1)
    results['ans_prob'] = batch['ans_prob']
    if not batched:
        for key in results:
            results[key] = results[key][0]
    return results

def data_generator(shards: List, args: ScriptArguments, mix_dist: Optional[Tuple[float]] = None, num_mix: Optional[int] = None, train=True):
    assert len(shards) == 1
    for i in shards[0]:
        yield inner(i, args=args, mix_dist=mix_dist, num_mix=num_mix, train=train)

# print(f"Trying prompt size: {args.prompt_size}")
# dat = next(data_generator(1, args, num_mix = 1))
# inp = tokenize(dat, tokenizer, args, batched=False, train=False)['input_ids']
# A, line_len = inp.view(inp.shape[0], -1).shape
# if line_len > args.max_position_embeddings:
#     raise ValueError(f"Input length {line_len} exceeds max length {args.max_position_embeddings}")

# import cProfile, pstats, io
# from pstats import SortKey
# pr = cProfile.Profile()
# pr.enable()

slice_size = args.num_train // train_args.dataloader_num_workers
train_dataset = IterableDataset.from_generator(
    data_generator,
    gen_kwargs={
        'shards': [range(i, i + slice_size) for i in range(0, args.num_train, slice_size)],
        "args": args, "num_mix": args.num_mix_train, "train": True
    }
).map(
    tokenize,
    batched=True,
    batch_size=1024,
    fn_kwargs={"tokenizer": tokenizer, "args": args, "batched": True, "train": True},
    remove_columns=['context', 'prompts']
)

# print(f'Cleaned up {train_dataset.cleanup_cache_files()} cache files')

print("Example from train dataset")
print(next(islice(train_dataset, 0, 1)))
print(tokenizer.batch_decode(next(islice(train_dataset, 0, 1))["input_ids"][0]))
# for i, batch in enumerate(islice(train_dataset, 0, 1000)):
#     if i == 1000:
#         break

# pr.disable()
# s = io.StringIO()
# sortby = SortKey.CUMULATIVE
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print(s.getvalue())
# breakpoint()

test_datasets = {}
for d in np.linspace(0, 1, 11):
    d = round(d, 3)
    test_datasets[str(round(d, 2))] = IterableDataset.from_generator(
        data_generator, 
        gen_kwargs={
            'shards': [range(args.num_test)],
            "args": args, "mix_dist": (d, 1-d), "train": False
        },
        # num_proc=10
    ).map(
        tokenize,
        batched=True,
        batch_size=128,
        fn_kwargs={"tokenizer": tokenizer, "args": args, "batched": True, "train": False},
        remove_columns=['context', 'prompts']
    )

print("Example from test dataset")
# print(next(islice(test_datasets['0.5'], 0, 1)))
print(tokenizer.batch_decode(next(islice(test_datasets['0.5'], 0, 1))["input_ids"].view(2, -1)))

model_config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,
    num_hidden_layers=args.num_layers,
    hidden_size=args.hidden_size,
    intermediate_size=args.hidden_size * 4,
    num_attention_heads=args.num_attention_heads,
    max_position_embeddings=args.max_position_embeddings,
    attn_implementation='flash_attention_2' if train_args.bf16 else 'sdpa'
)
model = LlamaForCausalLM(model_config)
print('Number of parameters:', model.num_parameters())

def compute_metrics(pred: EvalPrediction):
    B, A, P, _ = pred.inputs.shape
    pred.inputs = pred.inputs.reshape(B, A, -1)
    pred_ans_prob = np.zeros((B, A))
    for bi in range(B):
        for ai in range(A):
            sep_pos = np.nonzero(pred.inputs[bi, ai, :] == tokenizer.sep_token_id)[-1][-1]
            logits = pred.predictions[bi, ai, sep_pos:-1] # NOT [sep_pos + 1:] because logits are shifted by 1
            probs = torch.tensor(logits).softmax(dim=-1)
            probs[..., tokenizer.pad_token_id] = 1
            pred_ans_prob[bi, ai] = torch.gather(probs, -1, torch.tensor(pred.inputs[bi, ai, sep_pos + 1:, None])).prod().item()
            # pred_ans_prob[bi, ai] = torch.gather(probs, -1, torch.tensor(pred.inputs[bi, ai, sep_pos + 1:sep_pos + 2, None])).prod().item()
    pred_mean = pred_ans_prob.mean(axis=0)
    pred_var = pred_ans_prob.var(axis=0)
    ans_prob = pred.label_ids
    # kl_div = np.sum(ans_prob * np.log(ans_prob / pred_ans_prob))
    # metrics = {"kl_div": kl_div}
    metrics = {}
    for ai, (m, v) in enumerate(zip(pred_mean, pred_var)):
        metrics[f"ans_prob_mean_{ai}"] = m
        metrics[f"ans_prob_var_{ai}"] = v

    return metrics

trainer = MyTrainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=test_datasets,
    compute_metrics=compute_metrics
)

import wandb
wandb.init(project="icl-pretraining", entity="jackcai1206", name=train_args.run_name)
wandb.config.update(args, allow_val_change=True)

if train_args.do_train:
    if train_args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
    else:
        trainer.train()
if not train_args.do_train and train_args.do_eval:
    trainer._load_from_checkpoint(resume_from_checkpoint=train_args.resume_from_checkpoint)
    trainer.evaluate()