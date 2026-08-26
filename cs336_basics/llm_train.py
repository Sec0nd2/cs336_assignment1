import argparse
import math
import os
import time

import numpy as np
import numpy.typing as npt
import torch

from cs336_basics.Transformer import (
    Transformer_lm,
    AdamW,
    Get_batch,
    cross_entropy,
    Gradient_clipping,
    Lr_cosine_schedule,
    save_checkpoint,
    load_checkpoint,
)

def val_loss(
        model: torch.nn.Module,
        dataset: npt.NDArray,
        batch_size: int,
        context_length: int,
        val_iters: int,
        device: torch.device | None = None,
):
    model.eval()

    losses = []

    with torch.no_grad():
        for _ in range(val_iters):
            inputs, targets = Get_batch(dataset, batch_size, context_length, device=device)
            outputs = model(inputs)

            loss = cross_entropy(outputs, targets)

            losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)

def train(args):

    train_data = np.memmap(
        args.train_data,
        dtype=np.uint16,
        mode = 'r'
    )

    val_data = np.memmap(
        args.val_data,
        dtype=np.uint16,
        mode = 'r'
    )

    model = Transformer_lm(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta, device=args.device, dtype=torch.float32)

    model.train()

    optimizer = AdamW(model.parameters(), args.max_lr, (args.beta1, args.beta2), args.eps, args.weight_decay)

    start_iter = 0
    if args.resume is not None:
        start_iter = load_checkpoint(
            args.resume,
            model,
            optimizer
        ) + 1
        print(f"resume form iteration: {start_iter}")

    start_time = time.time()
    log_file = open(args.log_path, "w")
    log_file.write("step,time_sec,train_loss,val_loss,lr\n")

   # inputs, targets = Get_batch(train_data, args.batch_size, args.context_length, args.device)

    for it in range(start_iter, args.max_iter):

        lr = Lr_cosine_schedule(it, args.max_lr, args.min_lr, args.warm_iters, args.cosine_cycle_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        
        inputs, targets = Get_batch(train_data, args.batch_size, args.context_length, args.device)
        outputs = model(inputs)

        loss = cross_entropy(outputs, targets)

        optimizer.zero_grad()

        loss.backward()

        Gradient_clipping(model.parameters(), args.max_l2_norm)

        optimizer.step()

        if it % args.log_interval == 0:
            print(f"iter: {it:6d} | "f"train loss: {loss.item(): .4f} | "f"lr: {lr: .6e}")

        if it % args.eval_interval == 0 or it == args.max_iter - 1:
            v_loss = val_loss(model, val_data, args.batch_size, args.context_length, args.eval_iters, args.device)
            elapsed = time.time() - start_time

            
            print(f"iter: {it:6d} | "f"validation loss: {v_loss: .4f} | "f"time: {elapsed: .2f}")
            log_file.write(f"{it},{elapsed},{loss.item()},{v_loss},{lr}\n")
            log_file.flush()

        if (
            args.checkpoint_path is not None
            and it > 0
            and it % args.checkpoint_interval == 0
        ):
            save_checkpoint(model, optimizer, it, args.checkpoint_path)


    if args.checkpoint_path is not None:
        save_checkpoint(model, optimizer, args.max_iter - 1, args.checkpoint_path)



def parse_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True)

    # model
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)

    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    # optimization
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)

    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)

    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)

    parser.add_argument("--max-l2-norm", type=float, default=1.0)

    # training
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--warm-iters", type=int, default=500)
    parser.add_argument("--cosine-cycle-iters", type=int, default=5000)

    # logging
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--log-path", type=str, default="outputs/log_save.csv")

    # checkpoint
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None)

    # device
    parser.add_argument(
        "--device",
        type=str,
        default=get_default_device(),
    )

    return parser.parse_args()

def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

if __name__ == "__main__":
    args = parse_args()
    train(args)




  