import os
from datetime import datetime

import compyute as cp
import requests
from compyute import nn
from compyute.nn.utils.tensorboard import SummaryWriter
from simple_tokenizers import CharacterTokenizer

from transformer.attention_funcs import get_causal_mask
from transformer.gpt import GPTTransformer


def main() -> None:
    cp.random.set_seed(1337)
    device = cp.cuda

    # hyperparameters
    context_length = 256
    embed_dims = 384
    n_heads = 6
    n_blocks = 6
    batch_size = 64

    # training parameters
    step = 1
    max_steps = 10000
    label = "transformer_shakespeare_6"
    val_interval = 250
    checkpoint_interal = 500

    # load data
    DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(DATA_URL)
    data = response.text

    # tokenize data
    chars = sorted(list(set(data)))
    tokenizer = CharacterTokenizer()
    tokenizer.vocab = {i: c for i, c in enumerate(chars)}
    tokenizer.ivocab = {c: i for i, c in enumerate(chars)}

    # prepare data
    data_enc = cp.tensor(tokenizer.encode(data), dtype=cp.int32)
    X = cp.stack(
        [
            data_enc[i * context_length : i * context_length + context_length]
            for i in range(len(data_enc) // context_length)
        ]
    )
    y = cp.stack(
        [
            data_enc[i * context_length + 1 : i * context_length + context_length + 1]
            for i in range(len(data_enc) // context_length)
        ]
    )

    n = int(len(X) * 0.9)
    X_train = X.to_int()[:n]
    y_train = y.to_int()[:n]
    X_val = X.to_int()[n:]
    y_val = y.to_int()[n:]

    # create model
    mask = get_causal_mask(context_length)

    model = GPTTransformer(
        n_embeds=tokenizer.vocab_size,
        embed_dim=embed_dims,
        mlp_channels=4 * embed_dims,
        n_heads=n_heads,
        n_blocks=n_blocks,
        max_context_len=context_length,
        mask=mask,
        dropout=0.2,
    )
    model.to_device(device)

    # training
    train_dl = nn.utils.Dataloader((X_train, y_train), batch_size, device)
    val_dl = nn.utils.Dataloader((X_val, y_val), batch_size, device, False)
    loss_fn = nn.CrossEntropyLoss()
    optim = nn.optimizers.AdamW(model.get_parameters(), lr=3e-4)

    # load from checkpoint
    if step > 1:
        checkpoint = cp.load(f"{label}_{step}.cp")
        model.load_state_dict(checkpoint["model"], target_device=device)
        optim.load_state_dict(checkpoint["optim"], target_device=device)

    # create tensorboard logging directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logdir = f"./runs/{label}_{timestamp}/"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(log_dir=logdir)

    model.training()
    while step < max_steps:

        for x, y in train_dl():

            # training
            y_pred = model(x)
            loss = loss_fn(y_pred, y).item()
            loss_grads = loss_fn.backward()
            model.backward(loss_grads)

            optim.step()
            optim.reset_grads()
            writer.add_scalar("train/loss", loss, step)

            # validation
            if step > 1 and step % val_interval == 0:
                model.inference()
                val_loss = 0.0
                for x_val, y_val in val_dl():
                    y_pred = model(x_val)
                    val_loss += loss_fn(y_pred, y_val).item()
                val_loss /= len(val_dl)
                writer.add_scalar("val/loss", val_loss, step)

                model.training()

            # save checkpoints
            if step > 1 and step % checkpoint_interal == 0:
                model_state = model.get_state_dict()
                optim_state = optim.get_state_dict()
                state_dict = {"model": model_state, "optim": optim_state}
                checkpoint_name = f"{label}_{step}.cp"
                cp.save(state_dict, checkpoint_name)

            if step == max_steps:
                break
            step += 1
            print(f"{step=}", end="\r")


if __name__ == "__main__":
    main()
