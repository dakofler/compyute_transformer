import os
from datetime import datetime

import compyute as cp
from compyute import nn
from compyute.nn.utils.tensorboard import SummaryWriter
from compyute.preprocessing.text import CharacterTokenizer
from transformer_s import Transformer, get_causal_mask

cp.random.set_seed(1337)
device = cp.cuda

block_size = 256
embed_dims = 384

batch_size = 64
mini_batch_size = 32
val_interval = 250
max_iter = 10
checkpoint_interal = 10


with open("data/tinyshakespeare.txt", "r") as f:
    data = f.read()


chars = sorted(list(set(data)))
tokenizer = CharacterTokenizer()
tokenizer.vocab = {i: c for i, c in enumerate(chars)}
tokenizer.ivocab = {c: i for i, c in enumerate(chars)}

data_enc = cp.tensor(tokenizer.encode(data), dtype=cp.int32)
X = cp.stack(
    [
        data_enc[i * block_size : i * block_size + block_size]
        for i in range(len(data_enc) // block_size)
    ]
)
y = cp.stack(
    [
        data_enc[i * block_size + 1 : i * block_size + block_size + 1]
        for i in range(len(data_enc) // block_size)
    ]
)

n = int(len(X) * 0.9)
X_train = X.to_int()[:n]
y_train = y.to_int()[:n]
X_val = X.to_int()[n:]
y_val = y.to_int()[n:]

mask = get_causal_mask((block_size, block_size))

model = Transformer(
    n_embeddings=tokenizer.vocab_size,
    embedding_dim=embed_dims,
    ffwd_channels=4 * embed_dims,
    n_heads=6,
    n_blocks=6,
    max_seq_len=block_size,
    mask=mask,
)
model.to_device(device)


grad_accumulation_steps = batch_size // mini_batch_size
step = 0

train_dl = nn.utils.Dataloader((X_train, y_train), mini_batch_size, device)
val_dl = nn.utils.Dataloader((X_val, y_val), mini_batch_size, device, False)
loss_func = nn.CrossEntropy()
optim = nn.optimizers.AdamW(model.get_parameters(), lr=3e-4)

# create tensorboard logging directory
label = "transformer_shakespeare_test"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logdir = f"./runs/{label}_{timestamp}/"
if not os.path.exists(logdir):
    os.makedirs(logdir)

writer = SummaryWriter(log_dir=logdir)
loss = 0
accum_step = 0

while step < max_iter:
    for x, y in train_dl():
        accum_step += 1

        # training
        with model.train():
            # forward pass
            y_pred = model(x)
            loss += loss_func(y_pred, y).item() / grad_accumulation_steps

            # backward pass
            loss_grads = loss_func.backward() / grad_accumulation_steps
            model.backward(loss_grads)  # compute new gradients

        if accum_step == grad_accumulation_steps:
            optim.step()  # update parameters
            optim.reset_grads()  # reset all gradients
            writer.add_scalar("train/loss", loss, step)

            # validation
            if step > 1 and step % val_interval == 0:
                val_loss = 0
                for x_val, y_val in val_dl():
                    y_pred = model(x_val)
                    val_loss += loss_func(y_pred, y_val).item()
                val_loss /= len(val_dl)
                writer.add_scalar("val/loss", val_loss, step)

            # save checkpoints
            if step > 1 and step % checkpoint_interal == 0:
                model_state = model.get_state_dict()
                optim_state = optim.get_state_dict()
                state_dict = {"model": model_state, "optim": optim_state}
                checkpoint_name = f"{label}_{step}.cp"
                cp.save(state_dict, checkpoint_name)

            if step == max_iter:
                break
            step += 1
            loss = accum_step = 0
            print(f"{step=}", end="\r")