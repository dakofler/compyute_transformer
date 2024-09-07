import time

import compyute as cp
import compyute.nn as nn
from compyute.preprocessing.text import CharacterTokenizer
from transformer import Transformer, get_causal_mask

device = cp.cuda
embed_dims = 384
block_size = 256
batch_size = 32

val_interval = 200
checkpoint_interal = 500
max_iter = 100


with open("data/tinyshakespeare.txt", "r") as f:
    data = f.read()


chars = sorted(list(set(data)))
tokenizer = CharacterTokenizer()
tokenizer.vocab = {i: c for i, c in enumerate(chars)}
tokenizer.ivocab = {c: i for i, c in enumerate(chars)}
data_enc = tokenizer.encode(data)


data_enc = cp.tensor(data_enc, dtype=cp.int32)
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
X_train = X[:n]
y_train = y[:n]
X_val = X[n:]
y_val = y[n:]

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


train_dl = nn.utils.Dataloader((X_train, y_train), batch_size, device=device)
val_dl = nn.utils.Dataloader((X_val, y_val), batch_size, device=device)
loss_func = nn.CrossEntropy()
optim = nn.optimizers.AdamW(model.get_parameters(), lr=3e-4, beta1=0.9, beta2=0.95)


step = 1
while step <= max_iter:
    for x, y in train_dl():
        start = time.time()

        with model.train():
            loss = loss_func(model(x), y).item()
            model.backward(loss_func.backward())

        optim.step()  # update parameters
        optim.reset_grads()  # reset all gradients

        dt = time.time() - start

        # validation
        if step > 1 and step % val_interval == 0:
            val_loss = 0
            for x_val, y_val in val_dl():
                y_pred = model(x_val)
                val_loss += loss_func(y_pred, y_val).item()
            val_loss /= len(val_dl)
            cp.backend.free_cuda_memory()

        print(f"step {step:4} | loss {loss:.4f} | dt {dt:.4f} s")

        if step == max_iter:
            break
        step += 1

    if step == max_iter:
        break
