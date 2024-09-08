import time

import compyute as cp
from compyute import nn
from compyute.preprocessing.text import CharacterTokenizer
from transformer import Transformer, get_causal_mask

device = cp.cuda
embed_dims = 384
block_size = 256
batch_size = 32


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


train_dl = nn.utils.Dataloader((X_train, y_train), batch_size, device, True, True)
val_dl = nn.utils.Dataloader((X_val, y_val), batch_size, device, False, True)
loss_func = nn.CrossEntropy()
optim = nn.optimizers.AdamW(model.get_parameters(), lr=3e-4, beta1=0.9, beta2=0.95)

step = 1
while True:
    for x, y in train_dl():
        start = time.time()

        with model.train():
            loss = loss_func(model(x), y).item()
            model.backward(loss_func.backward())

        optim.step()  # update parameters
        optim.reset_grads()  # reset all gradients

        cp.backend.synchronize()
        dt = time.time() - start

        print(f"step {step:4} | loss {loss:.4f} | dt {dt:.4f} s")

        break

    break
