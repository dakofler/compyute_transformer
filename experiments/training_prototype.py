import time

import compyute as cp
from attention_s import get_causal_mask
from compyute import nn
from compyute.preprocessing.text import CharacterTokenizer
from gpt_transformer import GPTTransformer

cp.random.set_seed(1337)
device = cp.cuda
embed_dims = 384
block_size = 256
batch_size = 64
mini_batch_size = 32


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

model = GPTTransformer(
    n_embeddings=tokenizer.vocab_size,
    embedding_dim=embed_dims,
    ffwd_channels=4 * embed_dims,
    n_heads=6,
    n_blocks=6,
    max_seq_len=block_size,
    mask=mask,
)
model.to_device(device)


train_dl = nn.utils.Dataloader((X_train, y_train), mini_batch_size, device)
val_dl = nn.utils.Dataloader((X_val, y_val), mini_batch_size, device, False)
loss_func = nn.CrossEntropy()
optim = nn.optimizers.AdamW(model.get_parameters(), lr=3e-4)

grad_accum_steps = batch_size // mini_batch_size
step = 1
for x, y in train_dl():
    start = time.time()

    loss = 0
    for i in range(grad_accum_steps):
        with model.train():
            loss += loss_func(model(x), y).item() / grad_accum_steps
            model.backward(loss_func.backward() / grad_accum_steps)

    optim.step()  # update parameters
    optim.reset_grads()  # reset all gradients

    cp.backend.synchronize()
    dt = time.time() - start

    tok_per_s = batch_size * block_size / dt
    print(f"step {step:4} | loss {loss:.4f} | dt {dt:.4f} s | {tok_per_s:.1f} tokens/s")

    step += 1
