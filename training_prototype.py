import time

import compyute as cp
import requests
from compyute import nn
from tokenizers.character_tokenizer import CharacterTokenizer

from transformer.attention_funcs import get_causal_mask
from transformer.gpt import GPTTransformer

cp.random.set_seed(1337)
device = cp.cuda

embed_dims = 384
context_length = 256
n_heads = 6
n_blocks = 6
batch_size = 64
mini_batch_size = 32
val_interval = 250

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(DATA_URL)
data = response.text

chars = sorted(list(set(data)))
tokenizer = CharacterTokenizer()
tokenizer.vocab = {i: c for i, c in enumerate(chars)}
tokenizer.ivocab = {c: i for i, c in enumerate(chars)}
data_enc = tokenizer.encode(data)

data_enc_t = cp.tensor(data_enc, dtype=cp.int32)
X = cp.stack(
    [
        data_enc_t[i * context_length : i * context_length + context_length]
        for i in range(len(data_enc_t) // context_length)
    ]
)
y = cp.stack(
    [
        data_enc_t[i * context_length + 1 : i * context_length + context_length + 1]
        for i in range(len(data_enc_t) // context_length)
    ]
)
n = int(len(X) * 0.9)
X_train = X[:n]
y_train = y[:n]
X_val = X[n:]
y_val = y[n:]

mask = get_causal_mask(context_length)

model = GPTTransformer(
    n_embeds=tokenizer.vocab_size,
    embed_dim=embed_dims,
    mlp_channels=4 * embed_dims,
    n_heads=n_heads,
    n_blocks=n_blocks,
    max_context_len=context_length,
    mask=mask,
)
model.to_device(device)

train_dl = nn.utils.Dataloader((X_train, y_train), mini_batch_size, device)
val_dl = nn.utils.Dataloader((X_val, y_val), mini_batch_size, device, False)
loss_fn = nn.CrossEntropyLoss()
optim = nn.optimizers.AdamW(model.get_parameters(), lr=3e-4)

grad_accum_steps = batch_size // mini_batch_size
step = 1
for x, y in train_dl():
    start = time.perf_counter()

    model.training()
    loss = 0.0
    for i in range(grad_accum_steps):
        loss += loss_fn(model(x), y).item() / grad_accum_steps
        model.backward(loss_fn.backward() / grad_accum_steps)

    optim.step()  # update parameters
    optim.reset_grads()  # reset all gradients

    cp.backend.synchronize()
    dt = time.perf_counter() - start

    tok_per_s = batch_size * context_length / dt
    print(f"step {step:4} | loss {loss:.4f} | dt {dt:.4f} s | {tok_per_s:.1f} tokens/s")

    if step > 1 and step % val_interval == 0:
        model.inference()
        print("Running validation.")
        with nn.no_cache_ctx():
            val_loss = 0.0
            for x_val, y_val in val_dl():
                y_pred = model(x_val)
                val_loss += loss_fn(y_pred, y_val).item()
            val_loss /= len(val_dl)

    step += 1
