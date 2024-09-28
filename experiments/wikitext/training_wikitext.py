import os
from datetime import datetime

import compyute as cp
from compyute import nn
from compyute.nn.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

from experiments.attention_funcs import get_causal_mask
from experiments.transformer_gpt import GPTTransformer

cp.random.set_seed(1337)
device = cp.cuda

vocab_size = 8192
block_size = 1024
embed_dims = 384

batch_size = 64
mini_batch_size = 8
val_interval = 250
max_iter = 2500
checkpoint_interal = 500


dataset = load_dataset(path="Salesforce/wikitext", name="wikitext-2-v1")


def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset["train"][i : i + 1000]["text"]


file = "wikitext_tokenizer.json"

if not os.path.exists(file):
    tokenizer = Tokenizer(
        models.WordPiece(unk_token="[UNK]", max_input_chars_per_word=1000000000)
    )
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )
    pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
    )
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        continuing_subword_prefix="",
    )
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    tokenizer.save(file)
else:
    tokenizer = Tokenizer.from_file(file)


def encode(split):
    lines = dataset[split]["text"]
    encodings = tokenizer.encode_batch(lines)
    token_id_lists = [encoding.ids for encoding in encodings]
    token_ids = [
        token_id for token_id_list in token_id_lists for token_id in token_id_list
    ]

    return cp.tensor(token_ids).to_int()


train_data_enc = encode("train")
val_data_enc = encode("validation")

X_train = cp.stack(
    [
        train_data_enc[i * block_size : i * block_size + block_size]
        for i in range(len(train_data_enc) // block_size)
    ]
)
y_train = cp.stack(
    [
        train_data_enc[i * block_size + 1 : i * block_size + block_size + 1]
        for i in range(len(train_data_enc) // block_size)
    ]
)

X_val = cp.stack(
    [
        val_data_enc[i * block_size : i * block_size + block_size]
        for i in range(len(val_data_enc) // block_size)
    ]
)
y_val = cp.stack(
    [
        val_data_enc[i * block_size + 1 : i * block_size + block_size + 1]
        for i in range(len(val_data_enc) // block_size)
    ]
)

mask = get_causal_mask((block_size, block_size))

model = GPTTransformer(
    n_embeddings=vocab_size,
    embedding_dim=embed_dims,
    ffwd_channels=4 * embed_dims,
    n_heads=6,
    n_blocks=6,
    max_seq_len=block_size,
    mask=mask,
    dropout=0.2,
)
model.to_device(device)

label = "transformer_wikitext_2"
summary = nn.utils.modules.get_module_summary(
    model, (block_size,), input_dtype=cp.int32
)
with open(label + ".txt", "w") as f:
    f.write(summary)

grad_accumulation_steps = batch_size // mini_batch_size
step = 0

train_dl = nn.utils.Dataloader((X_train, y_train), mini_batch_size, device)
val_dl = nn.utils.Dataloader((X_val, y_val), mini_batch_size, device, False)
loss_fn = nn.CrossEntropy()
optim = nn.optimizers.AdamW(model.get_parameters(), lr=3e-4)

# create tensorboard logging directory
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logdir = f"./runs/{label}_{timestamp}/"
if not os.path.exists(logdir):
    os.makedirs(logdir)

writer = SummaryWriter(log_dir=logdir)
loss = 0.0
accum_step = 0

model.training()

while step < max_iter:
    for x, y in train_dl():
        accum_step += 1

        # training
        y_pred = model(x)
        loss += loss_fn(y_pred, y).item() / grad_accumulation_steps
        loss_grads = loss_fn.backward() / grad_accumulation_steps
        model.backward(loss_grads)  # compute new gradients

        if accum_step == grad_accumulation_steps:
            optim.step()  # update parameters
            optim.reset_grads()  # reset all gradients
            writer.add_scalar("train/loss", loss, step)

            # validation
            if step > 1 and step % val_interval == 0:
                model.inference()
                with nn.no_caching():
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

            if step == max_iter:
                break
            step += 1
            loss = accum_step = 0
            print(f"{step=}", end="\r")
