"""training script on tinyshakespeare using Pytorch"""

import math
import os
from datetime import datetime

import requests
import torch
import torch.nn.functional as F
from compyute.nn.utils.tensorboard import SummaryWriter
from simple_tokenizers import CharacterTokenizer
from torch import nn


class Transformer(nn.Module):
    def __init__(
        self,
        n_embeds,
        embed_dim,
        mlp_channels,
        n_heads,
        n_blocks,
        max_context_len,
        dropout,
    ):
        super().__init__()

        # Embeddings
        self.token_emb = nn.Embedding(n_embeds, embed_dim)
        self.pos_emb = nn.Embedding(max_context_len, embed_dim)
        std = 1 / math.sqrt(embed_dim)
        nn.init.normal_(self.token_emb.weight, std=std)
        nn.init.normal_(self.pos_emb.weight, std=std)

        # Transformer blocks
        out_scale = 1 / math.sqrt(2 * n_blocks)
        self.blocks = nn.ModuleList(
            TransformerBlock(embed_dim, mlp_channels, n_heads, dropout, out_scale)
            for _ in range(n_blocks)
        )

        # Language model head
        self.ln = nn.LayerNorm((embed_dim,))
        self.head = nn.Linear(embed_dim, n_embeds, bias=False)
        self.head.weight = self.token_emb.weight  # weight sharing

        self.pos = nn.Buffer(torch.arange(max_context_len).view(1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(x) + self.pos_emb(self.pos[:, : x.shape[-1]])
        for block in self.blocks:
            x = block(x)
        x = self.head(self.ln(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, mlp_channels, n_heads, dropout, out_scale):
        super().__init__()

        self.ln1 = nn.LayerNorm((embed_dim,))
        self.attn = MSA(embed_dim, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        std = out_scale / math.sqrt(embed_dim)
        torch.nn.init.uniform_(self.attn.out_proj.weight, -std, std)

        self.ln2 = nn.LayerNorm((embed_dim,))
        self.mlp = MLP(embed_dim, mlp_channels)
        self.dropout2 = nn.Dropout(dropout)

        std = out_scale / math.sqrt(mlp_channels)
        torch.nn.init.uniform_(self.mlp.down.weight, -std, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout1(self.attn(self.ln1(x)))
        x = x + self.dropout2(self.mlp(self.ln2(x)))
        return x


class MSA(nn.Module):
    """Multi-head self-attention module"""

    def __init__(self, embed_dim, num_heads, dropout) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.in_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        qkv = self.in_proj(x)
        q, k, v = qkv.split(D, dim=2)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        dropout = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=True)
        y = y.transpose(1, 2).contiguous().flatten(2)
        y = self.out_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_channels):
        super().__init__()
        self.up = nn.Linear(embed_dim, mlp_channels)
        self.down = nn.Linear(mlp_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


def main() -> None:
    torch.manual_seed(1337)
    device = "cuda"

    # hyperparameters
    context_length = 256
    embed_dims = 384
    n_heads = 6
    n_blocks = 6
    batch_size = 64
    dropout = 0.5

    # training parameters
    step = 1
    max_steps = 2500
    label = "transformer_shakespeare_pt_8"
    val_interval = 250

    # load data
    DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(DATA_URL)
    data = response.text

    # tokenize data
    chars = sorted(list(set(data)))
    tokenizer = CharacterTokenizer()
    tokenizer.vocab = {i: c for i, c in enumerate(chars)}

    # prepare data
    data_enc = torch.tensor(tokenizer.encode(data), dtype=torch.int32)
    X = torch.stack(
        [
            data_enc[i * context_length : i * context_length + context_length]
            for i in range(len(data_enc) // context_length)
        ]
    )
    y = torch.stack(
        [
            data_enc[i * context_length + 1 : i * context_length + context_length + 1]
            for i in range(len(data_enc) // context_length)
        ]
    )

    n = int(len(X) * 0.9)
    X_train = X.long()[:n]
    y_train = y.long()[:n]
    X_val = X.long()[n:]
    y_val = y.long()[n:]

    # create model
    model = Transformer(
        n_embeds=tokenizer.vocab_size,
        embed_dim=embed_dims,
        mlp_channels=4 * embed_dims,
        n_heads=n_heads,
        n_blocks=n_blocks,
        max_context_len=context_length,
        dropout=dropout,
    ).to(device)

    # training
    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)
    val_ds = torch.utils.data.TensorDataset(X_val, y_val)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size, shuffle=False)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # create tensorboard logging directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logdir = f"./runs/{label}_{timestamp}/"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(log_dir=logdir)

    model.train()
    while step < max_steps:

        for batch in train_dl:
            x = batch[0].to(device)
            y = batch[1].to(device)

            # training
            y_pred = model(x)
            loss = F.cross_entropy(y_pred.transpose(1, 2), y)
            loss.backward()

            optim.step()
            optim.zero_grad()
            writer.add_scalar("train/loss", loss.item(), step)

            # validation
            if step > 1 and step % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for batch in val_dl:
                        x_val = batch[0].to(device)
                        y_val = batch[1].to(device)
                        y_pred = model(x_val)
                        val_loss += F.cross_entropy(
                            y_pred.transpose(1, 2), y_val
                        ).item()
                    val_loss /= len(val_dl)
                writer.add_scalar("val/loss", val_loss, step)

                model.train()

            if step == max_steps:
                break
            step += 1
            print(f"{step=}", end="\r")


if __name__ == "__main__":
    main()
