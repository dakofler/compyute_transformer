# Compyute Transformer

This is a repo used to develop the transformer modules and functions for `Compyute` (https://github.com/dakofler/compyute) as part of my master's thesis.

## Installation

If you want to make use of GPUs, make sure to install the CUDA Toolkit following the installation guide of `CuPy` (https://docs.cupy.dev/en/stable/install.html).

```bash
git clone https://github.com/dakofler/compyute_transformer.git
cd compyute_transformer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Theory

`theory/attention_theory.ipynb` contains an overview of the attention mechanism as well as embeddings and other concepts used in the original transformer architecture by Vaswani et al.

In the `./dev` directory there are several notebooks related to the development of the transformer functions and modules.

### Implementation
In this project, three implementations of the Multi-Head Attention were implemented:
- `sequential`: contains an `AttentionHead` module that contains the Query, key and value projections and the SDP-Attention. The `SequentialMHA` cointains a list of attention head modules and computes each head sequentially (in a for loop).
- `semiparallel`: Query, key and value projections are computed in parallel for all heads and are then split for each head. Subsequently attention heads are computed sequentially (in a for loop).
- `parallel`: Query, key and value projections and attention heads are computed in parallel using batched matrix multiplications reshaping of tensors.

### Verification and Evaluation
- `verification_` notebooks contain code to verify that the implementation yields the same results (outputs and gradients) as Pytorch
- `implementation_benchmark.py` is a script to evaluate the performance of an implementation by passing `parallel`, `semiparallel` or `sequential` as an argument, for example

```bash
python3 dev/implementation_benchmark.py parallel
```

### Training

The `./transformer/` directory contains the final implementation of the transformer modules and functions. It also contains two transformer implementations, one following the original architecture by Vaswani et al. and one implementing the GPT architecture (Radford et al.).

`training_shakespeare.py` is an example script, where a GPT-Transformer is trained to mimic Shakespeare-like language by training on the `tinyshakespeare` dataset (https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

```bash
python training_shakespeare.py
```

`inference.ipynb` provides a way to use a trained model to generate text.

## Author
Daniel Kofler - AI Research Associate ([dkofler@outlook.com](mailto:dkofler@outlook.com))

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Final Notes
All modules developed will be integrated into `Compyute` once my thesis is done. If you have any suggestions or find any bugs, please don't hesitate to contact me.

Cheers,<br>
Daniel