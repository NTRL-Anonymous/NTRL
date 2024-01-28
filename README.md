# LTKE

Source code for the paper Open-world Knowledge Embedding in a Low Text Resource Environment

## Requirements

- [PyTorch](http://pytorch.org/) version >= 1.7.1
- [NumPy](http://numpy.org/) version >= 1.19.5
- transformers
- tqdm
- Python version >= 3.6
## Usage
For transductive reasoning:

```bash
cd transductive
python main.py --batch_size 16 --plm bert   --data wn18rr  --self_adversarial --num_facts 10  --epoch  50
```

For inductive reasoning:

```bash
cd inductive
python main.py --batch_size 16 --plm bert   --data wn18rr_v1  --self_adversarial --num_facts 10  --epoch  30
```

The arguments are as following:
* `--bert_lr`: learning rate of the language model.
* `--model_lr`: learning rate of other parameters.
* `--batch_size`: batch size used in training.
* `--weight_decay`: weight dacay used in training.
* `--data`: name of the dataset.
* `--plm`: choice of the language model. Choose from 'bert' and 'bert_tiny'.
* `--self_adversarial`: use self-adversarial negative sampling for efficient KE learning.
* `--model`: choice of the model. Choose from 'NTIE', 'BLP' and 'DKRL'.
* `--text_type`: choice of the text type. Choose from 'neighbor_text', 'desc_text' and 'con_text'.
* `--num_facts`: the number of entity first-order neighborhood facts.
* `--num_tokens`: the number of entity description text tokens.
* `--score_function`: choice of score function. Choose from 'transe', 'distmult' , 'complex' , 'simple' , 'FCN' and 'ProdFCN'.
* `--lamda`: choice of coefficient lamda . Choose from 'lamda1' and 'lamda2'.

### Datasets

The datasets are put in the folder 'transductive/data' and 'inductive/data'.
