# Pytorch implementation of various attention-based language models

## Purpose
Implements attention based language models in a unified structure, assuring code accuracy. Any PRs are warmly welcomed!

## How to run

### Set up env
```bash
pip3 install -r requirements.txt
```

### Prepare dataset
First download related spacy language model
```bash
python -m spacy download en
python -m spacy download de
```

Then, download related datasets of each attention model. Currently, `transformer`, `bert` are supported.
```bash
sh download.sh ${model}
```


### Run language model
You can run implemented language models using `attention/run_attention_model.py` file. Example of training `transformer` model is given below.

Detailed parameters of each models are given in next section.
```bash
python3 transformer/run_attention_model.py \
	--language_model transformer \
	--data_pkl .data/multi30k/m30k_deen_shr.pkl \
	--d_model 512 \
	--d_word_vec 512 \
	--d_inner_hid 2048 \
	--d_k 64 \
	--d_v 64 \
	--n_head 8 \
	--n_layers 6 \
	--batch_size 256 \
	--embs_share_weight \
	--proj_share_weight \
	--label_smoothing \
	--output_dir output \
	--no_cuda \
	--n_warmup_steps 128000 \
	--epoch 400
```

Currently, `transformer`, `bert` models are supported. To see the related arguments of each model, see below.

## List of implemented models and related parameters

|Models|Referred content|Run example|
|---|---|---|
|[Transformer](https://arxiv.org/abs/1706.03762)|[link](https://github.com/jadore801120/attention-is-all-you-need-pytorch)|[link](#Transformer)|
|[Bert](https://arxiv.org/pdf/1810.04805)|[link](https://github.com/codertimo/BERT-pytorch)|[link](#Bert)|

### Transformer
Should set `--language_model` parameter as `transformer`.
```bash
python3 attention/run_attention_model.py \
	--language_model transformer \
	--data_pkl .data/multi30k/m30k_deen_shr.pkl \
	--d_model 512 \
	--d_word_vec 512 \
	--d_inner_hid 2048 \
	--d_k 64 \
	--d_v 64 \
	--n_head 8 \
	--n_layers 6 \
	--batch_size 256 \
	--embs_share_weight \
	--proj_share_weight \
	--label_smoothing \
	--output_dir output \
	--no_cuda \
	--n_warmup_steps 128000 \
	--epoch 400
```
<details><summary>Parameter explanations</summary>

|Parameter name|Explanation|
|---|---|
|`language_model`|Name of `transformer`|
|`data_pkl`|Directory of preprocessed pickle file|
|`d_model`|Projection dimension of `q,k,v`|
|`d_word_vec`|Dimension of word vectors|
|`d_inner_hid`|Inner dimension when `feedforward network` is done|
|`d_k`|`d_model=d_k * n_head`|
|`d_v`|`d_model=d_v * n_head`|
|`n_head`|Number of multi-head attention|
|`batch_size`|Size of a batch|
|`embs_share_weight`|Whether sharing embedding weight between source, target vocab or not.<br>If set, both of vocab sizes will become **same**|
|`proj_share_weight`|Whether sharing embedding weight between target vocab and final projection vocab embedding|
|`label_smoothing`|Whether smoothing when calculating cross entropy|
|`output_dir`|Directory of model history, check point|
|`no_cuda`|Whether use cuda or not|
|`n_warmup_steps`|Warmup steps before training|
|`epoch`|number of epochs|
</details>

### Bert
Should set `--language_model` parameter as `Bert`.
```bash
python3 attention/run_attention_model.py \
    --language_model bert \
    --d_model 512 \
    --d_word_vec 512 \
    --d_inner_hid 2048 \
    --d_k 64 \
    --d_v 64 \
    --n_head 8 \
    --n_layers 6 \
    --batch_size 256 \
    --output_dir output \
    --no_cuda \
    --epoch 400 \
    --movie_conversations ./data/movie_conversations.txt \
    --movie_lines ./data/movie_lines.txt \
    --raw_text ./data \
    --output ./data
```

<details><summary>Parameter explanations</summary>

|Parameter name|Explanation|
|---|---|
|`language_model`|Name of `bert`|
|`d_model`|Projection dimension of `q,k,v`|
|`d_word_vec`|Dimension of word vectors|
|`d_inner_hid`|Inner dimension when `feedforward network` is done|
|`d_k`|`d_model=d_k * n_head`|
|`d_v`|`d_model=d_v * n_head`|
|`n_head`|Number of multi-head attention|
|`batch_size`|Size of a batch|
|`no_cuda`|Whether use cuda or not|
|`n_warmup_steps`|Warmup steps before training|
|`epoch`|number of epochs|
|`movie_conversations`|Directory where `movie_conversations.txt` exists|
|`movie_lines`|Directory where `movie_lines.txt` exists|
|`raw_text`|Directory where preprocessed text will be saved|
|`output`|Directory where results will be saved|
</details>

## How to run pytest
```
pytest
```