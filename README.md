# E2E BA-LANG : framework for language identification 

A training recipe for **spoken language identification (LID)**:
finetune a self-supervised speech backbone with an
MHFA head. Includes an optional **BA-LR** (Binary
Attribute Likelihood Ratio) classification head, which scores languages via
a learned Bernoulli model over a binarized embedding indicating presence/absence of interpretable language attributes.

## Related work : BA-Lang backend for LID

The BA-Lang classification head scores each language by
the Bernoulli log-likelihood under per-class.
Find more details about the BA-Lang scoring backend here : https://www.isca-archive.org/jep_2026/jelassi26_jep.html

## Systems Architecture

```
raw audio (16kHz)
      │
      ▼
 SSLBackbone            
     
      │
      ▼
 MultiHeadFactorizedAggregation (MHFA)
      │
      ▼
 LanguageClassification
   ├─ cosine classifier , or
   └─ BA-Lang : BinaryEncoder → per-class Bernoulli log-likelihood scoring
      │
      ▼
 language logits (cross-entropy loss)

```

See [src/balr_lid/models](src/balr_lid/models) for the implementation of
each block.

## Data

Training is done separately on two corpora : Tidylang and FLEURS.

## Downloading the datasets

The corpora used in this work are public, you can download them easily and
then build the manifests (see [docs/data_manifest.md](docs/data_manifest.md)).

- **FLEURS** (public): from the
  [google/fleurs dataset on Hugging Face](https://huggingface.co/datasets/google/fleurs),
  download each locale's raw `.tar.gz` archives (audio + tsv metadata) and
  extract them under a common `corpus_root`, one directory per locale (e.g.
  `corpus_root/fr_fr/`, `corpus_root/ha_ng/`, ...).



- **Tidylang** (public): `TidyVoiceX_ASV` (train/dev) and
  `TidyVoiceX2_ASV` (eval), from Mozilla Data Collective:
  - train/dev: <https://mozilladatacollective.com/datasets/cmihtsewu023so207xot1iqqw>
  - eval: <https://mozilladatacollective.com/datasets/cmkv32i5e02tumg07j79d3c35>

## Quickstart

```bash
# 1) Install
cd /path/to/balr-lid   # the repo root
python -m venv .venv && source .venv/bin/activate   
pip install -e ".[dev]"


# 2) Prepare manifests — see docs/data_manifest.md
#    data/manifests/fleurs-{train,dev,test}.csv
#    build them with:
python scripts/build_fleurs_manifest.py \
    --corpus-root /path/to/fleurs \
    --output-dir data/manifests

# 3) Stage 1: train the head (backbone frozen)
python -m balr_lid.train \
    data=fleurs recipe=head_stage \
    output_dir=exp/fleurs-stage1 \
    data.audio_root=/path/to/fleurs/audio \
    arch.backbone.pretrained_model_name_or_path=facebook/mms-1b

# 4) Stage 2: finetune backbone + head
python -m balr_lid.train \
    data=fleurs recipe=finetune_stage \
    output_dir=exp/fleurs-stage2 \
    data.audio_root=/path/to/fleurs/audio \
    arch.backbone.pretrained_model_name_or_path=facebook/mms-1b \
    recipe.checkpoint=exp/fleurs-stage1/checkpoints/last.ckpt

# 5) Inference
python -m balr_lid.inference \
    --checkpoint exp/fleurs-stage2/checkpoints/last.ckpt \
    --manifest data/manifests/fleurs-test.csv \
    --audio-root /path/to/fleurs/audio \
    --language-list afr amh ara ... \
    --output predictions.csv \
    --save-embeddings embeddings.h5
```

### Detailed examples (head configurations)

A full
FLEURS run with the BA-LR head looks like:

```bash
python -m balr_lid.train \
    data=fleurs recipe=head_stage \
    output_dir=/path/to/exp/fleurs-stage1 \
    data.audio_root=/path/to/fleurs \
    data.manifest_dir=/path/to/data/manifests \
    arch.backbone.pretrained_model_name_or_path=/path/to/mms-1b \
    arch.head.embed_dim=1280 \
    arch.head.num_heads=8 \
    arch.head.use_binary_encoder=true \
    arch.head.binary_dim=320 \
    arch.head.use_balr_head=true \
    recipe.optimizer.lr=1e-5 \
    recipe.train_sampler.batch_size=102 \
    recipe.train_sampler.balance_mode=batch \
    +trainer.check_val_every_n_epoch=5
```

When you want to enable structured
Nested dropout on the attributes space :

```bash
python -m balr_lid.train \
    data=tidylang recipe=head_stage \
    output_dir=/path/to/exp/tidylang-stage1 \
    data.audio_root=/path/to/TidyVoiceX_ASV \
    data.manifest_dir=/path/to/data/manifests \
    arch.backbone.pretrained_model_name_or_path=/path/to/mms-1b \
    arch.head.embed_dim=1280 \
    arch.head.num_heads=8 \
    arch.head.use_binary_encoder=true \
    arch.head.binary_dim=320 \
    arch.head.use_balr_head=true \
    arch.head.bit_dropout=true \
    arch.head.dropout_law=geometric \
    arch.head.dropout_rho=0.95 \
    arch.head.nested_dropout=true \
    recipe.optimizer.lr=1e-5 \
    recipe.train_sampler.batch_size=70 \
    recipe.train_sampler.balance_mode=batch \
    trainer.devices="$NUM_GPUS" \
    +trainer.check_val_every_n_epoch=1
```



> **Structured ("nested"/Matryoshka) dropout is off by default**
> (`arch.head.bit_dropout=false`). To enable it,
> add:
>
> - `arch.head.bit_dropout=true` : turns it on.
> - `arch.head.dropout_law=uniform|geometric` : how the per-example cutoff
>   `n` (number of leading attributes kept) is drawn:
>   - `uniform`: `n ~ Uniform(dropout_n_min, binary_dim)` : no attribute favored over another.
>   - `geometric`: `n ~ Geometric(dropout_rho)`, truncated to
>     `[dropout_n_min, binary_dim]` : short prefixes are drawn more often as
>     `dropout_rho -> 0`.
> - `arch.head.dropout_rho=<float in (0, 1)>` : only used when
>   `dropout_law=geometric`. Close to `1`: soft cutoff, most attributes stay
>   active most of the time. Close to `0`: aggressive cutoff.
> - `arch.head.nested_dropout=true|false` : how dropped attributes are
>   scored: `false` (default) excludes them from the loss (Matryoshka-inspired), `true` : Nested dropout.


SLURM job templates for both stages and both corpora are in
[scripts/](scripts/).

## Binary autoencoder (BAE) training

A separate training step (necessary only for BA-Lang backend): 

```bash
# Train
python -m balr_lid.train_bae \
    bae_data=fleurs output_dir=exp/fleurs-bae \
    bae_data.train_embedding_file=/path/to/train_embeddings.h5 \
    bae_data.val_embedding_file=/path/to/dev_embeddings.h5 \
    bae_data.test_embedding_file=/path/to/dev_embeddings.h5 \
    bae_model.input_dim=1280 bae_model.internal_dim=320

# Inference
python -m balr_lid.infer_bae \
    --checkpoint exp/fleurs-bae/checkpoints/last.ckpt \
    --h5-in /path/to/train_embeddings.h5 \
    --h5-out exp/fleurs-bae/binary/train_embeddings.h5
```

## BA-Lang backend

A closed-form Bernoulli classifier. See
[docs/ba_lang_backend.md](docs/ba_lang_backend.md).

```bash
# Enroll: per-language attribute activation probabilities
python -m balr_lid.train_ba_lang_backend \
    --binary-embeddings exp/fleurs-bae/binary/train_embeddings.h5 \
    --manifest data/manifests/fleurs-train.csv \
    --output exp/fleurs-balang/enrollment_statistics.csv

# Score a test split -> the same predictions.csv format as balr_lid.inference
python -m balr_lid.infer_ba_lang_backend \
    --enrollment-statistics exp/fleurs-balang/enrollment_statistics.csv \
    --binary-embeddings exp/fleurs-bae/binary/test_embeddings.h5 \
    --manifest data/manifests/fleurs-test.csv \
    --output exp/fleurs-balang/predictions_fleurs_test.csv

# Accuracy, with bootstrap confidence intervals
python scripts/compute_accuracy.py \
    --predictions exp/fleurs-balang/predictions_fleurs_test.csv --confidence-intervals
```

## Repository structure

```
configs/          Hydra configs (data / model architecture / recipe / bae_data / bae_model)
src/balr_lid/      library code (data loading, model, training, inference)
scripts/           SLURM job templates, per corpus and stage
docs/              data preparation, recipe, BAE and BA-Lang backend deep-dives
```


## AI Assistance

This code was developed with the help of Claude Code. The recipe, the models architecture, and the data format and the experimental protocole are the author's own design choices. 




