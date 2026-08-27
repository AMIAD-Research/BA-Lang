# E2E BA-LANG : framework for language identification 

A training recipe for **spoken language identification (LID)**:
finetune a self-supervised speech backbone with an
MHFA head. Includes an optional **BA-LR** (Binary
Attribute Likelihood Ratio) classification head, which scores languages via
a learned Bernoulli model over a binarized embedding indicating presence/absence of interpretable language attributes.

## Related work : BA-Lang backend for LID

The BA-Lang classification head (`use_balr_head=True`) scores each language by
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

SLURM job templates for both stages and both corpora are in
[scripts/](scripts/) (`scripts/{fleurs,tidylang}/stage{1,2}_*.sbatch`) —
fill in `--account`/`--qos` and the `REPO_ROOT`/`AUDIO_ROOT` environment
variables for your cluster.


## Repository structure

```
configs/          Hydra configs (data / model architecture / recipe)
src/balr_lid/      library code (data loading, model, training, inference)
scripts/           SLURM job templates, per corpus and stage
docs/              data preparation and recipe deep-dives
tests/             unit tests (sampler balance, param freezing, head shapes)
```






