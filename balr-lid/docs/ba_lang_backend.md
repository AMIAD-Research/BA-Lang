# 3-step BA-Lang backend

A closed-form Bernoulli classifier that scores a
language from a binary vector.


## Pipeline

```
Training a standard embeddings extractor system (step 1)
      │
      ▼
  Embeddings
      │
      ▼
Training the AutoEncoder (step 2)
      │
      ▼
binary embeddings (train split) + manifest (id, language)
      │
      ▼ (step 3)
"enrollment": count activations
per language, per attribute
      │                                  
      ▼
 (language, count, BA0, BA1, ..., BA{K-1})
      │
      ▼  
scoring of a test embedding          

        
```

# Binary autoencoder (BAE) training (step 2)


Reference: I. Ben-Amor, J.-F. Bonastre, S. Mdhaffar, "Extraction of
interpretable and shared speaker-specific speech attributes through binary
auto-encoder", Interspeech 2024.


## Training

```bash
python -m balr_lid.train_bae \
    bae_data=fleurs \
    output_dir=exp/fleurs-bae \
    bae_data.train_embedding_file=/path/to/embeddings/inference_fleurs_train.h5 \
    bae_data.val_embedding_file=/path/to/embeddings/inference_fleurs_dev.h5 \
    bae_data.test_embedding_file=/path/to/embeddings/inference_fleurs_dev.h5 \
    bae_model.input_dim=1280 \
    bae_model.internal_dim=320
```


### Structured bit-dropout (optional)

Off by default (`bae_model.bit_dropout=false`). 

```bash
python -m balr_lid.train_bae \
    bae_data=tidylang \
    output_dir=exp/tidylang-bae \
    bae_data.train_embedding_file=/path/to/train_embeddings.h5 \
    bae_data.val_embedding_file=/path/to/dev_embeddings.h5 \
    bae_data.test_embedding_file=/path/to/dev_embeddings.h5 \
    bae_model.input_dim=1280 \
    bae_model.internal_dim=320 \
    bae_model.bit_dropout=true \
    bae_model.dropout_law=geometric \
    bae_model.dropout_rho=0.95
```

- `bae_model.dropout_law=uniform|geometric`: how the per-example cutoff `n`
  (number of leading bits kept) is drawn.
- `bae_model.dropout_rho`: only used when `dropout_law=geometric`. Close to
  `1`: soft cutoff, most bits stay active most of the time. Close to `0`:
  aggressive cutoff.


## Inference 

```bash
python -m balr_lid.infer_bae \
    --checkpoint exp/fleurs-bae/checkpoints/last.ckpt \
    --h5-in /path/to/embeddings/inference_fleurs_train.h5 \
    --h5-out exp/fleurs-bae/binary/inference_fleurs_train.h5
```

`input_dim`, `internal_dim` and whether to L2-normalize the input are all
read back from the checkpoint's saved hyperparameters, so they can never
drift from what the model was actually trained with — no need to pass them
again on the command line.


# Commands for BA-Lang scoring (step 3)

```bash
# 1) Enroll (compute per-language attribute activation probabilities)
python -m balr_lid.train_ba_lang_backend \
    --binary-embeddings exp/fleurs-bae/binary/inference_fleurs_train.h5 \
    --manifest data/manifests/fleurs-train.csv \
    --output exp/fleurs-balang/enrollment_statistics.csv

# 2) Score a test split
python -m balr_lid.infer_ba_lang_backend \
    --enrollment-statistics exp/fleurs-balang/enrollment_statistics.csv \
    --binary-embeddings exp/fleurs-bae/binary/inference_fleurs_test.h5 \
    --manifest data/manifests/fleurs-test.csv \
    --output exp/fleurs-balang/predictions_fleurs_test.csv

# 3) Accuracy, with bootstrap confidence intervals
python scripts/compute_accuracy.py \
    --predictions exp/fleurs-balang/predictions_fleurs_test.csv \
    --confidence-intervals --per-language
```

### Scoring with a subset of attributes

`infer_ba_lang_backend` accepts `--k-first`/`--k-last` (mutually
exclusive), useful to check how few leading/trailing attributes are
needed to keep accuracy :

```bash
python -m balr_lid.infer_ba_lang_backend \
    --enrollment-statistics exp/fleurs-balang/enrollment_statistics.csv \
    --binary-embeddings exp/fleurs-bae/binary/inference_fleurs_test.h5 \
    --manifest data/manifests/fleurs-test.csv \
    --output exp/fleurs-balang/predictions_fleurs_test_k64.csv \
    --k-first 64
```
