# Data manifests

Training and evaluation both read a flat CSV **manifest**, one row per
utterance, loaded by `balr_lid.data.manifest.load_manifest` and
`balr_lid.data.dataset.LanguageIdDataset`.

## Manifest format

Required columns:

| column | type | meaning |
|---|---|---|
| `id` | str | unique utterance identifier |
| `audio_path` | str | path to the audio file, **relative to `audio_root`** |
| `language` | str | language code (must appear in the corpus's `language_list`, see `configs/data/*.yaml`) |
| `duration` | float | utterance duration in seconds |

Extra columns (speaker id, transcription, ...) are allowed and ignored.

Example (`data/manifests/fleurs-train.csv`):

```csv
id,audio_path,language,duration
fleurs-fra-0001,fra/train/0001.wav,fra,4.32
fleurs-eng-0001,eng/train/0001.wav,eng,3.87
...
```

Given this manifest and `data.audio_root=/corpora/fleurs/audio`, the file
actually loaded for the first row is
`/corpora/fleurs/audio/fra/train/0001.wav`.

One manifest per split, referenced from `configs/data/<corpus>.yaml`:

```
data/manifests/<corpus>-train.csv
data/manifests/<corpus>-dev.csv
data/manifests/<corpus>-test.csv
```

(paths are configurable via `data.manifest_dir`, default `data/manifests`).

## Building a manifest for FLEURS

If your corpus is laid out as the raw FLEURS release — a single root with
one subdirectory per locale, each holding its own tsv metadata and its own
audio subtree:

```
corpus_root/
  ha_ng/
    train.tsv  dev.tsv  test.tsv        # tab-separated, no header:
                                          # id, filename, raw_transcription,
                                          # normalized_transcription, phonemic,
                                          # num_samples, gender
    audio/
      train/<filename>.wav
      dev/<filename>.wav
      test/<filename>.wav
  fr_fr/
    ...
```

Use `scripts/build_fleurs_manifest.py` to generate the manifest CSVs
directly :

```bash
python scripts/build_fleurs_manifest.py \
    --corpus-root /path/to/fleurs \
    --output-dir data/manifests
```


## Building manifests for Tidylang

Raw layout (both the train corpus and the eval corpus): one directory per
speaker, one subdirectory per language, `.wav` files inside:

```
<corpus_root>/
  id010001/
    en/en_30308892.wav
    cy/cy_30309019.wav
    ...
  id010002/
    ...
```

**Train/dev** — `TidyVoiceX_ASV`, alongside a `manifest.txt` (one row per
utterance, whitespace-separated, no header):

```
<split_flag>  <speaker>/<lang>/<file>.wav  <lang_code>
```

Only `split_flag == 1` rows are used : 90/10, stratified per language.

```bash
python scripts/build_tidylang_manifest.py train \
    --corpus-root /path/to/TidyVoiceX_ASV \
    --output-dir data/manifests
```

**Eval** — `TidyVoiceX2_ASV`, alongside `tl26_lid.txt`:

```
<speaker>/<lang>/<file>.wav  <lang_code>
```

```bash
python scripts/build_tidylang_manifest.py eval \
    --corpus-root /path/to/TidyVoiceX2_ASV \
    --output-dir data/manifests
```



