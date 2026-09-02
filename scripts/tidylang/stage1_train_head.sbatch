#!/bin/bash
#SBATCH --job-name=lid-tidylang-stage1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --account=<your_slurm_account>
#SBATCH --qos=<your_qos>


# module load pytorch-gpu/py3/2.8.0
# source /path/to/venv/bin/activate

REPO_ROOT=${REPO_ROOT:?set REPO_ROOT to your balr-lid checkout}
AUDIO_ROOT=${AUDIO_ROOT:?set AUDIO_ROOT to your Tidylang audio directory}
BACKBONE=${BACKBONE:-facebook/mms-1b}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/exp/tidylang-stage1}

cd "$REPO_ROOT"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

python -m balr_lid.train \
    data=tidylang \
    recipe=head_stage \
    output_dir="$OUTPUT_DIR" \
    data.audio_root="$AUDIO_ROOT" \
    arch.backbone.pretrained_model_name_or_path="$BACKBONE" \
    arch.head.embed_dim=1280 \
    arch.head.num_heads=8 \
    arch.head.use_binary_encoder=true \
    arch.head.binary_dim=320 \
    arch.head.use_balr_head=true \
    arch.head.bit_dropout=true \
    arch.head.dropout_law=geometric \
    arch.head.dropout_rho=0.9 \
    arch.head.nested_dropout=true \
    recipe.optimizer.lr=1e-5 \
    recipe.train_sampler.batch_size=70 \
    recipe.train_sampler.balance_mode=batch \
    trainer.devices="$NUM_GPUS" \
    +trainer.check_val_every_n_epoch=5

