```bash 
python3 scripts/build_fleurs_manifest.py \
    --corpus-root /lustre/fsmisc/dataset/FLEURS \
    --output-dir $SCRATCH/data/manifests

python3 -m balr_lid.train \
    data=fleurs recipe=head_stage \
    output_dir=exp/fleurs-stage1 \
    data.audio_root=$CORPUS_ROOTDIR/fleurs \
    data.manifest_dir=$CORPUS_ROOTDIR/fleurs/list \
    arch.backbone.pretrained_model_name_or_path=$MODEL_ROOTDIR/mms-1b/mms-1b


python -m balr_lid.train \
    data=fleurs recipe=head_stage \
    output_dir=/lustre/fsn1/projects/rech/ldz/ucq25rb/exp/fleurs-stage1-mms \
    data.audio_root=/lustre/fsmisc/dataset/FLEURS \
    data.manifest_dir=/lustre/fsn1/projects/rech/ldz/ucq25rb/data/manifests \
    arch.backbone.pretrained_model_name_or_path=/lustre/fswork/projects/rech/ldz/commun/models/mms-1b/mms-1b \
    arch.head.embed_dim=1280 \
    arch.head.num_heads=8 \
    arch.head.use_binary_encoder=true \
    arch.head.binary_dim=320 \
    arch.head.use_balr_head=true \
    recipe.optimizer.lr=1e-5 \
    recipe.train_sampler.batch_size=102 \
    recipe.train_sampler.balance_mode=batch \
    +trainer.check_val_every_n_epoch=10

python scripts/build_tidylang_manifest.py eval \
        --corpus-root $SCRATCH/TidyVoiceX2_ASV/TidyVoiceX2_ASV \
        --output-dir $SCRATCH/data/manifests