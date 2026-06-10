#!/usr/bin/env bash
set -eo pipefail

# ============================================================
# MSPD-Net: Full Pipeline Script
# Step1: Initial Seed Generation with Contrastive Learning
# Step2: Refine with AffinityNet
# Step3: Segmentation training with DeepLab
# ============================================================

# -------------------- Configurable Variables --------------------
# Pretrained backbone for Step1 (contrastive training)
PRETRAINED_MODEL="../weight-files/resnet38_imagenet.pth"
# Pretrained backbone for Step2 (AffinityNet training, MXNet format)
AFF_PRETRAINED_MODEL="../weight-files/ilsvrc-cls_rna-a1_cls1000_ep-0001.params"

# Step1 trained weight (output of contrast_train.py -> ${SESSION_NAME}.pth)
CONTRAST_WEIGHT="${SESSION_NAME}.pth"

# Step2 trained weight (output of aff_train.py -> ${SESSION_NAME}_aff.pth)
AFF_WEIGHT="${SESSION_NAME}_aff.pth"


VOC12_ROOT="../VOCdevkit/VOC2012"
SESSION_NAME="mspd_net"

BATCH_SIZE=8
NUM_WORKERS=4

INFER_LIST="voc12/val.txt"
PSEUDO_INFER_LIST="voc12/trainaug.txt"

CAM_NPY_DIR="output/cam_npy"
CAM_PNG_DIR="output/cam_png"
CRF_PNG_DIR="output/crf_png"

CRF_TMP_DIR="output/crf_tmp"
LA_CRF_DIR="output/crf_la/4.00"
HA_CRF_DIR="output/crf_ha/32.00"

RW_DIR="output/rw"
PSEUDO_GT_DIR="output/pseudo_gt"

EVAL_LIST="VOC2012/ImageSets/Segmentation/val.txt"
GT_DIR="VOC2012/SegmentationClass"
EVAL_COMMENT="mspd_eval"
# -------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") [STEP]
Steps:
  setup       Print setup instructions and exit
  1           Step1: Contrastive Learning (train + infer + eval)
  1_train     Step1: Contrastive train only
  1_infer     Step1: Contrastive inference only
  1_eval      Step1: Evaluation only
  2_prepare   Step2: Prepare CRF for AffinityNet
  2_train     Step2: AffinityNet train
  2_infer     Step2: Random walk propagation
  2_pseudo    Step2: Pseudo mask generation
  3_train     Step3: DeepLab train
  3_test      Step3: DeepLab inference
  all         Run full pipeline (default)
  -h | help   Show this help
EOF
    exit 0
}

step=${1:-all}

require() {
    local var_name="$1"
    local var_value="${!var_name:-}"
    if [ -z "$var_value" ]; then
        echo "  [SKIP] $var_name is not set. Configure it at the top of the script."
        return 1
    fi
}

warn_dir() {
    local d="$1"
    if [ ! -d "$d" ]; then
        echo "  [SKIP] directory not found: $d"
        return 1
    fi
}

# ======================== Step 1 ========================
step1_train() {
    echo "[Step 1] Contrastive train..."
    require PRETRAINED_MODEL || return 0
    python contrast_train.py \
        --weights "$PRETRAINED_MODEL" \
        --voc12_root "$VOC12_ROOT" \
        --session_name "$SESSION_NAME" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS"
}

step1_infer() {
    echo "[Step 1] Contrastive inference..."
    require CONTRAST_WEIGHT || return 0
    mkdir -p "$CAM_NPY_DIR" "$CAM_PNG_DIR" "$CRF_PNG_DIR"
    python contrast_infer.py \
        --weights "$CONTRAST_WEIGHT" \
        --infer_list "$INFER_LIST" \
        --voc12_root "$VOC12_ROOT" \
        --num_workers "$NUM_WORKERS" \
        --out_cam "$CAM_NPY_DIR" \
        --out_cam_pred "$CAM_PNG_DIR" \
        --out_crf "$CRF_PNG_DIR"
}

step1_eval() {
    echo "[Step 1] Evaluation..."
    local ok=0
    if [ -d "$CAM_NPY_DIR" ]; then
        python eval.py \
            --list "$EVAL_LIST" \
            --predict_dir "$CAM_NPY_DIR" \
            --gt_dir "$GT_DIR" \
            --comment "$EVAL_COMMENT" \
            --type npy \
            --curve True && ok=1
    fi
    if [ -d "$CAM_PNG_DIR" ]; then
        python eval.py \
            --list "$EVAL_LIST" \
            --predict_dir "$CAM_PNG_DIR" \
            --gt_dir "$GT_DIR" \
            --comment "${EVAL_COMMENT}_png" \
            --type png \
            --curve True && ok=1
    fi
    [ "$ok" = 0 ] && echo "  [SKIP] no prediction dirs found ($CAM_NPY_DIR / $CAM_PNG_DIR)"
}

step1() {
    step1_train && step1_infer && step1_eval
}

# ======================== Step 2 ========================
step2_prepare() {
    echo "[Step 2] Prepare CRF for AffinityNet..."
    warn_dir "$CAM_NPY_DIR" || return 0
    mkdir -p "$CRF_TMP_DIR"

    echo "  -> LA CRF (alpha=4)..."
    python aff_prepare.py \
        --voc12_root "$VOC12_ROOT" \
        --cam_dir "$CAM_NPY_DIR" \
        --out_crf "${CRF_TMP_DIR}/la" \
        --alpha 4 \
        --infer_list "$PSEUDO_INFER_LIST" \
        --num_workers "$NUM_WORKERS"

    echo "  -> HA CRF (alpha=32)..."
    python aff_prepare.py \
        --voc12_root "$VOC12_ROOT" \
        --cam_dir "$CAM_NPY_DIR" \
        --out_crf "${CRF_TMP_DIR}/ha" \
        --alpha 32 \
        --infer_list "$PSEUDO_INFER_LIST" \
        --num_workers "$NUM_WORKERS"

    mkdir -p "$(dirname "$LA_CRF_DIR")" "$(dirname "$HA_CRF_DIR")"
    [ -d "${CRF_TMP_DIR}/la/4.00" ] && cp -r "${CRF_TMP_DIR}/la/4.00" "$(dirname "$LA_CRF_DIR")"
    [ -d "${CRF_TMP_DIR}/ha/32.00" ] && cp -r "${CRF_TMP_DIR}/ha/32.00" "$(dirname "$HA_CRF_DIR")"
}

step2_train() {
    echo "[Step 2] AffinityNet train..."
    local aff_w="${AFF_PRETRAINED_MODEL:-$PRETRAINED_MODEL}"
    if [ -z "$aff_w" ]; then
        echo "  [SKIP] no pretrained model (set PRETRAINED_MODEL or AFF_PRETRAINED_MODEL)"
        return 0
    fi
    warn_dir "$LA_CRF_DIR" || return 0
    warn_dir "$HA_CRF_DIR" || return 0
    python aff_train.py \
        --weights "$aff_w" \
        --voc12_root "$VOC12_ROOT" \
        --la_crf_dir "$LA_CRF_DIR" \
        --ha_crf_dir "$HA_CRF_DIR" \
        --session_name "${SESSION_NAME}_aff" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS"
}

step2_infer() {
    echo "[Step 2] Random walk propagation..."
    require AFF_WEIGHT || return 0
    warn_dir "$CAM_NPY_DIR" || return 0
    mkdir -p "$RW_DIR"
    python aff_infer.py \
        --weights "$AFF_WEIGHT" \
        --voc12_root "$VOC12_ROOT" \
        --infer_list "$INFER_LIST" \
        --cam_dir "$CAM_NPY_DIR" \
        --out_rw "$RW_DIR" \
        --num_workers "$NUM_WORKERS"
}

step2_pseudo() {
    echo "[Step 2] Pseudo mask generation (trainaug)..."
    require AFF_WEIGHT || return 0
    warn_dir "$CAM_NPY_DIR" || return 0
    mkdir -p "$PSEUDO_GT_DIR"
    python aff_infer.py \
        --weights "$AFF_WEIGHT" \
        --infer_list "$PSEUDO_INFER_LIST" \
        --cam_dir "$CAM_NPY_DIR" \
        --voc12_root "$VOC12_ROOT" \
        --out_rw "$PSEUDO_GT_DIR" \
        --num_workers "$NUM_WORKERS"
}

# ======================== Step 3 ========================
step3() {
    echo "[Step 3] DeepLab segmentation training..."
    cat <<EOF
  DeepLab training uses a separate codebase:
  https://github.com/YudeWang/semantic-segmentation-codebase

  Available configs:
$(ls segmentation/experiment/)

  Steps:
  1. Set DATA_PSEUDO_GT=$PSEUDO_GT_DIR in config.py
  2. cd segmentation/experiment/<config_dir>
  3. python train.py
  4. python test.py
EOF
}

# ======================== Setup ========================
step_setup() {
    cat <<EOF
=== MSPD-Net Setup ===

Required data:
  VOC2012/        Download PASCAL VOC 2012 dataset
  voc12/*.txt     Image split files (already provided)

Required pretrained weights (set variables at top of script):
  PRETRAINED_MODEL   Initial backbone weights
                     Download from: https://1drv.ms/u/s!AgGL9MGcRHv0mQSKoJ6CDU0cMjd2?e=dFlHgN

After Step1 training, set:
  CONTRAST_WEIGHT    = $SESSION_NAME.pth (output of contrast_train.py)

After Step2 training, set:
  AFF_WEIGHT         = ${SESSION_NAME}_aff.pth (output of aff_train.py)

Install dependencies:
  pip install torch torchvision tensorboardX pydensecrf imageio pandas tqdm

Run individual steps:
  bash run_pipeline.sh 1_train     # Step1: train
  bash run_pipeline.sh 1_infer     # Step1: inference
  bash run_pipeline.sh 2_prepare   # Step2: prepare CRF
  bash run_pipeline.sh 2_train     # Step2: train AffinityNet
  bash run_pipeline.sh 2_infer     # Step2: random walk
  bash run_pipeline.sh 2_pseudo    # Step2: pseudo mask generation
EOF
}

# ======================== Dispatch ========================
case "$step" in
    setup)      step_setup ;;
    1)          step1 ;;
    1_train)    step1_train ;;
    1_infer)    step1_infer ;;
    1_eval)     step1_eval ;;
    2_prepare)  step2_prepare ;;
    2_train)    step2_train ;;
    2_infer)    step2_infer ;;
    2_pseudo)   step2_pseudo ;;
    3)          step3 ;;
    3_train)    step3 ;;
    3_test)     step3 ;;
    all)
        echo "=== MSPD-Net Full Pipeline ==="
        echo "Run 'bash run_pipeline.sh setup' for setup instructions."
        echo ""
        step1
        step2_prepare
        step2_train
        step2_infer
        step2_pseudo
        step3
        echo ""
        echo "=== Pipeline finished ==="
        ;;
    -h | help) usage ;;
    *)
        echo "Unknown step: $step"
        usage
        ;;
esac
