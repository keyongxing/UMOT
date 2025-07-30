


set -x
set -o pipefail

OUTPUT_DIR=$1

# 清理 Python 编译文件
rmpyc() {
  find . -name "__pycache__" -exec rm -rf {} +
  find . -name "*.pyc" -exec rm -rf {} +
}

# 创建输出目录并进入
mkdir -p "$OUTPUT_DIR"
cp submit_dance.py "$OUTPUT_DIR"

pushd "$OUTPUT_DIR" || exit 1
args=$(cat motrv2.args)
popd || exit 1



python3    TrackEval-master/scripts/run_mot_challenge.py \
--BENCHMARK MOT17 \
--SPLIT_TO_EVAL train \
--METRICS HOTA CLEAR Identity \
--GT_FOLDER "/home/severs-s/kyx_use/pycharm_xinagmu/UMOT2/data/Dataset/mot/MOT17/images/train" \
--SEQMAP_FILE "/home/severs-s/kyx_use/pycharm_xinagmu/UMOT2/TrackEval-master/trackeval/data/seqmaps/MOT17_train.txt" \
--SKIP_SPLIT_FOL True \
--TRACKER_SUB_FOLDER . \
--TRACKERS_TO_EVAL tracker_mot17_motr \
--TRACKERS_FOLDER "/home/severs-s/kyx_use/pycharm_xinagmu/UMOT2/tracker" \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 8 \
--PLOT_CURVES False \
| tee -a $OUTPUT_DIR/eval_train_mot17_motr.log

