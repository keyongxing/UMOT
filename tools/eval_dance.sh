


set -x
set -o pipefail

# 硬编码 OUTPUT_DIR 到目标路径（原脚本通过参数 $1 传入，此处直接定义）
OUTPUT_DIR="/home/severs-s/kyx_use/pycharm_xinagmu/UMOT2/output"

# 清理 Python 编译文件
rmpyc() {
  find . -name "__pycache__" -exec rm -rf {} +
  find . -name "*.pyc" -exec rm -rf {} +
}

# 创建输出目录并进入
mkdir -p "$OUTPUT_DIR"
cp submit_dance.py "$OUTPUT_DIR"

pushd "$OUTPUT_DIR" || exit 1

cp ../UMOT.args .  # 添加此行：从上级目录复制 args 文件
args=$(cat UMOT.args)
popd || exit 1

# 运行评估命令，日志保存到 OUTPUT_DIR 下的 eval_val_dance_u70.log
python3 TrackEval-master/scripts/run_mot_challenge.py \
--BENCHMARK DanceTrack \
--SPLIT_TO_EVAL val \
--METRICS HOTA CLEAR Identity \
--GT_FOLDER "/home/severs-s/kyx_use/pycharm_xinagmu/UMOT-main/data/Dataset/mot/DanceTrack/val" \
--SEQMAP_FILE "/home/severs-s/kyx_use/pycharm_xinagmu/UMOT-main/TrackEval-master/trackeval/data/seqmaps/dancetrack_val.txt" \
--SKIP_SPLIT_FOL True \
--TRACKER_SUB_FOLDER . \
--TRACKERS_TO_EVAL tracker17 \
--TRACKERS_FOLDER "/home/severs-s/kyx_use/pycharm_xinagmu/UMOT2" \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 8 \
--PLOT_CURVES False \
| tee -a "$OUTPUT_DIR/eval_val_dance_umot.log"  # 确保路径指向 OUTPUT_DIR