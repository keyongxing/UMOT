
set -x
set -o pipefail

args=$(cat configs/motrv2.args)
python3 /home/severs-s/kyx_use/pycharm_xinagmu/UMOT2/submit_dance.py  ${args} --exp_name tracker24 --resume  /home/severs-s/kyx_use/pycharm_xinagmu/UMOT2/exps/motrv2/run72/checkpoint0000.pth