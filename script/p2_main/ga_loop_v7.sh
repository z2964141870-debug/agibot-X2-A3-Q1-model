#!/bin/bash
BASE=/opt/data/private/yuanqi_ops
PY=/opt/data/private/hechengzhu/ZHY/data/environments/a3-sonic/bin/python
S2S=/opt/data/private/hechengzhu/ZHY/script/vendor/sonic_for_a3/gear_sonic/scripts/sim2sim_a3_mujoco.py
MJCF=/opt/data/private/hechengzhu/ZHY/script/vendor/sonic_for_a3/gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml
CK035=/opt/data/private/hechengzhu/ZHY/data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt
CK9C=$BASE/data/training/p2_main/m9c/checkpoints/milestone_128/model_step_000128.pt
CK11=$BASE/data/training/p2_main/m11d/checkpoints/milestone_32/model_step_000032.pt
OUT=$BASE/data/experiments/p2_main/gen_audit_v7
mkdir -p $OUT
for SEQ in $BASE/data/gen_audit/sequences_v7/*/; do
  S=$(basename $SEQ)
  for ARM in 035 m9c m11d; do
    case $ARM in
      035) CK=$CK035;; m9c) CK=$CK9C;; m11d) CK=$CK11;;
    esac
    MOUT=$OUT/${S}__${ARM}.metrics.json
    [ -f $MOUT ] && continue
    $PY $S2S --checkpoint $CK --motion $SEQ/reference.csv --encoder-mode a3_fast --csv-source-fps 30 --csv-frame-stride 1 --mjcf $MJCF --batch-once --action-delay-ms 0 --metrics-out $MOUT >> $BASE/logs/p2_main/gen_audit_v7.log 2>&1
    echo "RC=$? $S $ARM" >> $BASE/logs/p2_main/gen_audit_v7.log
  done
done
echo V7_ALL_DONE >> $BASE/logs/p2_main/gen_audit_v7.log
