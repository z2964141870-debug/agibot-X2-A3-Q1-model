#!/bin/bash
BASE=/opt/data/private/yuanqi_ops
PY=/opt/data/private/hechengzhu/ZHY/data/environments/a3-sonic/bin/python
S2S=/opt/data/private/hechengzhu/ZHY/script/vendor/sonic_for_a3/gear_sonic/scripts/sim2sim_a3_mujoco.py
MJCF=/opt/data/private/hechengzhu/ZHY/script/vendor/sonic_for_a3/gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml
SEQDIR=/opt/data/private/hechengzhu/ZHY/data/experiments/a3_zhy_smpl_hull_20261013_E23/sequences
CK035=/opt/data/private/hechengzhu/ZHY/data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt
CK11=$BASE/data/training/p2_main/m11d/checkpoints/milestone_32/model_step_000032.pt
OUT=$BASE/data/experiments/p2_main/perturb_audit
mkdir -p $OUT
MOTIONS="G_0113_zj_train__183926_zj_walk G_0113_zj_train__192333_zj_warmUp G_20251226_fjw__fjw_140008_walk G_20251226_zls__zls_155013_walk G_20251229_yy__yy_125106_walk G_20251229_zzx__zzx_111505_walk"
for DELAY in 20 40 80; do
  for M in $MOTIONS; do
    for ARM in 035 m11d; do
      case $ARM in
        035) CK=$CK035;; m11d) CK=$CK11;;
      esac
      MOUT=$OUT/${M}__${ARM}__d${DELAY}.metrics.json
      [ -f $MOUT ] && continue
      $PY $S2S --checkpoint $CK --motion $SEQDIR/$M/reference.csv --encoder-mode a3_fast --csv-source-fps 30 --csv-frame-stride 1 --mjcf $MJCF --batch-once --action-delay-ms $DELAY --metrics-out $MOUT >> $BASE/logs/p2_main/perturb_audit.log 2>&1
      echo "RC=$? $M $ARM d$DELAY" >> $BASE/logs/p2_main/perturb_audit.log
    done
  done
done
echo PERTURB_ALL_DONE >> $BASE/logs/p2_main/perturb_audit.log