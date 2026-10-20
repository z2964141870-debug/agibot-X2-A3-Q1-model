#!/bin/bash
BASE=/opt/data/private/yuanqi_ops
EVDIR=/opt/data/private/hechengzhu/ZHY/script/vendor/sonic_for_a3/gear_sonic/scripts
PY=/opt/data/private/hechengzhu/ZHY/data/environments/a3-sonic/bin/python
while pgrep -f a3_035_finetune_m11c >/dev/null; do sleep 20; done
if ls $BASE/data/training/p2_main/m11c/checkpoints/milestone_128/model_step_000128.pt >/dev/null 2>&1 && [ ! -d $BASE/data/experiments/p2_main/m11c_eval ]; then
  sed -e "s|p2_main/m11/checkpoints/milestone_32/model_step_000032.pt|p2_main/m11c/checkpoints/milestone_128/model_step_000128.pt|" -e "s|p2_main/m11_eval|p2_main/m11c_eval|" -e "s/P2-M11/P2-M11C/g" $BASE/script/p2_main/p2_m11_eval.sh > $BASE/script/p2_main/p2_m11c_eval.sh
  chmod +x $BASE/script/p2_main/p2_m11c_eval.sh
  bash $BASE/script/p2_main/p2_m11c_eval.sh > $BASE/logs/p2_main/m11c_eval.log 2>&1
fi
CK035=/opt/data/private/hechengzhu/ZHY/data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt
CK9C=$BASE/data/training/p2_main/m9c/checkpoints/milestone_128/model_step_000128.pt
CK11=$BASE/data/training/p2_main/m11/checkpoints/milestone_32/model_step_000032.pt
OUT=$BASE/data/experiments/p2_main/gen_audit
mkdir -p $OUT
find $OUT -name "*.metrics.json" -size -100c -delete 2>/dev/null
for SEQ in $BASE/data/gen_audit/sequences_30s/*/; do
  S=$(basename $SEQ)
  for ARM in 035 m9c m11; do
    case $ARM in
      035) CK=$CK035;; m9c) CK=$CK9C;; m11) CK=$CK11;;
    esac
    MOUT=$OUT/${S}__${ARM}.metrics.json
    [ -f $MOUT ] && continue
    cd $EVDIR || exit 1
    $PY $EVDIR/sim2sim_a3_mujoco.py --checkpoint $CK --motion $SEQ/reference.csv --encoder-mode a3_fast --csv-source-fps 30 --csv-frame-stride 1 --mjcf a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml --batch-once --action-delay-ms 0 --metrics-out $MOUT >> $BASE/logs/p2_main/gen_audit_eval.log 2>&1
    RC=$?
    echo "RC=$RC $S $ARM $( [ -f $MOUT ] && echo OK || echo FAIL )" >> $BASE/logs/p2_main/gen_audit_eval.log
  done
done
echo ALL_DONE_V3 >> $BASE/logs/p2_main/gen_audit_eval.log
