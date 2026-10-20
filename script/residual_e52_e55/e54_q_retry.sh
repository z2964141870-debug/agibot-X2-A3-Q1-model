#!/bin/bash
set -u
cd /opt/data/private/yuanqi_ops
PY=/opt/data/private/hechengzhu/ZHY/data/environments/a3-sonic/bin/python
export E54_PYTHON=$PY
$PY e54_trial.py train --arm Q --tier 16 || exit 1
$PY e54_trial.py evaluate --arm Q --tier 16 --kind cloth || exit 1
$PY e54_trial.py evaluate --arm Q --tier 16 --kind official || exit 1
$PY e54_trial.py effect || exit 1
echo E54_Q_PIPELINE_DONE
