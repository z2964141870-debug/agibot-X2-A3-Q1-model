#!/bin/bash
# AMASS-SMPLX 镜像分卷下载器(断点续传,串行)
DEST=/opt/data/private/yuanqi_ops/data/motionx
mkdir -p $DEST
cd $DEST
FILES="AMASS_smplxg.z01 AMASS_smplxg.z02 AMASS_smplxg.z03 AMASS_smplxg.z04 AMASS_smplxg.z05 AMASS_smplxg.z06 AMASS_smplxg.z07 AMASS_smplxg.z08 AMASS_smplxg.z09 AMASS_smplxg.z10 AMASS_smplxg.z11 AMASS_smplxg.z12 AMASS_smplxg.z13 AMASS_smplxg.z14 AMASS_smplxg.z15 AMASS_smplxg.z16 AMASS_smplxg.z17 AMASS_smplxg.z18 AMASS_smplxg.z19 AMASS_smplxg.z20 AMASS_smplxg.z21 AMASS_smplxg.z22 AMASS_smplxg.z23 AMASS_smplxg.z24 AMASS_smplxg.z25 AMASS_smplxg.z26 AMASS_smplxg.z27 AMASS_smplxg.z28 AMASS_smplxg.z29 AMASS_smplxg.zip"
for F in $FILES; do
  for TRY in 1 2 3 4 5; do
    curl -sL -C - --max-time 3600 --retry 3 -o $F "https://hf-mirror.com/datasets/SteveZeyuZhang/motionx/resolve/main/$F" 2>>dl_err.log && break
    sleep 10
  done
  echo "$(date +%T) $F $(du -h $F 2>/dev/null | cut -f1)" >> dl_progress.log
done
echo "$(date +%T) ALL_DOWNLOADS_DONE" >> dl_progress.log
