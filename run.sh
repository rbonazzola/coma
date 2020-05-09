#!/bin/bash

source activate coma

DEVICE_ID=0

for YAML in `cat runs_yamls.txt`; do
  LOGS_FILE=${YAML##*/}
  LOGS_FILE=/MULTIX/DATA/INPUT/disk_2/coma/Cardio/output/logs/${LOGS_FILE%%yaml}log
  python main.py --config_yaml yamls/$YAML --device_id $DEVICE_ID &> $LOGS_FILE &
  DEVICE_ID=$((DEVICE_ID+1))
  # aws s3 sync /MULTIX/DATA/INPUT/disk_2/coma/Cardio/output s3://users-rbonazzola/coma/output/
done
