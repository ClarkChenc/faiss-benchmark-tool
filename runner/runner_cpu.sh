#!/usr/bin/env bash
set -x
set -euo pipefail
OUTPUT_PATH="./runner_cpu.log"
rm -rf ${OUTPUT_PATH}

CONFIG_PATH="./runner_config/config_cpu.yaml"
for i in {1..10};do
  let new_thread_num=$((i * 10))
  sed -i "s/num_threads: .*/num_threads: $new_thread_num/" ${CONFIG_PATH}
  echo "running ${new_thread_num}" >> ${OUTPUT_PATH}
  python main.py --config ${CONFIG_PATH} >> ${OUTPUT_PATH} 2>&1
done
