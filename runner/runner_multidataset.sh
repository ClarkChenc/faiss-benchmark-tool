#!/usr/bin/env bash
set -x

# dataset_list=(StreamAnnRecallV13_100w StreamAnnRecallV13_300w StreamAnnRecallV13_500w StreamAnnRecallV13_1000w)
dataset_list=(StreamAnnRecallV13_1000w StreamAnnRecallV14_1000w SugAnnV7_1000w SugAnnV8_1000w GoodsAll64SPU_1000w U2I312_1000w)

CONFIG_TEMPLATE_PATH="./runner_config/config_hnswsplit_template.yaml"
CONFIG_PATH="./runner_config/config_hnswsplit.yaml"

# 测试不同数据集下 keep_indegree_rate 对 recall 的影响
OUTPUT_PATH="./runner_split_recall_keep_indegree_rate.log"
rm -rf ${OUTPUT_PATH}
cp -rf ${CONFIG_TEMPLATE_PATH} ${CONFIG_PATH}
for dataset in "${dataset_list[@]}";do
  echo $dataset
  sed -i "s/dataset: .*/dataset: \"${dataset}\"/" ${CONFIG_PATH}
  val_list=(0.3 0.1 0.01)
  for val in "${val_list[@]}";do
    echo "use val: ${val}"
    sed -i "s/keep_indegree_rate: .*/keep_indegree_rate: $val/" ${CONFIG_PATH}
    python main.py --config ${CONFIG_PATH} >> ${OUTPUT_PATH} 2>&1
  done
done

# 测试不同数据集下 seg_num 对 recall 的影响
OUTPUT_PATH="./runner_split_recall_seg_num.log"
rm -rf ${OUTPUT_PATH}
cp -rf ${CONFIG_TEMPLATE_PATH} ${CONFIG_PATH}
for dataset in "${dataset_list[@]}";do
  echo $dataset
  sed -i "s/dataset: .*/dataset: \"${dataset}\"/" ${CONFIG_PATH}
  val_list=(3 5 10)
  for val in "${val_list[@]}";do
    echo "use val: ${val}"
    sed -i "s/seg_num: .*/seg_num: $val/" ${CONFIG_PATH}
    python main.py --config ${CONFIG_PATH} >> ${OUTPUT_PATH} 2>&1
  done
done

# 测试不同数据集下 merge_ratio 对 recall 的影响
OUTPUT_PATH="./runner_split_recall_merge_ratio.log"
rm -rf ${OUTPUT_PATH}
cp -rf ${CONFIG_TEMPLATE_PATH} ${CONFIG_PATH}
for dataset in "${dataset_list[@]}";do
  echo $dataset
  sed -i "s/dataset: .*/dataset: \"${dataset}\"/" ${CONFIG_PATH}
  val_list=(1 0.5 0.3)
  for val in "${val_list[@]}";do
    echo "use val: ${val}"
    sed -i "s/merge_ratio: .*/merge_ratio: $val/" ${CONFIG_PATH}
    python main.py --config ${CONFIG_PATH} >> ${OUTPUT_PATH} 2>&1
  done
done


