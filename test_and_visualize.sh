#!/bin/bash -i
curdir=`pwd`
export PYTHONPATH=$PYTHONPATH:$curdir/video_backbone/TSP
export PYTHONPATH=$PYTHONPATH:$curdir/video_backbone/TSP/data
export PYTHONPATH=$PYTHONPATH:$curdir/video_backbone/TSP/extract_features
export PYTHONPATH=$PYTHONPATH:$curdir/visualization

DATA_PATH=$1 # path of the raw video folder
OUTPUT_FOLDER=$2 # path of the output folder to save generated captions
PDVC_MODEL_PATH=$3
OUTPUT_LANGUAGE=$4

if [ -z "$DATA_PATH" ]; then
    echo "DATA_PATH variable is not set."
    echo "Please set DATA_PATH to the folder containing the videos you want to process."
    exit 1
fi

if [ -z "$OUTPUT_FOLDER" ]; then
    echo "OUTPUT_FOLDER variable is not set."
      echo "Please set OUTPUT_FOLDER to the folder you want to save generate captions."
    exit 1
    exit 1
fi

if [ -z "$PDVC_MODEL_PATH" ]; then
    echo "PDVC_MODEL_PATH variable is not set."
    echo "Please set the pretrained PDVC model path (only support PDVC with TSP features)."
    exit 1
fi

####################################################################################
########################## PARAMETERS THAT NEED TO BE SET ##########################
####################################################################################

METADATA_CSV_FILENAME=$DATA_PATH/"metadata.csv" # path/to/metadata/csv/file. Use the ones provided in the data folder.
RELEASED_CHECKPOINT=r2plus1d_34-tsp_on_activitynet


# Choose the stride between clips, e.g. 16 for non-overlapping clips and 1 for dense overlapping clips
STRIDE=16

# Optional: Split the videos into multiple shards for parallel feature extraction
# Increase the number of shards and run this script independently on separate GPU devices,
# each with a different SHARD_ID from 0 to NUM_SHARDS-1.
# Each shard will process (num_videos / NUM_SHARDS) videos.
SHARD_ID=0
NUM_SHARDS=1
DEVICE=cuda
WORKER_NUM=8

echo "START GENERATE METADATA"
python video_backbone/TSP/data/generate_metadata_csv.py --video-folder $DATA_PATH --output-csv $METADATA_CSV_FILENAME

FEATURE_DIR=$OUTPUT_FOLDER/${RELEASED_CHECKPOINT}_stride_${STRIDE}/
mkdir -p $OUTPUT_DIR

echo "START EXTRACT VIDEO FEATURES"
python video_backbone/TSP/extract_features/extract_features.py \
--data-path $DATA_PATH \
--metadata-csv-filename $METADATA_CSV_FILENAME \
--released-checkpoint $RELEASED_CHECKPOINT \
--stride $STRIDE \
--shard-id $SHARD_ID \
--num-shards $NUM_SHARDS \
--device $DEVICE \
--output-dir $FEATURE_DIR \
--workers $WORKER_NUM

echo "START Dense-Captioning"
python eval.py --eval_mode test --eval_save_dir $OUTPUT_FOLDER --eval_folder generated_captions --eval_model_path $PDVC_MODEL_PATH --test_video_feature_folder $FEATURE_DIR --test_video_meta_data_csv_path $METADATA_CSV_FILENAME

echo "START VISUALIZATION"
python visualization/visualization.py --input_mp4_folder $DATA_PATH --output_mp4_folder  $OUTPUT_FOLDER/vis_videos --dvc_file $OUTPUT_FOLDER/generated_captions/dvc_results.json --output_language $OUTPUT_LANGUAGE
