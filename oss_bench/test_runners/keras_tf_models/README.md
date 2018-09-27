# Running tensorflow/models
Hints for running code in tensorflow/models/official.  This may get stale and was kept to record 
the steps needed to setup these tests.

## Setup
The official models code needs some setup to work correctly.

**Install the requirements and set PYTHONPATH**

```bash
pip install --user -r official/requirements.txt
# from the root of the models repo, e.g. parent of official.
export PYTHONPATH=$(pwd)
```

## Running the examples

```bash
#fp32 real:
python imagenet_main.py --data_dir $PATH_TO_IMAGENET --model_dir $DESIRED_SNAPSHOT_PATH --batch_size $MAKE_THIS_BIG_ENOUGH_TO_FILL_THE_GPUS --num_gpus $NUM_GPUS --use_keras

#fp32 synthetic
python imagenet_main.py --data_dir $PATH_TO_IMAGENET --model_dir $DESIRED_SNAPSHOT_PATH --batch_size $MAKE_THIS_BIG_ENOUGH_TO_FILL_THE_GPUS --num_gpus $NUM_GPUS --use_synthetic_data --use_keras
```
