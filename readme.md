### LLM Training
# Step 1
go to app folder
cd /app

# Step 2

Run the preprocess axolotl function to tokenize dataset within the container with: 
python -m axolotl.cli.preprocess /app/config.yaml

# Step 3

Start the training with: 
accelerate launch -m axolotl.cli.train /app/config.yaml


### Voice Training
# Step 1
Run:  docker exec -it trainer-app-1 /bin/bash

Activate environment:
source ~/piper/src/python/.venv/bin/activate

Go to python dir:
cd ~/piper/src/python/

Edit requirements txt Replace everything in there with this:
cython>=0.29.0,<1
librosa>=0.9.2,<1
piper-phonemize~=1.1.0
numpy>=1.19.0
onnxruntime>=1.11.0
pytorch-lightning~=1.9.0
onnx

Run:
pip install -e

Run:
build_monotonic_align.sh

# Step 2
Download existing model to fine tune from
For highquality:
wget https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/high/epoch%3D2218-step%3D838782.ckpt -O ~/piper/epoch=2218-step=838782.ckpt
* Pass in the "--quality high" flag for this one

For normal:
wget https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt -O ~/piper/epoch=2164-step=1355540.ckpt

# Step 3
Copy over the dataset
cp -r /app/dataset ~/piper/my-dataset

# Step 4
cd ~/piper/src/python/

# Step 5
Run the preprocess

python3.10 -m piper_train.preprocess \
  --language en \
  --input-dir ~/piper/my-dataset \
  --output-dir ~/piper/my-training \
  --dataset-format ljspeech \
  --single-speaker \
  --sample-rate 22050

# Step 6
Run the training

* Modify batches based on how vram you have
* Realistically you only need 1000 epochs for fine tunes and 2000 epochs for training from scratch

python3.10 -m piper_train \
    --dataset-dir ~/piper/my-training \
    --accelerator 'gpu' \
    --devices 1 \
    --batch-size 12 \
    --validation-split 0.0 \
    --num-test-examples 0 \
    --max_epochs 6000 \
    --resume_from_checkpoint ~/piper/epoch=2218-step=838782.ckpt \
    --checkpoint-epochs 100 \
    --precision 32 \
    --quality high

# Step 7
Monitor training with tensorboard

* Open new terminal exec -it into the container
    *  docker exec -it trainer-app-1 /bin/bash

* Activate Env
source ~/piper/src/python/.venv/bin/activate

* Run tensorboard --logdir ~piper/my-training/lightning_logs

* Open url