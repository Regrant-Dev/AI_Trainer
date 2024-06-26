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

