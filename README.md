# Extractive Summarization 

**This code is the simple re-implementation for DistilSum / ThresSum / HAHSum / DifferSum / NLSSum.**

Some codes are borrowed from PreSumm(https://github.com/nlpyang/PreSumm)

## Requirements
```
datasets==1.17.0
line_profiler==3.4.0
nltk==3.4
numpy==1.20.3
pytorch_lightning==1.5.8
rouge_score==0.0.4
torch==1.10.2
transformers==4.15.0
wandb==0.12.10
```

* The datasets are from `HuggingFace Datasets`.
* The pre-trained language models are from `HuggingFace Transformers`.
* The `Trainer` is from `Pytorch Lightning`.
* The hyperparameter tuning is from `Wandb Sweep`.

## Commands

### Data Preprocess (Optional)

```shell
python datas/cnndm.py --num_proc 40 --max_pos 800
```
* This step will auto download the CNN/DM dataset from `HuggingFace Dataset Hub`.
* The dataset will be processed as `Apache Arrow Tabular Datasets`.
* The data preprocess is also involved in the training.

### Training and Evaluation

```shell
export HF_DATASETS_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs 5 --accumulate_grad_batches 2 --val_check_interval 500 --gpus 1 --lr 1e-5 --warmup_steps 30000 --max_pos 800
```
* `HF_DATASETS_OFFLINE=1` and ` export TRANSFORMERS_OFFLINE=1` are able to run `Transformers` in a firewalled or offline environment by only using local files.
* These hyperparameters are chosen randomly, and you can tune them by `Wandb Sweep`.

### Hyperparameter Tuning with Wandb Sweep (Optional)
```shell
wandb login             
wandb init              # Initialize your project repo
wandb sweep sweep.yaml  # returns your SWEEP_ID
wandb agent <sweep_id>  # start to tune the hyperparameters
```
* Wandb can tune the hyperparameters on different machines.


## Results

Results on CNN/DailyMail:

| Model            | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------------------|---------|---------|---------|
| DistilSum (Base) | 44.16   | 20.03   | 40.61   |
