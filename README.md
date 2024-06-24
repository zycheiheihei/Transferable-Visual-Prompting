# Transferable Visual Prompting for Multimodal Large Language Models



### Installation

1. Create the virtual environment for the project.
```
cd Transferable_VP_MLLM
conda create -n transvp python=3.11
pip install -r requirements.txt
```

2. Prepare the model weights

Put the model weights under `./model_weights`

* MiniGPT-4: Follow [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) and prepare the `MiniGPT-4-Vicuna-V0-7B`
* InstructBLIP: Follow [LAVIS](https://github.com/salesforce/LAVIS) and prepare the `InstructBLIP-Vicuna-7b-v1.1`
* BLIP2: Follow [LAVIS](https://github.com/salesforce/LAVIS) and prepare the `BLIP2-FlanT5-xl`
* VPGTrans: Follow MiniGPT-4 and prepare `Vicuna-v0-7B` as LLM
* BLIVA: Follow [BLIVA](https://github.com/mlpc-ucsd/BLIVA#prepare-weight) and prepare `BLIVA-Vicuna-7B`
* VisualGLM-6B: No special operation needed.

### To Reproduce Reproduced Results

1. On CIFAR10
```
python transfer_cls.py --dataset cifar10 --model_name minigpt-4 --target_models instructblip blip2 --learning_rate 10 --fca 0.005 --tse 0.001 --epochs 1
```

2. Inference with a model
Specify the path to checkpoint if you want to evaluate on the dataset with trained prompt. A reproducible checkpoint is placed in `save/checkpoint_best.pth`.
```
python transfer_cls.py --dataset cifar10 --model_name minigpt-4 --evaluate --checkpoint $PATH_TO_PROMPT
```

