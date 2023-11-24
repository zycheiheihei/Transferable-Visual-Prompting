import os
import time
import clip
import torch
import torchvision
import wandb
from torchvision.datasets import CIFAR10
from torchvision.datasets.vision import StandardTransform
from torch.utils.data import Subset, random_split
from dataset import SampledCIFAR10
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import datetime
import torch.nn.functional as F
from scipy import stats as st
import argparse
import torchvision.models as models
from util.tool import refine_classname, topk, _convert_image_to_rgb, add_weight_decay
from util.get_index import get_index
from torchvision.transforms import (
    Compose,
    ToTensor,
    InterpolationMode,
)
from sklearn.metrics import roc_auc_score
import cv2
import json
    
    
class Pertubation_LLM(torch.nn.Module):
    def __init__(self, args, pad_h, pad_w, model_name, model_llm, clip_model, device):
        super().__init__()
        self.mask = torch.ones((3, 224, 224))
        self.mask[:, pad_h: 224 - pad_h, pad_w: 224 - pad_w] = 0

        delta = torch.zeros((3, 224, 224))
        delta.require_grad = True

        self.perturbation = torch.nn.Parameter(
            delta.float(), requires_grad=True)
        
        self.clip_model = clip_model
        
        self.model_name = model_name
        self.device = device
        if model_name=='minigpt-4':
            from util.minigpt4_utils import load_model, model_loss, CLS_PROMPT, model_loglikelihood_for_postfixes
            self.model = load_model(model_llm, device)
            self.loss = model_loss
            self.prompt = CLS_PROMPT
            self.probability = model_loglikelihood_for_postfixes
            
        elif model_name=='instructblip':
            from util.instructblip_utils import load_model, model_loss, model_loglikelihood_for_postfixes, CLS_PROMPT
            self.model = load_model(model_llm, device)
            self.probability = model_loglikelihood_for_postfixes
            self.loss = model_loss
            self.prompt = CLS_PROMPT
            
        elif model_name=='blip2':
            from util.blip2_utils import load_model, model_loss, model_loglikelihood_for_postfixes, CLS_PROMPT
            model_llm = 'flant5xl'
            self.model = load_model(model_llm, device)
            self.probability = model_loglikelihood_for_postfixes
            self.loss = model_loss
            self.prompt = CLS_PROMPT
            
        elif model_name=='vpgtrans':
            from util.vpgtrans_utils import load_model, model_loglikelihood_for_postfixes, model_loss, CLS_PROMPT
            self.model = load_model(model_llm, device)
            self.loss = model_loss
            self.prompt = CLS_PROMPT
            self.probability = model_loglikelihood_for_postfixes
            
        elif model_name=='bliva':
            from util.bliva_utils import load_model, model_loss, model_loglikelihood_for_postfixes, CLS_PROMPT
            self.model = load_model(model_llm, device)
            self.probability = model_loglikelihood_for_postfixes
            self.loss = model_loss
            self.prompt = CLS_PROMPT
        
        elif model_name=='visualglm':
            from util.visualglm_utils import load_model, model_loss, model_loglikelihood_for_postfixes, CLS_PROMPT
            self.model = load_model(model_llm, device)
            self.probability = model_loglikelihood_for_postfixes
            self.loss = model_loss
            self.prompt = CLS_PROMPT
            
    def forward(self, images, class_names, prompt=None):
        if prompt is None:
            prompt = "This is a photo of a"
        if type(prompt) is str:
            prompt = [prompt]*images.shape[0]
        assert len(prompt) == images.shape[0]
        samples = {"image": images, "text_input":prompt, "prompt":self.prompt, "text_output":class_names}
        loss = self.loss(self.model, samples)['loss']

        return loss
    
    def infer(self, images, class_names, prompt=None):
        if prompt is None:
            prompt = "This is a photo of a"
        if type(prompt) is str:
            prompt = [prompt]*images.shape[0]
        assert len(prompt) == images.shape[0]
        samples = {"image": images, "text_input":prompt, "prompt":self.prompt}
        probs = self.probability(self.model, samples, class_names)

        return probs


def parse_option():
    parser = argparse.ArgumentParser("Visual Prompting for CLIP")

    # training
    parser.add_argument(
        "--batch_size_train",
        type=int,
        default=16,
        help="batch_size")
    parser.add_argument(
        "--batch_size_eval",
        type=int,
        default=32,
        help="batch_size")
    parser.add_argument(
        "--num_workers", type=int, default=16, help="num of workers to use"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of training epoch5s"
    )

    # optimization
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=10,
        help="learning rate")

    # model
    parser.add_argument("--arch", type=str, default="ViT-B/32")
    parser.add_argument("--model_name", type=str, default="minigpt-4")
    parser.add_argument("--model_llm", type=str, default="vicuna7b")
    parser.add_argument("--llm_loss", action='store_true', default=True)
    parser.add_argument("--post_load_black", action='store_true', default=False)
    parser.add_argument("--target_models",
                        nargs='+',
                        default=[])
    parser.add_argument("--fca", type=float, default=0)
    parser.add_argument("--tse", type=float, default=0)
    parser.add_argument(
        "--prompt_size", type=int, default=30, help="size for visual prompts"
    )

    # dataset
    parser.add_argument(
        "--root",
        type=str,
        default=os.path.expanduser("../data"),
        help="dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="dataset")
    parser.add_argument(
        "--image_size",
        type=int,
        default=164,
        help="image size")

    # save
    parser.add_argument(
        "--save_path",
        type=str,
        default="./reproduce/",
        help="path to save models")

    # seed
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for initializing training"
    )

    # eval
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Perform evaluation only')

    parser.add_argument(
        "--checkpoint", type=str, help="The checkpoint of trained model"
    )

    # wandb
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="whether to use wandb")
    parser.add_argument(
        "--project",
        type=str,
        default="TVP",
        help="The name of wandb project name",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="cifar100",
        help="The name of wandb job name")
    parser.add_argument(
        "--entity", type=str, default="", help="Your user name of wandb"
    )
    parser.add_argument(
        "--roc_auc",
        action="store_true",
        help="whether to calculate roc auc")

    args = parser.parse_args()
    return args



def sample_indices(dataset, classes, few_shot, name):
    label_index = {}
    for k in classes:
        label_index[k] = []
    for i in range(len(dataset)):
        target = dataset[i][1]
        label_index[classes[target]].append(i)
    chosen_index = []
    print([len(label_index[k]) for k in label_index.keys()])
    for key in classes:
        chosen_index.extend(random.choices(label_index[key], k=few_shot))
        
    random.shuffle(chosen_index)
    
    return chosen_index



def sample_percent_indices(dataset, classes, percent, name):
    label_index = {}
    for k in classes:
        label_index[k] = []
    for i in range(len(dataset)):
        target = dataset[i][1]
        label_index[classes[target]].append(i)
    chosen_index = []
    for key in classes:
        chosen_index.extend(random.choices(label_index[key], k=int(percent*len(label_index[key]))))
        
    random.shuffle(chosen_index)
    
    return chosen_index



def main():
    args = parse_option()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    args.save_path = os.path.join(args.save_path, f'{args.model_name}_dataset_{args.dataset}_lr_{args.learning_rate}_FCA_{args.fca}_TSE_{args.tse}_seed_{args.seed}')

    # log setting
    log_wandb = args.use_wandb
    project = args.project
    job_name = args.job_name
    save_path = args.save_path
    
    if not os.path.exists(save_path) and not args.evaluate:
        os.makedirs(save_path)
    if log_wandb:
        wandb.init(
            project=str(project),
            name=str(args.dataset))

    # Load the clip model
    clip_model, preprocess = clip.load(args.arch, device)
    _, preprocess_test = clip.load(args.arch, device)
    
    del _

    # Prepare the dataset
    # Normalize the image and noise together
    normalization = preprocess.transforms[-1]
    preprocess_test.transforms.pop(-1)
    preprocess = Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Resize(
                args.image_size, interpolation=InterpolationMode.BICUBIC
            ),
            torchvision.transforms.RandomCrop(args.image_size),
            _convert_image_to_rgb,
            ToTensor(),
        ]
    )
    preprocess_test = Compose(
        [
            torchvision.transforms.Resize(
                args.image_size,
                interpolation=InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(
                size=(
                    args.image_size,
                    args.image_size)),
            _convert_image_to_rgb,
            ToTensor(),
        ])

    
    if args.dataset=="cifar10":
        train_set = CIFAR10(
            './data',
            download=True,
            train=True,
            transform=preprocess)
        classes_names = train_set.classes
        test_set = CIFAR10(
            './data',
            download=True,
            train=False,
            transform=preprocess_test)
        sampled_valset = SampledCIFAR10(
            0.05,
            './data',
            download=True,
            train=False,
            transform=preprocess_test
        )
    

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size_eval,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        sampled_valset,
        batch_size=args.batch_size_eval,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers
    )
   

    # Training setting
    epoch = args.epochs
    lr = args.learning_rate

    # Initialize the prompt
    prompt = Pertubation_LLM(args, args.prompt_size, args.prompt_size, args.model_name, args.model_llm, clip_model, "cuda:0")
    
    target_prompts = []
    if not args.post_load_black:
        for target in args.target_models:
            target_prompts.append(Pertubation_LLM(args, args.prompt_size, args.prompt_size, target, args.model_llm, None, "cuda:0"))
    else:
        target_prompts = args.target_models
    
    pad_length = int((224 - args.image_size) / 2)
    pad_dim = (pad_length, pad_length, pad_length, pad_length)

    # Optimizer setting
    prompt.model.requires_grad_(False)
    param_groups = add_weight_decay(prompt, 0.0, skip_list=("perturbation"))
    # print(param_groups)
    optimizer = torch.optim.SGD(param_groups, lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epoch)

    max_acc = 0
    max_auc = 0
    
    results = []
    # Begin training
    if not args.evaluate:
        if log_wandb:
            wandb.watch(prompt)
        
        print('Start Training')
        for e in range(epoch):
            val_log, train_top1, args = train_with_prompt(
                args,
                epoch=e,
                train_loader=train_loader,
                prompt=prompt,
                text_inputs=classes_names,
                pad_dim=pad_dim,
                criterion=criterion,
                optim=optimizer,
                normalization=normalization,
                device=device,
                val_loader=val_loader,
                target_prompts = target_prompts
            )
            schedule.step()
            results += val_log
            if (e+1)==args.epochs:
                test_item = {}
                test_acc1 = eval(
                    args,
                    test_loader=test_loader,
                    prompt=prompt,
                    pad_dim=pad_dim,
                    text_inputs=classes_names,
                    normalization=normalization,
                    device=device
                )
                test_item[prompt.model_name] = test_acc1
                
                for target_prompt in target_prompts:
                    if args.post_load_black:
                        target_prompt = Pertubation_LLM(args, args.prompt_size, args.prompt_size, target_prompt, args.model_llm, clip_model, "cuda:0")
                    target_prompt.perturbation.data = prompt.perturbation.data.clone()
                    black_acc = eval(
                        args,
                        test_loader=test_loader,
                        prompt=target_prompt,
                        pad_dim=pad_dim,
                        text_inputs=classes_names,
                        normalization=normalization,
                        device=device
                    )
                    test_item[target_prompt.model_name] = black_acc
                    if args.post_load_black:
                        del target_prompt
                results.append(test_item)
                print("Prompt Value: ", prompt.perturbation.abs().max())
                if test_acc1 > max_acc:
                    max_acc = test_acc1
                    model_state = prompt.state_dict()
                    save_dict = {"perturbation": model_state["perturbation"]}
                    save_path = args.save_path
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    torch.save(save_dict, save_path + "/checkpoint_best.pth")
                print("max acc is {}".format(str(max_acc)))
                if log_wandb:
                    log_stauts = {
                        "lr": optimizer.param_groups[0]["lr"],
                        "train_top1": train_top1,
                        "test_acc1": test_acc1,
                    }
                    wandb.log(log_stauts, step=e)
    # Begin testing
    else:
        print('Start Evaluating')
        
        if args.checkpoint is not None:
            checkpoint = args.checkpoint
            state_dict = torch.load(checkpoint, map_location="cpu")
            perturbation_state = prompt.state_dict()
            perturbation_state["perturbation"] = state_dict["perturbation"]
            prompt.load_state_dict(perturbation_state)
        
        test_acc1 = eval(
            args,
            test_loader=test_loader,
            prompt=prompt,
            pad_dim=pad_dim,
            text_inputs=classes_names,
            normalization=normalization,
            device=device
        )
        print("Overall accuracy for {} is {}".format(prompt.model_name, str(test_acc1)))

        


def train_with_prompt(
    args,
    epoch,
    train_loader,
    prompt,
    text_inputs,
    pad_dim,
    criterion,
    optim,
    normalization,
    device,
    val_loader,
    target_prompts
):
    device = prompt.device
    start_time = time.time()
    lr = optim.param_groups[0]["lr"]
    all_loss = []
    all_top1 = []
    validate_log = []
    idx = 0
        
    if prompt.model_name == 'minigpt-4':
        img_encoder = lambda x: prompt.model.encode_img(x)[0]

    for samples in tqdm(train_loader):
        images, labels = samples
        prompts = None

        if (idx+1)%1000==0:
            validate_item = {}
            print("Prompt Value: ", prompt.perturbation.abs().max())
            print(prompt.model_name)
            white_acc = eval(args, val_loader, prompt, pad_dim, text_inputs, normalization, device)
            validate_item[prompt.model_name] = white_acc
            if not args.post_load_black:
                for target_prompt in target_prompts:
                    print(target_prompt.model_name)
                    target_prompt.perturbation.data = prompt.perturbation.data.clone()
                    black_acc = eval(args, val_loader, target_prompt, pad_dim, text_inputs, normalization, device)
                    validate_item[target_prompt.model_name] = black_acc
            validate_log.append(validate_item)
        
        
        emb_loss = 0
        clip_loss = 0
        # Pad the image
        
        clean_images = F.pad(images, pad_dim, "constant", value=0)
        clean_images = clean_images.to(device)
        
        if args.fca > 0:
            input_images = normalization(clean_images)
            clean_batch_feats = img_encoder(input_images)
        
        
        images = F.pad(images, pad_dim, "constant", value=0)
        images = images.to(device)
        noise = prompt.perturbation.to(device)
        noise = noise.repeat(images.size(0), 1, 1, 1)
        noise.retain_grad()

        # Normalize the image and noise
        images = normalization(images + noise)
        images.require_grad = True


        targets = []
        for j in range(labels.shape[0]):
            targets.append(text_inputs[labels[j]])

        main_loss = prompt(images, targets, prompts)

        if args.fca>0:
            prompted_batch_feats = img_encoder(images)

            diff = prompted_batch_feats - clean_batch_feats
            emb_loss = torch.norm(diff, dim=-1).mean()
            
        if args.tse>0:
            captions = torch.cat([clip.tokenize(f"this is a photo of a {targets[j]}") for j in range(labels.shape[0])]).to(device)
            
            image_features = prompt.clip_model.encode_image(images)
            text_features = prompt.clip_model.encode_text(captions)
            norm_image_features = image_features / \
                image_features.norm(dim=-1, keepdim=True)
            norm_text_features = text_features / \
                text_features.norm(dim=-1, keepdim=True)
            similarities = (
                prompt.clip_model.logit_scale.exp()
                * norm_image_features
                @ norm_text_features.T
            )
            clip_loss = torch.diag(similarities).mean()

        prompt.clip_model.logit_scale.data = torch.clamp(prompt.clip_model.logit_scale.data, 0, 4.6052)
        
        loss = main_loss + args.fca*emb_loss - args.tse*clip_loss
        loss.backward()
        # update the perturbation
        grad_p_t = noise.grad
        grad_p_t = grad_p_t.mean(0).squeeze(0)
        g_norm = torch.norm(grad_p_t.view(-1), dim=0).view(1, 1, 1)
        scaled_g = grad_p_t / (g_norm + 1e-10)
        scaled_g_pad = scaled_g * prompt.mask.to(device)    
        updated_pad = scaled_g_pad * lr
        prompt.perturbation.data = prompt.perturbation.data - updated_pad.detach().cpu()
        prompt.zero_grad()

        all_loss.append(loss.float().detach().cpu().numpy())
        idx += 1
        

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(
        "At the {} epoch, the Lr is {}, the top1 is {} and training time  is {}".format(
            str(epoch), str(lr), str(
                np.mean(all_top1)), total_time_str))

    return validate_log, np.mean(all_top1), args

    

@torch.no_grad()
def eval(
        args,
        test_loader,
        prompt,
        pad_dim,
        text_inputs,
        normalization,
        device):
    device = prompt.device
    start_time = time.time()
    all_top1, all_top5 = [], []
    all_labels = []
    all_scores = []
    all_mean_logits = []
    all_gt_logits = []
    print("starting evaluation")
    for samples in tqdm(test_loader):
        with torch.no_grad():
            images, labels=samples
            prompts = None
            
            images = F.pad(images, pad_dim, "constant", value=0)
            images = images.to(device)
            noise = prompt.perturbation.to(device)

            images = normalization(images + noise)
            
            probs = prompt.infer(images, text_inputs, prompts)
            
            for k in range(images.shape[0]):
                all_mean_logits.append(probs[k].mean().item())
                all_gt_logits.append(probs[k][int(labels[k])].item())
                
            if len(text_inputs)>5:
                top1, top5 = topk(probs, (labels).to(device), ks=(1, 5))
                all_top1.extend(top1.cpu())
                all_top5.extend(top5.cpu())
            else:
                top1 = topk(probs, (labels).to(device))
                all_top1.extend(top1[0].cpu())
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Testing time {}".format(total_time_str))
    print(f"top1 {np.mean(all_top1):.2%}, " f"top5 {np.mean(all_top5):.2%}")
    print(f"Average Mean Logits: {np.mean(all_mean_logits)}, Average GT Logits: {np.mean(all_gt_logits)}")
    return np.mean(all_top1)


if __name__ == "__main__":
    main()
