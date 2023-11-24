from typing import Callable, Optional, Any, List
from torchvision.datasets import CIFAR100, CIFAR10, ImageFolder
from torch.utils.data import Dataset
import random
import numpy as np
import json
import os
from util.utils import pil_loader, remove_special_chars
from torchvision import transforms
from PIL import Image


class SampledCIFAR10(CIFAR10):
    def __init__(self, percentage:float, root: str, train: bool = True, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, download: bool = False) -> None:
        super().__init__(root, train, transform, target_transform, download)
        
        print("Loading Sampled Training Dataset...")
        label_index = {}
        for k in self.classes:
            label_index[k] = []
        for i in range(len(self)):
            target = self.targets[i]
            label_index[self.classes[target]].append(i)
        chosen_index = []
        for key in self.classes:
            chosen_index.extend(random.choices(label_index[key], k=int(percentage*len(label_index[key]))))
            
        random.shuffle(chosen_index)
        
        data = []
        targets = []
        for idx in chosen_index:
            img = self.data[idx]
            label = self.targets[idx]
            data.append(img[np.newaxis,:,:,:])
            targets.append(label)
        
        self.data = np.vstack(data)
        self.targets = targets
        
        print("Size of sampled dataset: ", len(self))
        

class CIFAR10Corruption(Dataset):
    def __init__(self, root, corruption, severity, transform=None):
        self.corruption = corruption
        self.severity = severity
        self.transform = transform
        self.labels = np.load(os.path.join(root, "labels.npy"))
        self.images = np.load(os.path.join(root, f"{corruption}.npy"))
        self.images = self.images[(self.severity-1)*10000:self.severity*10000]
        self.labels = self.labels[(self.severity-1)*10000:self.severity*10000]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,index):
        img = self.images[index]
        image = Image.fromarray(img)
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[index]
        
        
class SampledCIFAR100(CIFAR100):
    def __init__(self, percentage:float, root: str, train: bool = True, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, download: bool = False) -> None:
        super().__init__(root, train, transform, target_transform, download)
        
        print("Loading Sampled Training Dataset...")
        label_index = {}
        for k in self.classes:
            label_index[k] = []
        for i in range(len(self)):
            target = self.targets[i]
            label_index[self.classes[target]].append(i)
        chosen_index = []
        for key in self.classes:
            chosen_index.extend(random.choices(label_index[key], k=int(percentage*len(label_index[key]))))
            
        random.shuffle(chosen_index)
        
        data = []
        targets = []
        for idx in chosen_index:
            img = self.data[idx]
            label = self.targets[idx]
            data.append(img[np.newaxis,:,:,:])
            targets.append(label)
        
        self.data = np.vstack(data)
        self.targets = targets
        
        print("Size of sampled dataset: ", len(self))
        
        
        
class HatefulMemesDataset(Dataset):
    def __init__(self, root, split="dev", transform=None):
        self.split = split
        self.image_dir_path = os.path.join(root, "facebook-hateful-memes/img/")
        if type(split) is str:
            split = [split]
        self.annotations = []
        for anno in split:
            annotations_path = os.path.join(os.path.join(self.image_dir_path, os.path.pardir), anno+".jsonl")
            with open(annotations_path, "r") as f:
                self.annotations.extend([json.loads(line) for line in f])
        self.classes=["no", "yes"]
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_dir_path, annotation["img"].split("/")[-1])
        img = pil_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        
        return img, annotation["label"], f'This is an image with: "{annotation["text"]}" written on it. Is it hateful?'
    
    
    

class CLEVRDataset(Dataset):
    def __init__(self, root, split='val', transform=None) -> None:
        self.split = split
        self.dataset_dir = os.path.join(root, "CLEVR_v1.0")
        self.image_path = os.path.join(os.path.join(self.dataset_dir, "images"), split)
        self.ques_file = os.path.join(os.path.join(self.dataset_dir, "questions"), f"CLEVR_{split}_questions.json")
        self.scene_file = os.path.join(os.path.join(self.dataset_dir, "scenes"), f"CLEVR_{split}_scenes.json")
        self.transform = transform
        
        
        
def _process_punctuation(inText):
    import re
    outText = inText
    punct = [
        ';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
        '>', '<', '@', '`', ',', '?', '!'
    ]
    commaStrip = re.compile('(\d)(,)(\d)')  # noqa: W605
    periodStrip = re.compile('(?!<=\d)(\.)(?!\d)')  # noqa: W605
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(
                commaStrip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub('', outText, re.UNICODE)
    return outText


def _process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    articles = ['a', 'an', 'the']
    manualMap = {
        'none': '0',
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
    }
    contractions = {
        'aint': "ain't",
        'arent': "aren't",
        'cant': "can't",
        'couldve': "could've",
        'couldnt': "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        'didnt': "didn't",
        'doesnt': "doesn't",
        'dont': "don't",
        'hadnt': "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        'hasnt': "hasn't",
        'havent': "haven't",
        'hed': "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        'hes': "he's",
        'howd': "how'd",
        'howll': "how'll",
        'hows': "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        'Im': "I'm",
        'Ive': "I've",
        'isnt': "isn't",
        'itd': "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        'itll': "it'll",
        "let's": "let's",
        'maam': "ma'am",
        'mightnt': "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        'mightve': "might've",
        'mustnt': "mustn't",
        'mustve': "must've",
        'neednt': "needn't",
        'notve': "not've",
        'oclock': "o'clock",
        'oughtnt': "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        'shant': "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        'shouldve': "should've",
        'shouldnt': "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": 'somebodyd',
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        'somebodyll': "somebody'll",
        'somebodys': "somebody's",
        'someoned': "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        'someonell': "someone'll",
        'someones': "someone's",
        'somethingd': "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        'somethingll': "something'll",
        'thats': "that's",
        'thered': "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        'therere': "there're",
        'theres': "there's",
        'theyd': "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        'theyll': "they'll",
        'theyre': "they're",
        'theyve': "they've",
        'twas': "'twas",
        'wasnt': "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        'weve': "we've",
        'werent': "weren't",
        'whatll': "what'll",
        'whatre': "what're",
        'whats': "what's",
        'whatve': "what've",
        'whens': "when's",
        'whered': "where'd",
        'wheres': "where's",
        'whereve': "where've",
        'whod': "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        'wholl': "who'll",
        'whos': "who's",
        'whove': "who've",
        'whyll': "why'll",
        'whyre': "why're",
        'whys': "why's",
        'wont': "won't",
        'wouldve': "would've",
        'wouldnt': "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        'yall': "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        'youd': "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        'youll': "you'll",
        'youre': "you're",
        'youve': "you've",
    }
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText 
        
        
        
class CLEVRCountingDataset(CLEVRDataset):
    
    def __init__(self, root, split='val', transform=None) -> None:
        super().__init__(root, split, transform)
        self.scene_annotations = json.load(open(self.scene_file))["scenes"]
        classes = set()
        for scene in self.scene_annotations:
            classes.add(len(scene["objects"]))
        self.classes = [str(c) for c in classes]
        

    def __len__(self):
        return len(self.scene_annotations)
    
    def __getitem__(self, index) -> Any:
        scene = self.scene_annotations[index]
        img_path = os.path.join(self.image_path, scene["image_filename"])
        img = pil_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.classes.index(str(len(scene['objects']))), "How many objects are there in this image? Answer with a single number.", scene["image_filename"]
    
    
    def eval_vqa(self, answer, gt_answers):
        has_word = 0
        num = 0
        for ans, gt in zip(answer, gt_answers):
            ans = remove_special_chars(ans).lower()
            ans = _process_digit_article(ans)
            if gt in ans:
                has_word+=1
            num+=1
            
        return has_word/num*100
    
    
    
    
class POPEDataset(Dataset):
    def __init__(self, root, split="val", type='adversarial', transform=None):
        self.split = split
        self.image_dir_path = os.path.join(root, 'coco2014/{}2014'.format(split))
        self.annotations_path = f'./data/POPE/{split}/coco_{split}_pope_{type}.json'
        
        self.annotations = []
        
        f = open(self.annotations_path,'r')
        for line in f:
            self.annotations.append(json.loads(line))
        
        if self.split=='val':
            self.annotations = self.annotations[:int(len(self.annotations)/2)]
        self.classes=["no", "yes"]
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_dir_path, annotation["image"])
        img = pil_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.split=='val':
            return img, self.classes.index(annotation["label"]), annotation['text']
        else:
            return img, self.classes.index(annotation["label"]), (annotation['text'], ', '.join(annotation['objects']))
        

        
        
class ImageNette(ImageFolder):
    def __init__(self, root, split='train', transform=None):
        super().__init__(os.path.join(root,split), transform=transform)
        self.class_names = ['']*len(self.classes)
        f = open(os.path.join(root, '../data_miniimagenet/imagenet_class_index.json'),'r')
        dic = json.load(f)
        mark_to_name = {}
        for k in dic.keys():
            mark_to_name[dic[k][0]] = dic[k][1]
        for tag in self.classes:
            self.class_names[self.class_to_idx[tag]] = ' '.join(mark_to_name[tag].split('_'))
            

class ImageNetteCorruption(ImageNette):
    def __init__(self, root, split='train', transform=None, corruption=None, severity=3):
        super().__init__(root, split, transform)
        from imagecorruptions import corrupt
        self.corruption = corruption
        self.severity = severity
        self.corrupt_func = corrupt
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        numpy_img = np.asarray(sample)
        # TODO: add corruptions
        corrupted_img = self.corrupt_func(numpy_img, corruption_name=self.corruption, severity=self.severity)
        sample = Image.fromarray(corrupted_img, mode='RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

