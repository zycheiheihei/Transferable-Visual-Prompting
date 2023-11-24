import sys
import torch
# sys.path.append("../MiniGPT-4")
from models.minigpt4.models.mini_gpt4 import MiniGPT4
# sys.path.pop(-1)
from typing import Dict, List
from copy import deepcopy
from transformers import StoppingCriteriaList, StoppingCriteria


SYS_PROMPT = (
            'Give the following image: <Img>ImageContent</Img>.'
            'You will be able to see the image once I provide it to you.'
            'Please answer my questions. ###Human: <Img><ImageHere></Img>')

INFER_PROMPT = '{question} ###Assistant:'

CLS_PROMPT = "<Img><ImageHere></Img>"

VQA_PROMPT = "{question} Answer this question in a single word. ###Assistant:"


def load_model(llm_name:str, device='cuda') -> MiniGPT4:
        
    llama_model = "../models/mini-GPT4/vicuna-7b-v0" if '7b' in llm_name.lower() \
                else "../models/mini-GPT4/vicuna-13b-v0"
    model = MiniGPT4(llama_model=llama_model, max_txt_len=256)
    ckpt_path = "../models/mini-GPT4/prerained_minigpt4_7b.pth" if '7b' in llm_name.lower() \
                else "../models/mini-GPT4/pretrained_minigp4.pth"
    print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    msg = model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    model.to(device=device)
        
    return model


def diffable_vis_processor(image:torch.Tensor, normalize=True) -> torch.Tensor:
    mean = torch.Tensor((0.48145466, 0.4578275, 0.40821073)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(image.device) # RGB
    std = torch.Tensor((0.26862954, 0.26130258, 0.27577711)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(image.device)
    
    if image.dim()==3:
        image = image.unsqueeze(0)
        
    assert image.shape[1] == 3 # Shape (1,C,H,W)
    
    resized_image = torch.nn.functional.interpolate(image, (224,224), mode='bicubic')
    if normalize:
        normalized_image = (resized_image - mean) / std
        return normalized_image
    else:
        return resized_image, mean, std
        
        
class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False
    
    
def model_encoder(model:MiniGPT4, samples:Dict[str,torch.Tensor|List[str]|str|None]):
    return model.encode_img(samples["image"])[0]


    
def model_generate(model:MiniGPT4, samples:Dict[str,torch.Tensor|List[str]|str|None], **generate_kwargs):
    image = samples["image"]
    img_embeds, atts_img = model.encode_img(image)
    # print(img_embeds)
    img_embeds, atts_img = model.prompt_wrap(img_embeds, atts_img, SYS_PROMPT)
    
    batch_size = img_embeds.shape[0]
    bos = torch.ones([batch_size, 1],
                    dtype=torch.int64).to(model.device) * model.llama_tokenizer.bos_token_id
    bos_embeds = model.llama_model.model.embed_tokens(bos)
    inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
    atts_bos = atts_img[:, :1]
    atts_img = torch.cat([atts_bos, atts_img], dim=1)
    
    stop_words_ids = [
            torch.tensor([835]).to(model.device),
            torch.tensor([2277, 29937]).to(model.device),
        ]
    max_tokens = generate_kwargs.get("max_length", 128)
    num_beams = generate_kwargs.get("num_beams", 5)
    do_sample = generate_kwargs.get("do_sample", False)
    min_length = generate_kwargs.get("min_length", 1)
    top_p = generate_kwargs.get("top_p", 0.9)
    repetition_penalty = generate_kwargs.get("repetition_penalty", 1.0)
    length_penalty = generate_kwargs.get("length_penalty", -1.0)
    temperature = generate_kwargs.get("temperature", 1.0)
    num_return_sequences = generate_kwargs.get("num_return_sequences", 1)
    stopping_criteria = generate_kwargs.get("stopping_criteria",StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)]))

    to_regress_tokens = model.llama_tokenizer(
        samples["prompt"],
        # return_tensors='pt',
        add_special_tokens=False,
    )
    
    # # padding and locate
    max_length = 0
    input_token_length = []
    for i in range(len(samples["prompt"])):
        input_token_length.append(len(to_regress_tokens.input_ids[i]))
        max_length = max(input_token_length[-1], max_length)
    
    for i in range(len(samples["prompt"])):
        to_regress_tokens.input_ids[i] += [model.llama_tokenizer.pad_token_id]*(max_length-input_token_length[i])
        to_regress_tokens.attention_mask[i] += [0]*(max_length-input_token_length[i])
    
    to_regress_tokens.input_ids = torch.LongTensor(to_regress_tokens.input_ids).to(model.device)
    to_regress_tokens.attention_mask = torch.LongTensor(to_regress_tokens.attention_mask).to(model.device)

    to_regress_embeds = model.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
    inputs_embeds = torch.cat([inputs_embeds, to_regress_embeds], dim=1)
    attention_mask = torch.cat([atts_img, to_regress_tokens.attention_mask], dim=1)
    
    for i in range(len(samples["prompt"])):
        inputs_embeds[i] = torch.roll(inputs_embeds[i], max_length-input_token_length[i], dims=0)
        attention_mask[i] = torch.roll(attention_mask[i], max_length-input_token_length[i], dims=0)
    
    # generate output
    outputs = model.llama_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_length=max_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
        min_length=min_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        temperature=temperature,
        stopping_criteria=stopping_criteria,
        num_return_sequences=num_return_sequences)
    
    
    output_txt = []
    for b in range(batch_size):
        output_token = outputs[b]
        if output_token[0] == 0:
            output_token = output_token[1:]
        if output_token[0] == 1:
            output_token = output_token[1:]
        output_text = model.llama_tokenizer.decode(output_token,
                                        add_special_tokens=False)
        output_txt.append(output_text)
    
    return output_txt
    
    


def model_loss(model:MiniGPT4, samples:Dict[str,torch.Tensor|List[str]|str|None]):
    image = samples["image"]
    img_embeds, atts_img = model.encode_img(image)
    img_embeds, atts_img = model.prompt_wrap(img_embeds, atts_img, samples["prompt"])
    
    assert len(samples["text_input"]) == len(samples["text_output"])
    
    model.llama_tokenizer.padding_side = "right"
    
    
    input_to_regress_tokens = model.llama_tokenizer(
        samples["text_input"],
        padding=False,
        add_special_tokens=False
    )
    
    output_to_regress_tokens = model.llama_tokenizer(
        [t+model.end_sym for t in samples["text_output"]],
        padding=False,
        add_special_tokens=False
    )
    
    token_type_ids = []
    max_length = 0
    to_regress_tokens = deepcopy(input_to_regress_tokens)
    to_regress_tokens.attention_mask = []
    to_regress_tokens.input_ids = []
    
    
    for i in range(len(samples["text_input"])):
        token_type_ids.append([0]*len(input_to_regress_tokens.input_ids[i])+[1]*len(output_to_regress_tokens.input_ids[i]))
        to_regress_tokens.attention_mask.append(input_to_regress_tokens.attention_mask[i]+output_to_regress_tokens.attention_mask[i])
        to_regress_tokens.input_ids.append(input_to_regress_tokens.input_ids[i]+output_to_regress_tokens.input_ids[i])
        max_length = max(max_length, len(token_type_ids[-1]))
    
    # Padding the to_regress_tokens
    for i in range(len(samples["text_input"])):
        to_regress_tokens.attention_mask[i] += [0]*(max_length-len(to_regress_tokens.attention_mask[i]))
        to_regress_tokens.input_ids[i] += [model.llama_tokenizer.pad_token_id]*(max_length-len(to_regress_tokens.input_ids[i]))
        token_type_ids[i] += [0]*(max_length-len(token_type_ids[i]))
        
    to_regress_tokens.input_ids = torch.LongTensor(to_regress_tokens.input_ids).to(model.device)
    to_regress_tokens.attention_mask = torch.LongTensor(to_regress_tokens.attention_mask).to(model.device)
    to_regress_tokens["input_ids"] = to_regress_tokens.input_ids
    to_regress_tokens["attention_mask"] = to_regress_tokens.attention_mask
    token_type_ids = torch.Tensor(token_type_ids).to(model.device)

    targets = to_regress_tokens.input_ids.masked_fill(
        token_type_ids != 1, -100
    )

    empty_targets = (
        torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
    )
    targets = torch.cat([empty_targets, targets], dim=1)

    batch_size = img_embeds.shape[0]
    bos = torch.ones([batch_size, 1],
                    dtype=to_regress_tokens.input_ids.dtype,
                    device=to_regress_tokens.input_ids.device) * model.llama_tokenizer.bos_token_id
    bos_embeds = model.llama_model.model.embed_tokens(bos)
    atts_bos = atts_img[:, :1]

    to_regress_embeds = model.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
    inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
    attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)
    
    with model.maybe_autocast():
        outputs = model.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
    loss = outputs.loss

    return {"loss": loss}






def model_loglikelihood_for_postfixes(model:MiniGPT4, samples:Dict[str,torch.Tensor|List[str]|str|None], postfixes:List[str], normalize=True):
    image = samples["image"]
    img_embeds, atts_img = model.encode_img(image)
    img_embeds, atts_img = model.prompt_wrap(img_embeds, atts_img, samples["prompt"])
    
    to_regress_tokens = model.llama_tokenizer(
        samples["text_input"],
        # return_tensors='pt',
        add_special_tokens=False,
    )
    
    # # padding and locate
    max_length = 0
    input_token_length = []
    for i in range(len(samples["text_input"])):
        input_token_length.append(len(to_regress_tokens.input_ids[i]))
        max_length = max(input_token_length[-1], max_length)
    
    for i in range(len(samples["text_input"])):
        to_regress_tokens.input_ids[i] += [model.llama_tokenizer.pad_token_id]*(max_length-input_token_length[i])
        to_regress_tokens.attention_mask[i] += [0]*(max_length-input_token_length[i])
    
    to_regress_tokens.input_ids = torch.LongTensor(to_regress_tokens.input_ids).to(model.device)
    to_regress_tokens.attention_mask = torch.LongTensor(to_regress_tokens.attention_mask).to(model.device)
    
    batch_size = img_embeds.shape[0]
    bos = torch.ones([batch_size, 1],
                        dtype=to_regress_tokens.input_ids.dtype,
                        device=to_regress_tokens.input_ids.device) * model.llama_tokenizer.bos_token_id
    bos_embeds = model.llama_model.model.embed_tokens(bos)
    atts_bos = atts_img[:, :1]

    to_regress_embeds = model.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
    inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
    attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)
    
    for i in range(len(samples["text_input"])):
        inputs_embeds[i] = torch.roll(inputs_embeds[i], max_length-input_token_length[i], dims=0)
        attention_mask[i] = torch.roll(attention_mask[i], max_length-input_token_length[i], dims=0)
    
    with model.maybe_autocast():
        outputs = model.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
    precomputed_logits = outputs.logits
    precomputed_past_key_values = outputs.past_key_values
    
    postfix_loglikelihood = []
    
    for postfix in postfixes:
        postfix_tokens = model.llama_tokenizer(
            [postfix],
            add_special_tokens=False,
            return_tensors='pt'
        ).to(model.device)
        
        postfix_embeds = model.llama_model.model.embed_tokens(postfix_tokens.input_ids).repeat(batch_size, 1, 1)
        postfix_attn_mask = postfix_tokens.attention_mask.repeat(batch_size, 1)
        # temp_input_embeds = torch.cat([inputs_embeds, postfix_embeds], dim=1)
        temp_attention_mask = torch.cat([attention_mask, postfix_attn_mask], dim=1)
        
        postfix_outputs = model.llama_model(
            inputs_embeds = postfix_embeds,
            attention_mask = temp_attention_mask,
            return_dict = True,
            past_key_values = precomputed_past_key_values
        )
        
        postfix_logits = postfix_outputs.logits
        
        logits = torch.cat([precomputed_logits, postfix_logits], dim=1)
        logprobs = torch.log_softmax(logits, dim=-1)
        gen_probs = logprobs[
                :, -postfix_tokens.input_ids.shape[1] - 1 : -1, :
            ]  # (B, num_tokens_in_classname, vocab_len)
        gen_probs = torch.gather(
                gen_probs, 2, postfix_tokens.input_ids[:, :, None].repeat(batch_size,1,1)
            ).squeeze(-1)
        
        if normalize:
            class_prob = torch.mean(gen_probs, dim=1)
        else:
            class_prob = torch.sum(gen_probs, dim=1)
            
        postfix_loglikelihood.append(class_prob)
    overall_probs = torch.vstack(postfix_loglikelihood).T  # shape (B, num_classes)
    return overall_probs
        
        
    
        
    



