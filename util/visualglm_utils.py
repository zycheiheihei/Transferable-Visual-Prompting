import os
import torch
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List


SYS_PROMPT = '<img></img> Q:'
INFER_PROMPT = '{question} \nA:'
CLS_PROMPT = "<img></img>"


visualglm_tokenizer = None


def load_model(llm, device="cuda"):
    global visualglm_tokenizer
    visualglm_tokenizer = AutoTokenizer.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True).half().to(device=device)
    return model


def model_generate(model: AutoModel, samples:Dict[str,torch.Tensor|List[str]|str|None], **generate_kwargs):
    images = samples["image"].to(device=model.device, dtype=model.dtype)
    
    max_tokens = generate_kwargs.pop("max_length", 128)
    
    batch_size = images.shape[0]
    
    prompt = ' '.join([SYS_PROMPT, samples['prompt'][0]])
    input0 = visualglm_tokenizer.encode(prompt[:5], add_special_tokens=False)
    input1 = [visualglm_tokenizer.unk_token_id] * model.image_length
    input2 = visualglm_tokenizer.encode(prompt[5:], add_special_tokens=False)
    inputs = sum([input0, input1, input2], [])
    inputs = {
        "input_ids": torch.tensor([visualglm_tokenizer.build_inputs_with_special_tokens(inputs)], dtype=torch.long).to(
            model.device).repeat((batch_size,1)),
        "pre_image_length": len(input0),
        "images": images}
    
    generate_kwargs["do_sample"] = False
    generate_kwargs["max_new_tokens"] = max_tokens
    
    outputs = model.generate(**inputs, **generate_kwargs)
    outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
    response = visualglm_tokenizer.decode(outputs)
    return [response]

    
def model_loss(model: AutoModel, samples:Dict[str,torch.Tensor|List[str]|str|None]):
    images = samples["image"].to(device=model.device, dtype=model.dtype)
    batch_size = images.shape[0]
    assert batch_size==1
    
    prompt = ' '.join([samples["prompt"], samples['text_input'][0]])
    input0 = visualglm_tokenizer.encode(prompt[:5], add_special_tokens=False)
    input1 = [visualglm_tokenizer.unk_token_id] * model.image_length
    input2 = visualglm_tokenizer.encode(prompt[5:], add_special_tokens=False)
    target = visualglm_tokenizer.encode(samples["text_output"][0], add_special_tokens=False, return_tensors='pt').to(model.device).repeat(batch_size,1)
    inputs = sum([input0, input1, input2], [])
    
    inputs = {
        "input_ids": torch.cat([torch.tensor([visualglm_tokenizer.build_inputs_with_special_tokens(inputs)], dtype=torch.long).to(
            model.device).repeat((batch_size,1)), target], dim=1),
        "pre_image_length": len(input0),
        "images": images}
    
    labels = inputs["input_ids"].clone()
    labels[:-target.shape[1]] = -100
    
    inputs["labels"] = labels
    inputs["return_dict"] = True    
    output = model(**inputs).loss
    
    return output
    



def model_loglikelihood_for_postfixes(model:AutoModel, samples:Dict[str,torch.Tensor|List[str]|str|None], postfixes:List[str], normalize=True):
    image = samples["image"].to(device=model.device, dtype=model.dtype)
    batch_size = image.shape[0]
    postfix_loglikelihood = []
    
    for postfix in postfixes:
        prompt = ' '.join([samples['prompt'], samples['text_input'][0]])
        input0 = visualglm_tokenizer.encode(prompt[:5], add_special_tokens=False)
        input1 = [visualglm_tokenizer.unk_token_id] * model.image_length
        input2 = visualglm_tokenizer.encode(prompt[5:], add_special_tokens=False)
        inputs = sum([input0, input1, input2], [])
        postfix_tokens = visualglm_tokenizer.encode(postfix, add_special_tokens=False, return_tensors='pt').to(model.device).repeat(batch_size, 1)
        inputs = {
            "input_ids": torch.cat([torch.tensor([visualglm_tokenizer.build_inputs_with_special_tokens(inputs)], dtype=torch.long).to(
                model.device).repeat((batch_size,1)), postfix_tokens], dim=1),
            "pre_image_length": len(input0),
            "return_dict": True,
            "images": image}
        
        outputs = model(**inputs)
        postfix_logits = outputs.logits
        
        
        logprobs = torch.log_softmax(postfix_logits, dim=-1)
        gen_probs = logprobs[
                :, -postfix_tokens.shape[1] - 1 : -1, :
            ]  # (B, num_tokens_in_classname, vocab_len)
        gen_probs = torch.gather(
                gen_probs, 2, postfix_tokens[:, :, None]
            ).squeeze(-1)
        
        if normalize:
            class_prob = torch.mean(gen_probs, dim=1)
        else:
            class_prob = torch.sum(gen_probs, dim=1)
            
        postfix_loglikelihood.append(class_prob)
        
    overall_probs = torch.vstack(postfix_loglikelihood).T  # shape (B, num_classes)
    return overall_probs
        
        