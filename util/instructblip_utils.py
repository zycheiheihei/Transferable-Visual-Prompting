import sys
import torch
from models.instruct_blip.models.blip2_models.blip2_vicuna_instruct import Blip2VicunaInstruct
from models.instruct_blip.models.blip2_models.blip2_t5_instruct import Blip2T5Instruct
from models.instruct_blip.models import load_model_and_preprocess
from typing import Dict, List
from copy import deepcopy



SYS_PROMPT = ""

INFER_PROMPT = "{question}"

CLS_PROMPT = "{question}"

VQA_PROMPT = "{question} Answer this question in a single word."



def load_model(llm_name:str, device='cuda') -> Blip2VicunaInstruct | Blip2T5Instruct:
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
    return model


def model_encode(model:Blip2T5Instruct|Blip2VicunaInstruct, samples:Dict[str,torch.Tensor|List[str]|str|None]):
    image = samples["image"]
    with model.maybe_autocast():
        image_embeds = model.ln_vision(model.visual_encoder(image))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

    bs = image.size(0)
    
    if "prompt" in samples:
        samples["text_input"] = [samples["prompt"].format(question=q) for q in samples["text_input"]]

    query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
    if model.qformer_text_input:
        text_Qformer = model.tokenizer(
            samples["text_input"],
            padding='longest',
            truncation=True,
            max_length=model.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

        query_output = model.Qformer.bert(
            text_Qformer.input_ids,
            attention_mask=Qformer_atts,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
    else:
        query_output = model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
    return query_output.last_hidden_state[:,:query_tokens.size(1),:]


def model_generate(model:Blip2T5Instruct|Blip2VicunaInstruct, samples:Dict[str,torch.Tensor|List[str]|str|None], **generate_kwargs):
    do_sample = generate_kwargs.pop("do_sample", False)
    output_text = model.generate(samples, use_nucleus_sampling=do_sample, **generate_kwargs)
    return output_text


def model_loss(model:Blip2T5Instruct|Blip2VicunaInstruct, samples:Dict[str,torch.Tensor|List[str]|str|None]):
    if "prompt" in samples:
        samples["text_input"] = [samples["prompt"].format(question=q) for q in samples["text_input"]]
    return model(samples)
    
        
        
        
def model_loglikelihood_for_postfixes(model:Blip2T5Instruct|Blip2VicunaInstruct, samples:Dict[str,torch.Tensor|List[str]|str|None], postfixes:List[str], normalize=True):
    image = samples["image"]
    with model.maybe_autocast():
        image_embeds = model.ln_vision(model.visual_encoder(image))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

    bs = image.size(0)
    
    if "prompt" in samples:
        samples["text_input"] = [samples["prompt"].format(question=q) for q in samples["text_input"]]

    query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
    if model.qformer_text_input:
        text_Qformer = model.tokenizer(
            samples["text_input"],
            padding='longest',
            truncation=True,
            max_length=model.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

        query_output = model.Qformer.bert(
            text_Qformer.input_ids,
            attention_mask=Qformer_atts,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
    else:
        query_output = model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
    
    if isinstance(model, Blip2VicunaInstruct):
        inputs_llm = model.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
        
        
        
        to_regress_tokens = model.llm_tokenizer(
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
            to_regress_tokens.input_ids[i] += [model.llm_tokenizer.pad_token_id]*(max_length-input_token_length[i])
            to_regress_tokens.attention_mask[i] += [0]*(max_length-input_token_length[i])
        
        to_regress_tokens.input_ids = torch.LongTensor(to_regress_tokens.input_ids).to(model.device)
        to_regress_tokens.attention_mask = torch.LongTensor(to_regress_tokens.attention_mask).to(model.device)

        to_regress_embeds = model.llm_model.get_input_embeddings()(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_llm, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, to_regress_tokens.attention_mask], dim=1)
        
        for i in range(len(samples["text_input"])):
            inputs_embeds[i] = torch.roll(inputs_embeds[i], max_length-input_token_length[i], dims=0)
            attention_mask[i] = torch.roll(attention_mask[i], max_length-input_token_length[i], dims=0)
        
        
        # text_input_tokens = model.llm_tokenizer(
        #     samples['text_input'],
        #     return_tensors="pt",
        #     padding="longest",
        #     truncation=True,
        #     max_length=model.max_txt_len,
        # ).to(image.device)
        
        # inputs_embeds = model.llm_model.get_input_embeddings()(text_input_tokens['input_ids'])
        # inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        # attention_mask = torch.cat([atts_llm, text_input_tokens['attention_mask']], dim=1)
        
        with model.maybe_autocast():
            outputs = model.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
            
        precomputed_logits = outputs.logits
        precomputed_past_key_values = outputs.past_key_values
        
        postfix_loglikelihood = []
        
        for postfix in postfixes:
            postfix_tokens = model.llm_tokenizer(
                [postfix],
                add_special_tokens=False,
                return_tensors='pt'
            ).to(model.device)
            
            postfix_embeds = model.llm_model.get_input_embeddings()(postfix_tokens.input_ids).repeat(bs, 1, 1)
            postfix_attn_mask = postfix_tokens.attention_mask.repeat(bs, 1)
            # temp_input_embeds = torch.cat([inputs_embeds, postfix_embeds], dim=1)
            temp_attention_mask = torch.cat([attention_mask, postfix_attn_mask], dim=1)
            
            postfix_outputs = model.llm_model(
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
                    gen_probs, 2, postfix_tokens.input_ids[:, :, None].repeat(bs,1,1)
                ).squeeze(-1)
            
            if normalize:
                class_prob = torch.mean(gen_probs, dim=1)
            else:
                class_prob = torch.sum(gen_probs, dim=1)
                
            postfix_loglikelihood.append(class_prob)
        overall_probs = torch.vstack(postfix_loglikelihood).T  # shape (B, num_classes)
        return overall_probs
        