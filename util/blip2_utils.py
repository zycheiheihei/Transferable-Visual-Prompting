import torch
from models.instruct_blip.models.blip2_models.blip2_opt import Blip2OPT
from models.instruct_blip.models.blip2_models.blip2_t5 import Blip2T5
from models.instruct_blip.models import load_model_and_preprocess
from typing import Dict, List
from copy import deepcopy



SYS_PROMPT = ""

INFER_PROMPT = "Question: {question} Answer:"

CLS_PROMPT = "{question}"

VQA_PROMPT = "Question: {question} Short answer:"


def load_model(llm_name:str, device='cuda') -> Blip2OPT | Blip2T5:
    model_cls = None
    model_type = None
    if "opt" in llm_name.lower():
        model_cls = 'blip2_opt'
        if "2.7b" in llm_name.lower():
            model_type = "pretrain_opt2.7b"
        elif "6.7b" in llm_name.lower():
            model_type = "pretrain_opt6.7b"
    else:
        model_cls = 'blip2_t5'
        if "xxl" in llm_name.lower():
            model_type = "pretrain_flant5xxl"
        else:
            model_type = "pretrain_flant5xl"
    
    model, vis_processors, _ = load_model_and_preprocess(
        name=model_cls,
        model_type=model_type,
        is_eval=True,
        device=torch.device(device),
    )
    
    return model



def model_generate(model:Blip2T5|Blip2OPT, samples:Dict[str,torch.Tensor|List[str]|str|None], **generate_kwargs):
    do_sample = generate_kwargs.pop("do_sample", False)
    output_text = model.generate(samples, use_nucleus_sampling=do_sample, **generate_kwargs)
    return output_text


def model_loss(model:Blip2T5|Blip2OPT, samples:Dict[str,torch.Tensor|List[str]|str|None]):
    if "prompt" in samples:
        samples["text_input"] = [samples["prompt"].format(question=q) for q in samples["text_input"]]
    return model(samples)
    
        
        
        
def model_loglikelihood_for_postfixes(model:Blip2T5|Blip2OPT, samples:Dict[str,torch.Tensor|List[str]|str|None], postfixes:List[str], normalize=True):
    image = samples["image"]
    with model.maybe_autocast():
        image_embeds = model.ln_vision(model.visual_encoder(image))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

    bs = image.size(0)
    
    if "prompt" in samples:
        samples["text_input"] = [samples["prompt"].format(question=q) for q in samples["text_input"]]

    query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_output = model.Qformer.bert(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )
    
    if isinstance(model, Blip2OPT):
        inputs_opt = model.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)


        
        to_regress_tokens = model.opt_tokenizer(
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
            to_regress_tokens.input_ids[i] += [model.opt_tokenizer.pad_token_id]*(max_length-input_token_length[i])
            to_regress_tokens.attention_mask[i] += [0]*(max_length-input_token_length[i])
        
        to_regress_tokens.input_ids = torch.LongTensor(to_regress_tokens.input_ids).to(model.device)
        to_regress_tokens.attention_mask = torch.LongTensor(to_regress_tokens.attention_mask).to(model.device)

        to_regress_embeds = model.opt_model.get_input_embeddings()(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, to_regress_tokens.attention_mask], dim=1)
        
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
        
        # inputs_embeds = model.opt_model.get_input_embeddings()(text_input_tokens['input_ids'])
        # inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        # attention_mask = torch.cat([atts_llm, text_input_tokens['attention_mask']], dim=1)
        
        with model.maybe_autocast():
            outputs = model.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
            
        precomputed_logits = outputs.logits
        precomputed_past_key_values = outputs.past_key_values
        
        postfix_loglikelihood = []
        
        for postfix in postfixes:
            postfix_tokens = model.opt_tokenizer(
                [postfix],
                add_special_tokens=False,
                return_tensors='pt'
            ).to(model.device)
            
            postfix_embeds = model.opt_model.get_input_embeddings()(postfix_tokens.input_ids).repeat(bs, 1, 1)
            postfix_attn_mask = postfix_tokens.attention_mask.repeat(bs, 1)
            # temp_input_embeds = torch.cat([inputs_embeds, postfix_embeds], dim=1)
            temp_attention_mask = torch.cat([attention_mask, postfix_attn_mask], dim=1)
            
            postfix_outputs = model.opt_model(
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
    
    else:
        inputs_t5 = model.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
        
        bs = inputs_t5.shape[0]
        
        postfix_loglikelihood = []
        
        for postfix in postfixes:
            
            with model.maybe_autocast(dtype=torch.bfloat16):
                input_tokens = model.t5_tokenizer(
                    samples["text_input"],
                    padding="longest",
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(image.device)
                output_tokens = model.t5_tokenizer(
                    [postfix]*bs,
                    padding="longest",
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=False
                ).to(image.device)

                encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

                targets = output_tokens.input_ids.masked_fill(
                    output_tokens.input_ids == model.t5_tokenizer.pad_token_id, -100
                )

                inputs_embeds = model.t5_model.encoder.embed_tokens(input_tokens.input_ids)
                inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

                outputs = model.t5_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=encoder_atts,
                    decoder_attention_mask=output_tokens.attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            
            
            logits = outputs.logits
            logprobs = torch.log_softmax(logits, dim=-1)
            gen_probs = logprobs
            gen_probs = torch.gather(
                    gen_probs, 2, output_tokens.input_ids[:, :, None]
                ).squeeze(-1)
            
            if normalize:
                class_prob = torch.mean(gen_probs, dim=1)
            else:
                class_prob = torch.sum(gen_probs, dim=1)
                
            postfix_loglikelihood.append(class_prob)
        overall_probs = torch.vstack(postfix_loglikelihood).T  # shape (B, num_classes)
        return overall_probs
        