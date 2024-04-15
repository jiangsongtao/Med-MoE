import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init
from moellava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from moellava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
# os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
# os.environ.setdefault('MASTER_PORT', str(port))
# if rank == 0:
#     logger.info('init process group: '
#                 f'MASTER_ADDR={os.environ["MASTER_ADDR"]} '
#                 f'MASTER_PORT={os.environ["MASTER_PORT"]} '
#                 f'world_size={world_size}')

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"




detail_describe_instructions = [
    "Describe the following image in detail.",
    "Provide a detailed description of the given image.",
    "Give an elaborate explanation of the image you see.",
    "Share a comprehensive rundown of the presented image.",
    "Offer a thorough analysis of the image.",
    "Explain the various aspects of the image before you.",
    "Clarify the contents of the displayed image with great detail.",
    "Characterize the image using a well-detailed description.",
    "Break down the elements of the image in a detailed manner.",
    "Walk through the important details of the image.",
    "Portray the image with a rich, descriptive narrative.",
    "Narrate the contents of the image with precision.",
    "Analyze the image in a comprehensive and detailed manner.",
    "Illustrate the image through a descriptive explanation.",
    "Examine the image closely and share its details.",
    "Write an exhaustive depiction of the given image.",
]

concise_describe_instructions = [
    "Describe the following image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the following image.",
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo's key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo below.",
    "Write a terse but informative summary of the following picture.",
    "Create a compact narrative representing the image presented.",
]

prompt_pool = detail_describe_instructions + concise_describe_instructions

prompt_pool = [ "Describe the following image in detail."]


def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)





def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    if args.return_gating_logit is not None:
        from moellava.utils import get_gating_logit_by_hook
        print(model)
        fea_hooks = get_gating_logit_by_hook(model)
        all_gating_logits = {}
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    cnt = -1
    for i, line in enumerate(tqdm(questions)):
        cnt += 1
        idx = line["id"]
        # question = line['conversations'][0]
        # gt_ans = line["conversations"][1]

        try:
            question = line["conversations"][0] # ['value'].split('\n')[0]
            gt_ans = line["conversations"][1] # ['value']        
        except:
            question = line["conversatons"][0] # ['value'].split('\n')[0]
            gt_ans = line["conversatons"][1] # ['value']    

        qs = question['value']

        qs = qs.replace('<image>', '').strip()
        cur_prompt = qs
        #qs=f"""Question:{qs}\nAnswer:"""
        
        if 'image' in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
            images = image_processor['image'].preprocess(image, return_tensors='pt')['pixel_values'].to(model.device, dtype=torch.float16)
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None
        if(line['answer_type']=='close' or line['answer_type']=='CLOSED'):
            #qs = qs + '\n' + "Answer with yes or no"
            if args.single_pred_prompt:
                qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
                cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=2048,
                    use_cache=True if args.return_gating_logit is None else False,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=[stopping_criteria],
                )
            if args.return_gating_logit is not None:
                all_gating_logits[cnt] = dict(gating_logit=[i.fea for i in fea_hooks],
                                                images=images if images is None else images.detach().cpu(),
                                                input_ids=input_ids.detach().cpu(),
                                                output_ids=output_ids.detach().cpu())
                print(input_ids.shape, output_ids.shape, fea_hooks[0].fea.shape, images.shape if images is not None else [])
                print('The number of hooks is:', len(fea_hooks))
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        else:
            #qs = qs + '\n' + "Answer with a single word or short phrase"
            if args.single_pred_prompt:
                qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
                cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=2048,
                    use_cache=True if args.return_gating_logit is None else False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            if args.return_gating_logit is not None:
                all_gating_logits[cnt] = dict(gating_logit=[i.fea for i in fea_hooks],
                                                images=images if images is None else images.detach().cpu(),
                                                input_ids=input_ids.detach().cpu(),
                                                output_ids=output_ids.detach().cpu())
                print(input_ids.shape, output_ids.shape, fea_hooks[0].fea.shape, images.shape if images is not None else [])
                # assert fea_hooks[0].fea.shape[0] + 1 == output_ids.shape[1] + 575
                print('The number of hooks is:', len(fea_hooks))
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()
    if args.return_gating_logit is not None:
        torch.save(all_gating_logits, f'{args.return_gating_logit}.pt')
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="phi")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--return_gating_logit", type=str, default=None)
    args = parser.parse_args()

    eval_model(args)
