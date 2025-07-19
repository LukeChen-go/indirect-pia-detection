import io
import json
import os
import sys
import time

import tqdm

# os.environ['HF_HOME'] = './'
# # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# access_token = ""
# os.environ['HF_TOKEN'] =access_token
import argparse
import random
import torch
# from utils import jload, Logger
from chatbot import HuggingfaceChatbot, GPTChatbot
from instruction_attack_defense_tools import *
from chatbot import ClassificationChatbot, DetectionChatbot

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict
class Logger(object):

    def __init__(self, log_path, on=True):
        self.log_path = log_path
        self.on = on

        if self.on:
            while os.path.isfile(self.log_path):
                self.log_path += '+'

    def log(self, string, newline=True, force=False):
        string = str(string)
        if self.on or force:
            with open(self.log_path, 'a') as logf:
                logf.write(string)
                if newline: logf.write('\n')

            sys.stdout.write(string)
            if newline: sys.stdout.write('\n')
            sys.stdout.flush()

def test(args):
    logger = Logger(args.log_path)
    logger.log(str(args))
    if "gpt" in args.model_path:
        model = GPTChatbot(args.model_path)
    else:
        model = HuggingfaceChatbot(args.model_path)
    filter_bot = None
    if args.filter_bot is not None and args.extract_bot is not None:
        if "prompt" not in args.filter_bot:
            filter_bot = ClassificationChatbot(args.filter_bot, args.extract_bot)
        else:
            filter_bot = DetectionChatbot(args.filter_bot, args.extract_bot)

    for s in args.side:
        for d in args.defense:
            for a in args.attack:
                time_start = time.time()

                logger.log(f"############# Attack Method {a}, Defense Method {d},Side {s}, Begin ###############")
                count = []
                injection_count = []
                data = jload(args.data_path)
                llm_input, d_items= form_llm_input(
                    data,
                    eval(a),
                    PROMPT_FORMAT,
                    defense=d,
                    injection_type=args.injection_type,
                    filter_bot=filter_bot,
                    side=s,
                    purify_method=args.purify_method,

                )
                for i in tqdm.tqdm(range(len(llm_input))):
                    message = llm_input[i]
                    injection_count.append(data[i]['injection'].lower() not in message["user_content"].lower())
                    # if args.injection_type == "acc":
                    #     if data[i]['injection'].lower() in message["user_content"].lower(): continue
                    answer = message["answer"]
                    response = model.respond(message)
                    print(answer)
                    print(response)

                    count.append(answer.lower() in response.lower())

                    # acc_count.append(answer.lower() in response.lower())
                    if args.debug:
                        logger.log(str(message))
                        logger.log("--------------------------------------")
                        logger.log(response)
                        logger.log("###########################")
                        logger.log(sum(count) / len(count))

                time_end = time.time()
                asr = sum(count) / len(count)
                rm_acc = sum(injection_count) / len(injection_count)
                # acc = sum(acc_count) / len(acc_count)

                time_cost = (time_end - time_start) / len(data)
                if args.debug:
                    print(time_end - time_start)
                logger.log(f"*********** ASR: {asr} ***********")
                logger.log(f"*********** COST: {time_cost} ***********")
                logger.log(f"*********** INJ ACC: {rm_acc} ***********")
                logger.log(f"############# Attack Method {a}, Defense Method {d}, Side {s}, Over ###############")

def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='gpt-4o')
    parser.add_argument('--data_path', type=str, default='../data/crafted_instruction_data_tri_injection_qa.json')
    parser.add_argument("--filter_bot", type=str, default=None)
    parser.add_argument("--extract_bot", type=str, default=None)
    parser.add_argument('--defense', type=str, default=['sandwich'], nargs='+')
    parser.add_argument('--attack', type=str, default=['naive'], nargs='+')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--log_path", type=str, default='logs/debug.txt')
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--defense_cross_prompt", action="store_true", default=False)
    parser.add_argument("--acc", action="store_true", default=False)
    parser.add_argument("--injection_type", type=str, default="adv")
    parser.add_argument("--side", type=str, default=['end'], nargs='+')
    parser.add_argument("--purify_method", type=str, default=None)
    parser.add_argument("--trigger", type=str, nargs='+', default=["<data>", "</data>"])

    args = parser.parse_args()
    set_seeds(args)
    test(args)



