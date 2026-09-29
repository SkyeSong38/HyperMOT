import argparse
import yaml
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of MID')
    parser.add_argument('--config', default='./configs/mot.yaml')
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--resume', action='store_true', default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    from easydict import EasyDict
    with open(args.config) as f:
       config = yaml.safe_load(f)

    for k, v in vars(args).items():
       if v is not None:
           config[k] = v
    config["exp_name"] = args.config.split("/")[-1].split(".")[0]
    if not config.get("dataset"):
        config["dataset"] = {
            "mot": "MOT17", "mot17": "MOT17",
            "dancetrack": "DanceTrack", "sportsmot": "SportsMOT",
        }.get(config["exp_name"].removesuffix("_test"), config["exp_name"])

    config = EasyDict(config)
    from hyperssm import HyperSSM
    agent = HyperSSM(config)

    if config["eval_mode"]:
        agent.eval()
    else:
        agent.train()


if __name__ == '__main__':
    main()
