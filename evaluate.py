from config import create_parser
import os
from main import Exp, set_seed
import wandb
import torch
from src.utils.main_utils import print_log
import logging
import os.path as osp

if __name__ == '__main__':
    args = create_parser()
    config = args.__dict__
    print(config)
    
    os.environ["WANDB_DISABLED"] = "true"
    if not args.no_wandb:
        os.environ["WANDB_API_KEY"] = "ddb1831ecbd2bf95c3323502ae17df6e1df44ec0"
        wandb.init(project="test-project", entity="motifretro", config=config, name=args.ex_name)
        

    set_seed(111)
    exp = Exp(args)
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    path = osp.join(args.res_dir, args.ex_name)
    print(f"log path:{path}")
    logging.basicConfig(level=logging.INFO, filename=osp.join(path, 'log.log'),
                        filemode='a', format='%(asctime)s - %(message)s')
    print_log(f"log path:{path}")
    
    
    all_metrics = {}
    for ckpt in os.listdir(f"/gaozhangyang/experiments/MotifRetro/results/{args.ex_name}/checkpoints"):
        if ckpt[-4:] != ".pth":
            continue
        exp.method.model.load_state_dict(torch.load(f"/gaozhangyang/experiments/MotifRetro/results/{args.ex_name}/checkpoints/{ckpt}"))
        test_metric = exp.test()
        all_metrics[ckpt] = test_metric
        print_log(all_metrics)
    
    print_log(all_metrics)