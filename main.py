import os
# os.environ['CUDA_VISIBLE_DEVICES'] = str(7)
import nni
import wandb
import logging
import pickle
import json
import torch
import os.path as osp
from config import create_parser
from src.utils.load_data import get_dataset
from src.utils.main_utils import print_log, check_dir

import warnings
import torch.backends.cudnn as cudnn
warnings.filterwarnings('ignore')


import random 
import numpy as np

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


from src.utils.recorder import Recorder

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" python -m torch.distributed.launch --nproc_per_node 6 main.py
# CUDA_VISIBLE_DEVICES="1" python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234 main.py

class Exp:
    def __init__(self, args, show_params=True):
        self.args = args
        self.config = args.__dict__
        self.device = self._acquire_device()
        self.total_step = 0
        self._preparation()
        if show_params:
            print_log(self.args)
    
    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:0')
            print('Use GPU:',device)
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _preparation(self):
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the method
        self._build_method()

    def _build_method(self):
        steps_per_epoch = len(self.train_loader)
            
        if self.args.method == "MotifRetro_GNN":
            # from methods.MotifRetro2_GNN import MotifRetro
            # self.method = MotifRetro(self.args, self.device, steps_per_epoch, self.train_loader.dataset.feat_vocab, self.train_loader.dataset.action_vocab)
            
            from methods.MotifRetro_GNN import MotifRetro
            self.method = MotifRetro(self.args, self.device, steps_per_epoch, self.train_loader.dataset.feat_vocab, self.train_loader.dataset.action_vocab)
            
            # from methods.MotifRetro5_GNN import MotifRetro
            # self.method = MotifRetro(self.args, self.device, steps_per_epoch, self.train_loader.dataset.feat_vocab, self.train_loader.dataset.action_vocab)

    def _get_data(self):
        self.train_loader = get_dataset(args=self.args, data_name=self.args.dataset_key, 
                                        featurizer_key=self.args.featurizer_key,
                                        data_path=self.args.data_path,
                                        # keep_action = self.args.keep_action,
                                        use_reaction_type=self.args.reaction_type_given,
                                        num_workers = self.args.num_workers,
                                        batch_size = self.args.batch_size,
                                        vocab_path = self.args.vocab_path,
                                        mode="train")
        
        self.valid_loader = get_dataset(args=self.args, data_name=self.args.dataset_key, 
                                        featurizer_key=self.args.featurizer_key,
                                        data_path=self.args.data_path,
                                        # keep_action = self.args.keep_action,
                                        use_reaction_type=self.args.reaction_type_given,
                                        num_workers = self.args.num_workers,
                                        vocab_path = self.args.vocab_path,
                                        batch_size = self.args.batch_size,
                                        mode="valid")
        
        self.test_loader = get_dataset(args=self.args, data_name=self.args.dataset_key, 
                                        featurizer_key=self.args.featurizer_key,
                                        data_path=self.args.data_path,
                                        # keep_action = self.args.keep_action,
                                        use_reaction_type=self.args.reaction_type_given,
                                        num_workers = self.args.num_workers,
                                        batch_size = self.args.batch_size,
                                        vocab_path = self.args.vocab_path,
                                        mode="test")

    def _save(self, name=''):
        torch.save(self.method.model.state_dict(), osp.join(self.checkpoints_path, name + '.pth'))
        fw = open(osp.join(self.checkpoints_path, name + '.pkl'), 'wb')
        state = self.method.scheduler.state_dict()
        pickle.dump(state, fw)

    def _load(self, epoch):
        self.method.model.load_state_dict(torch.load(osp.join(self.checkpoints_path, str(epoch) + '.pth')))
        fw = open(osp.join(self.checkpoints_path, str(epoch) + '.pkl'), 'rb')
        state = pickle.load(fw)
        self.method.scheduler.load_state_dict(state)

    def train(self):
        recorder = Recorder(self.args.patience, verbose=True)
        for epoch in range(self.args.epoch):
            self.method.epoch = epoch
            train_metric = self.method.train_one_epoch(self.train_loader)

            new_train_metric = {}
            for k, v in train_metric.items():
                new_train_metric['train/' + k] = v
            if not args.no_wandb:
                wandb.log(new_train_metric)
            print_log(new_train_metric)

            if epoch>self.args.epoch-3:
                self._save("epoch_{}".format(epoch))
            if epoch % self.args.log_step == 0:
                valid_metric = self.valid()
                
                print_log('Epoch: {}, Steps: {} | Train Loss: {:.4f}  Valid Loss: {:.4f}\n'.format(epoch + 1, len(self.train_loader), train_metric['loss'], valid_metric['loss']))
                recorder(-valid_metric['acc'], self.method.model, self.path)
            
            if self.method.break_flag:
                break
            
        best_model_path = osp.join(self.path, 'checkpoint.pth')
        self.method.model.load_state_dict(torch.load(best_model_path))

    def valid(self):
        epoch_metric = self.method.valid_one_epoch(self.valid_loader)
        print_log('step_acc: {:.4f}  acc: {:.4f}  loss: {:.4f}'.format(epoch_metric['step_acc'], epoch_metric['acc'], epoch_metric['loss']))
        epoch_metric['default'] = epoch_metric['step_acc']
        new_epoch_metric = {}
        for k, v in epoch_metric.items():
            new_epoch_metric['valid/' + k] = v
        if not args.no_wandb:
            wandb.log(new_epoch_metric)
        print_log(new_epoch_metric)
        return epoch_metric

    def test(self):
        epoch_metric = self.method.test_one_epoch(self.test_loader)
        if not args.no_wandb:
            wandb.log(epoch_metric)
        print_log(epoch_metric)

        return epoch_metric


if __name__ == '__main__':
    # debug: CUDA_VISIBLE_DEVICES="0" python -m debugpy --listen 5671 --wait-for-client -m torch.distributed.launch --nproc_per_node 1 main.py
    # torch.distributed.init_process_group(backend='nccl')
    args = create_parser()
    config = args.__dict__

    tuner_params = nni.get_next_parameter()
    config.update(tuner_params)
    print(config)
    
    os.environ["WANDB_DISABLED"] = "true"
    if not args.no_wandb:
        os.environ["WANDB_API_KEY"] = "ddb1831ecbd2bf95c3323502ae17df6e1df44ec0"
        wandb.init(project="test-project", entity="motifretro", config=config, name=args.ex_name)
        

    set_seed(111)
    exp = Exp(args)
    # args.only_test = True

    # exp.method.model.load_state_dict(torch.load("/gaozhangyang/experiments/MotifRetro/results/search2023-03-22 21:57:17.241017/checkpoint.pth"))
    

    if not args.only_test:
    #     print('>>>>>>>>>>>>>>>>>>>>>>>>>> training <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        if not args.only_valid:
            exp.train()
        else:
            exp.valid()
        # exp.train()
        # exp.valid()
    # print('>>>>>>>>>>>>>>>>>>>>>>>>> testing  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    else:
        test_metric = exp.test()
    print("finished")
