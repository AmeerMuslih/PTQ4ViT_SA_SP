#packages that need to be install: torch, torchVision, timm, 
from torch.nn.modules import module
from ameer_example.test_vit import *
from quant_layers.conv import MinMaxQuantConv2d
from quant_layers.linear import MinMaxQuantLinear, PTQSLQuantLinear
from quant_layers.matmul import MinMaxQuantMatMul, PTQSLQuantMatMul
import matplotlib.pyplot as plt
from utils.net_wrap import wrap_certain_modules_in_net
from tqdm import tqdm
import torch.nn.functional as F
import pickle as pkl
from itertools import product
import types
from utils.quant_calib import HessianQuantCalibrator, QuantCalibrator
from utils.models import get_net
from utils.integer import get_model_weight, set_model_weight
import time
from Matmul_SA import csvFilesMaker
import sys
import os

HF_HUB_DISABLE_SYMLINKS_WARNING=1

class cfg_modifier():
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self,name,value)

    def __call__(self, cfg):
        # bit setting
        cfg.bit = self.bit_setting
        cfg.w_bit = {name: self.bit_setting[0] for name in cfg.conv_fc_name_list}
        cfg.a_bit = {name: self.bit_setting[1] for name in cfg.conv_fc_name_list}
        cfg.A_bit = {name: self.bit_setting[1] for name in cfg.matmul_name_list}
        cfg.B_bit = {name: self.bit_setting[1] for name in cfg.matmul_name_list}

        # conv2d configs
        cfg.ptqsl_conv2d_kwargs["n_V"] = self.linear_ptq_setting[0]
        cfg.ptqsl_conv2d_kwargs["n_H"] = self.linear_ptq_setting[1]
        cfg.ptqsl_conv2d_kwargs["metric"] = self.metric
        cfg.ptqsl_conv2d_kwargs["init_layerwise"] = False

        # linear configs
        cfg.ptqsl_linear_kwargs["n_V"] = self.linear_ptq_setting[0]
        cfg.ptqsl_linear_kwargs["n_H"] = self.linear_ptq_setting[1]
        cfg.ptqsl_linear_kwargs["n_a"] = self.linear_ptq_setting[2]
        cfg.ptqsl_linear_kwargs["metric"] = self.metric
        cfg.ptqsl_linear_kwargs["init_layerwise"] = False

        # matmul configs
        cfg.ptqsl_matmul_kwargs["metric"] = self.metric
        cfg.ptqsl_matmul_kwargs["init_layerwise"] = False

        return cfg
    
    
if __name__=='__main__':
    name = "vit_base_patch16_224"
    config_name = "PTQ4ViT"
    calib_size = 32
    cfg = cfg_modifier(linear_ptq_setting=(1,1,1), metric="hessian", bit_setting=(8,8))
    #os.chdir('.') #/raid/ori.sch/PTQ4ViT
    #RUN_ITER = 300
    IMG_PATH = '/datasets/ImageNet'
    start_idx = int(sys.argv[1])
    assert start_idx >= 0 and start_idx < 1000
    print(f"Classifying image: {start_idx}")
    os.environ['start_idx'] = str(start_idx)
    quant_cfg = init_config(config_name)
    quant_cfg = cfg(quant_cfg)

    net = get_net(name)

    wrapped_modules=net_wrap.wrap_modules_in_net(net,quant_cfg)

    g=datasets.ViTImageNetLoaderGenerator(IMG_PATH,'imagenet',32,32,2, kwargs={"model":net}) 
    test_loader=g.test_loader()
    
    weights_path = f"./weights/{name}.pth"
    if os.path.exists(weights_path):
        weights = torch.load(weights_path)
        set_model_weight(wrapped_modules, weights)
        print("weights loaded")
    else:
        calib_loader=g.calib_loader(num=calib_size,seed=88)
        calib_start_time = time.time()
        quant_calibrator = HessianQuantCalibrator(net,wrapped_modules,calib_loader,sequential=False,batch_size=4) # 16 is too big for ViT-L-16
        quant_calibrator.batching_quant_calib()
        calib_end_time = time.time()
        print(f"calibration time: {(calib_end_time-calib_start_time)/60}min")
        weights = get_model_weight(wrapped_modules)
        torch.save(weights, weights_path)
    
    # add timing
    acc_start_time = time.time()
    acc = test_classification(net,test_loader, description=quant_cfg.ptqsl_linear_kwargs["metric"], start_idx=start_idx, max_iteration=1) #max_iteration=RUN_ITER,
    acc_end_time = time.time()
    print(f"original accuracy: {acc}")
    print(f"original run time: {(acc_end_time-acc_start_time)/60}min")
    #csvFilesMaker()
    
    # for j in range(6):
    #     model = copy.deepcopy(net)
    #     for i in range(12):
    #         if j == 0:
    #             model.blocks[i].attn.matmul1 = NbSmtMatMul(net.blocks[i].attn.matmul1,i,chunks=32)
    #         elif j == 1:
    #             model.blocks[i].attn.matmul2 = NbSmtMatMulScores(net.blocks[i].attn.matmul2,i,chunks=32)
    #         elif j == 2:
    #             model.blocks[i].attn.qkv = NbSmtLinear(net.blocks[i].attn.qkv,layer_num=i,chunks=72)
    #         elif j == 3:
    #             model.blocks[i].attn.proj = NbSmtLinear(net.blocks[i].attn.proj,layer_num=i,chunks=24)
    #         elif j == 4:
    #             model.blocks[i].mlp.fc1 = NbSmtLinear(net.blocks[i].mlp.fc1,layer_num=i,chunks=64)
    #         else:
    #             model.blocks[i].mlp.fc2 = NbSmtGeluLinear(net.blocks[i].mlp.fc2,layer_num=i,chunks=128)

    #     g=datasets.ViTImageNetLoaderGenerator(IMG_PATH,'imagenet',32,32,16, kwargs={"model":net})
    #     test_loader=g.test_loader()
    #     acc_start_time = time.time()
    #     acc = test_classification(model,test_loader, description=quant_cfg.ptqsl_linear_kwargs["metric"], max_iteration=RUN_ITER)
    #     acc_end_time = time.time()
    #     print(f"new model {j} accuracy: {acc}")
    #     print(f"new model {j} run time: {(acc_end_time-acc_start_time)/60}min")

    #     del model