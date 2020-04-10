from easydict import EasyDict as edict
from pathlib import Path
import torch
from torchvision import transforms as trans

def get_config(training = True):
    conf = edict()
    conf.data_path = Path('data')
    conf.work_path = Path('model/')
    conf.save_path = conf.work_path
    conf.password = 'abcd'
    conf.input_size = [112, 112]
    conf.embedding_size = 512
    conf.use_mobilfacenet = True
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    # conf.data_mode = 'emore'
    # conf.vgg_folder = conf.data_path/'faces_vgg_112x112'
    # conf.ms1m_folder = conf.data_path/'faces_ms1m_112x112'
    # conf.emore_folder = conf.data_path/'faces_emore'
    # conf.batch_size = 100 # irse net depth 50
#   conf.batch_size = 200 # mobilefacenet
#--------------------Training Config ------------------------    

#--------------------Inference Config ------------------------
    conf.facebank_path = conf.data_path/'facebank'
    conf.threshold = 0.9
    conf.face_limit = 2#10
    #when inference, at maximum detect 10 faces in one image, my laptop is slow
    conf.min_face_size = 112 #30
    # the larger this value, the faster deduction, comes with tradeoff in small faces
    return conf