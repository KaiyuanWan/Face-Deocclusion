import os.path
import sys
sys.path.append('./models')
import generator
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision.datasets.folder import is_image_file
from PIL import Image
import numpy as np
import time
import cv2
from ops import get_one_hot, hard_prob

gpu_id = 3
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
cudnn.benchmark = True

class Option:
    def __init__(self):
        self.name = 'DeocNet'
        self.image_channel = 3
        self.image_size = 128
        self.detect_parsing_arch = 'detect_parsing'
        self.reconstruct_arch = 'reconstruct'
        self.test_dir = ''
        self.save_path = ''
        self.paletteID = [0, 0, 0, 128, 0, 0, 128, 0, 128, 128, 128, 0, 0, 128, 128, 128, 192, 0, 0, 64, 128]
        self.deoc_norm_mean = [93.5940, 104.7624, 129.1863]

opt = Option()
opt.test_dir = './test_image' ##your test data path
root_save_path = './trained_models'  ##trained models path
opt.save_path = './result'           ##test result save path
os.makedirs(opt.save_path, exist_ok=True)
opt.detect_parsing_model = os.path.join(root_save_path, 'detect_parsing.pth.tar')
opt.reconstruct_model = os.path.join(root_save_path, 'reconstruction.pth.tar')


# create detection and parsing model
print("=> using pre-trained detect and parsing model '{}'".format(opt.detect_parsing_model))
detect_parsing_model = generator.__dict__[opt.detect_parsing_arch]()
trained_deoc_model = torch.load(opt.detect_parsing_model)['state_dict']
detect_parsing_model.load_state_dict(trained_deoc_model)


# create reconstruction model
print("=> using pre-trained reconstruct model '{}'".format(opt.reconstruct_model))
reconstruct_model = generator.__dict__[opt.reconstruct_arch]()
trained_deoc_model = torch.load(opt.reconstruct_model)['state_dict']
reconstruct_model.load_state_dict(trained_deoc_model)

detect_parsing_model = detect_parsing_model.cuda()
reconstruct_model = reconstruct_model.cuda()
detect_parsing_model.eval()
reconstruct_model.eval()

print('load model successfully!')
start = time.time()
num = 0

print('=> Loading images and processing')
for i, fname in enumerate(os.listdir(opt.test_dir)):
    print('\rProcessing: {}/{}'.format(i+1, len(os.listdir(opt.test_dir))), end="")
    if is_image_file(fname):
        num += 1
        img_path = os.path.join(opt.test_dir, fname)
        split_filename = os.path.splitext(fname)[0]

        #image pre process
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        img = img.resize((opt.image_size, opt.image_size), Image.BILINEAR)
        img = np.array(img)
        img = img.transpose(2, 0, 1)  # HWC->CHW
        deoc_img = torch.from_numpy(np.ascontiguousarray(img[::-1, :, :])).contiguous().float()
        for c, m in zip(deoc_img, opt.deoc_norm_mean):
            c.sub_(m)
        img_var = deoc_img.view(1, 3, opt.image_size, opt.image_size).cuda()  # CHW->BCHW

        ##detection and parsing stage
        seg_mask, parsing_mask = detect_parsing_model(img_var)
        seg_mask, seg_save = hard_prob(seg_mask)
        Prob = F.softmax(parsing_mask, dim=1)
        _, parsing_mask = Prob.topk(1, 1)
        parsing_save = parsing_mask.data.cpu().numpy().squeeze().astype(np.float32)
        parsing_mask = get_one_hot(parsing_mask[:,0,:,:].cpu(), 7)

        ##reconstruction stage
        gen_face = reconstruct_model(img_var, seg_mask.cuda(), parsing_mask.cuda())
        gen_face = gen_face[0, :, :, :]
        gen_face = gen_face.data.cpu().mul(0.5).add(0.5).mul(255).byte().numpy().transpose(1, 2, 0)
        gen_face = gen_face[:, :, (2, 1, 0)]

        for output, save_path, form in zip([seg_save, parsing_save, gen_face], ['detect_mask', 'parsing_mask', 'rec_face'], ['.png','.png','.png']):

            if not os.path.exists(os.path.join(opt.save_path, save_path)):
                os.mkdir(os.path.join(opt.save_path, save_path))

            if save_path=='parsing_mask':
                output = Image.fromarray(output).convert('L')
                output.putpalette(opt.paletteID)
                output.save(os.path.join(opt.save_path, save_path, split_filename+form))
            else:
                path = os.path.join(opt.save_path, save_path, split_filename +form)
                cv2.imwrite(path, output)

runtime = time.time() - start
print('\nDone! Totally spend: {:.4f}s.  Average time: {:.4f}s'.format(runtime, runtime/num))
