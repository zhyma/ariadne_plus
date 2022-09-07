import cv2
import torch
import numpy as np
import os
import arrow 
from termcolor import cprint

# ariadne
import scripts.core as network
from scripts.utils.dataset import BasicDataset
from scripts.core.tripletnet import TripletNet
from scripts.core.crossnet import CrossNet
from scripts.core.core import Ariadne, AriadnePath
from scripts.core.curvature import CurvatureVonMisesPredictor
from scripts.utils.spline import Spline, SplineMask

class AriadnePlus():

    def __init__(self, main_folder, num_segments, type_model = "STANDARD"):
        
        cprint("Initializing Ariadne+ model...", "white")

        self.num_segments = num_segments

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load Model - DEEPLAB
        if type_model == "STANDARD":
            checkpoint_deeplab = os.path.join(main_folder, 'checkpoints/model_deeplab.pth')
        elif type_model == "SYNTHETIC":
            checkpoint_deeplab = os.path.join(main_folder, 'checkpoints/model_deeplab_synt.pth')
        else:
            print("type model not known!")

        self.model = network.deeplabv3plus_resnet101(num_classes=1, output_stride=16)
        network.convert_to_separable_conv(self.model.classifier)
        checkpoint = torch.load(checkpoint_deeplab, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval() 

        cprint("Done!", "green")


    def predictImg(self, net, img, device):

        img = torch.from_numpy(BasicDataset.pre_process(np.array(img)))
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            output = net(img)
            probs = torch.sigmoid(output)
            probs = probs.squeeze(0).cpu()
            full_mask = probs.squeeze().cpu().numpy()

        return full_mask


    def runAriadne(self, img, debug=False):

        t0 = arrow.utcnow()
        if debug: print("-> Computing image binary mask ... ")
        ##################################
        # Semantic Segmentation
        ##################################
        mask = self.predictImg(net=self.model, img=img, device=self.device) 
        result = (mask * 255).astype(np.uint8)
        img_mask = result.copy()
        img_mask[img_mask < 127] = 0
        img_mask[img_mask >=127] = 255

        result_spline = []
        mask_final = None
        t3 = arrow.utcnow()

        return {"spline_msg": result_spline, 
                "img_mask": img_mask, 
                "img_final": mask_final, 
                "time": (t3-t0).total_seconds()}
        

