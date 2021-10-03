import tqdm
import torch
from dataloader import *
from torch.utils.data import Dataset,DataLoader
from model import model_builder
from HandposeEvaluation import *
import pickle
import tqdm
import sys
import os
import time
import argparse

import re
def getNumber(s):
    return int(re.findall('[0-9]+', s)[0])

def OutputToPred(outputs,cubesize,com):
    outputs[:,:,0:2]=outputs[:,:,0:2]
    return Normalize_depth(outputs,cubesize,com,add_com=True)

###############################################3
@torch.no_grad()
def evaluate(model_path, model_name, loader, args, test_set, print_detail_crop=True,print_detail_uvd=False,print_detail_xyz=False):
    
    model.load_state_dict(torch.load(model_path)["model"])
    model.eval()
    
    GT_crop, GT_UVD_orig, GT_3D_orig, GT_matrix, estimation_cropped= [], [], [], [], []
    
    loop = tqdm.tqdm(loader)#preds=[[],[],[],[]] args.joint_dim
    for i, data in enumerate(loop):
        loop.set_description(model_name)

        inputs, gt2Dcrop, gt2Dorignal, gt3Dorignal, com, M_inv, cubesize = data[0].to(device),data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device),data[5].to(device),data[6].to(device)
        
        outputs = model(inputs)

        preds=OutputToPred(outputs,cubesize,com) # (B,k,3) it should be standard, both in UV and D
        
        gt_crop=Normalize_depth(gt2Dcrop,cubesize,com,add_com=True)
        
        
        GT_crop.append(gt_crop)
        GT_UVD_orig.append(gt2Dorignal)
        GT_3D_orig.append(gt3Dorignal)
        GT_matrix.append(M_inv)

        estimation_cropped.append(preds)



    GT_crop=torch.cat(GT_crop).cpu() # (B,K,joint_dim)
    GT_UVD_orig=torch.cat(GT_UVD_orig).cpu() # (B,K,joint_dim)
    GT_3D_orig=torch.cat(GT_3D_orig).cpu() # (B,K,joint_dim)
    GT_matrix=torch.cat(GT_matrix).cpu() # (B,3,3)
    estimation_cropped=torch.cat(estimation_cropped).cpu() # (B,K,joint_dim)

    Evaluator=HandposeEvaluation(estimation_cropped,GT_crop)

    output_message=f'--------------------- {model_name} ------------------------\n'
    
    # Cropped UVD
    
    if print_detail_crop:    
        res=Evaluator.getErrorPerDimension(printOut=False)
        output_message=output_message+"\nUVD_Cropped:\n"+res+"\n###############################\n"
        
    output_message=output_message+f"\nThe error UVD in the cropped version={Evaluator.getMeanError():.3f}\n"+"###############################\n"
    
    # Original UVD
    
    prediction_UVDorig=CropToOriginal(estimation_cropped,GT_matrix.float())
    del estimation_cropped,GT_matrix
    Evaluator.update(prediction_UVDorig,GT_UVD_orig)
    
    if print_detail_uvd:    
        res=Evaluator.getErrorPerDimension(printOut=False)
        output_message=output_message+"\nUVD_Original:\n"+res+"\n###############################\n"
        
    output_message=output_message+f"\nThe error in Original UVD ={Evaluator.getMeanError():.3f}\n"+"###############################\n"

    # 3D XYZ
    estimation_xyz=test_set.convert_uvd_to_xyz_tensor( prediction_UVDorig )
    Evaluator.update(estimation_xyz,GT_3D_orig)
 
    if print_detail_xyz:    
        res=Evaluator.getErrorPerDimension(printOut=False)
        output_message=output_message+ "\n3D results:\n" + res +"\n###############################\n"
        
    output_message=output_message+f"\nFinal 3D error results: {Evaluator.getMeanError():.3f}\n\n"+ 100*"=" + "\n"


   # data=(spheres_save,g)
   # pickle_out = open("Sphpreds_gt_cropped.pickle","wb")
   # pickle.dump(data, pickle_out)
   # pickle_out.close()
    
    if args.save_results:
        f= open("results.txt","a+")
        f.write(output_message)
        f.close()
    else:
        print(output_message)

    
    


####### MAIN LOOP #############

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--batch_size', default=32,type=int, help="batch_size")
    parser.add_argument('--cuda_id', default=0, type=int, help="Cuda ID")
    parser.add_argument('--path', default="", type=str, help="the address of the dataset",required=True)
    parser.add_argument('--num_workers', default=4, type=int, help="num of subprocesses for data loading")
    parser.add_argument('--joint_dim', default=3, type=int, help="determine if it is 3D or 2D")
    parser.add_argument('--save_results', default=1, type=int, help="determine if the the results are written into a file")
    parser.add_argument('--center_refined', default=1, type=int, help="determine if the the results are written into a file")
    parser.add_argument('--dataset',default="nyu", choices=('nyu', 'icvl','msra'),type=str,help="which dataset to use")

    
    args = parser.parse_args()
    default_cuda_id = "cuda:{}".format(args.cuda_id)

    list_files=os.listdir(args.path)
    list_files.sort(key=getNumber)
    
    
    model_path=os.path.join(args.path,list_files[0])
    setting=torch.load(model_path)["args"]
    args.dataset=setting.dataset
    if args.dataset=="nyu":
        print("NYU dataset will be used")
        test_set=NYUHandPoseDataset(train=False,basepath=os.environ.get('NYU_PATH'),center_refined=args.center_refined)
        
    elif args.dataset=="icvl":
        print("ICVL dataset will be used")
        test_set=ICVLHandPoseDataset(train=False,basepath=os.environ.get('ICVL_PATH'),center_refined=args.center_refined)


    device = torch.device(default_cuda_id if torch.cuda.is_available() else "cpu")
    model = model_builder(setting.model_name,num_joints=test_set.num_joints).to(device)

    print("Initialization Done, Ready to start evaluationg...\n")

    for file in list_files:
            model_path=os.path.join(args.path,file)
            setting=torch.load(model_path)["args"]
            
            test_set.cropSize = (setting.cropSize,setting.cropSize)
            test_set.cropSize3D = [setting.cubic_size,setting.cubic_size,setting.cubic_size]

            if args.dataset=="nyu":
                test_set.camID = setting.camid
                test_set.doLoadRealSample = (setting.dataset_type=="real")
            
            testloader = DataLoader(test_set , batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory = True)

            
            evaluate(model_path,file,testloader,args,test_set,print_detail_crop=True,print_detail_uvd=False,print_detail_xyz=False)

