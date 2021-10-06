import torch
import numpy as np
import builtins

import torch.optim as optim
import time
import os
import torch.distributed as dist
import torch.multiprocessing as mp


from model import model_builder
from config import *
from dataloader import *
from utils import interleave, de_interleave, loss_masked, compute_MeanSTD, print_tensor


def main(args):
        
    if os.path.exists(args.checkpoints_dir):
        print("checkpoint dir already exists")
    else:
        os.mkdir(args.checkpoints_dir)
        print("checkpoint dir created")
        
    ngpus_per_node = [int(i) for i in args.ngpus_per_node.split(",")]
    current_node_GPU_counts=ngpus_per_node[args.rank]
    
    if args.paralelization_type=="DDP":
        args.world_size = np.sum(ngpus_per_node)
        mp.spawn(main_worker, nprocs=current_node_GPU_counts, args=(ngpus_per_node, args, current_node_GPU_counts))
    else:
        main_worker(int(args.default_cuda_id), ngpus_per_node, args , current_node_GPU_counts)

      

def main_worker(gpu, ngpus_per_node, args, current_node_GPU_counts):
    
    # supress print if it is not the master process       
    if args.paralelization_type=="DDP" and gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    
    ########################## Model ##########################
    if args.clip_max_norm > 0:
        print("Gradient Clipping will be used")
        
    rank=-1
    if args.dataset=="nyu":
        num_j=14
    elif args.dataset=="icvl":
        num_j=16
    model= model_builder(args.model_name,num_joints=num_j)
    
    device_IDs=[int(i) for i in args.device_IDs.split(",")]
        
    default_cuda_id = "cuda:{}".format(args.default_cuda_id)

    
    if args.paralelization_type=="DDP":
        assert len(device_IDs)==current_node_GPU_counts
        
        ngpus_per_node_padded=[0]+ngpus_per_node
        rank = np.sum(ngpus_per_node_padded[:args.rank+1]) + gpu
        
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=rank)
        torch.distributed.barrier()
        
        print("All processes joined, ready to start!")
        
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        torch.cuda.set_device("cuda:{}".format(device_IDs[gpu]))
        model.cuda(device_IDs[gpu])
        
        args.batch_size = int(args.batch_size / current_node_GPU_counts)
        #args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_IDs[gpu]],find_unused_parameters=True)
        
        device = device_IDs[gpu]

        
    elif args.paralelization_type=="DP":
        device = torch.device(default_cuda_id)
        model=model.to(device)
        model=torch.nn.DataParallel(model,device_ids=device_IDs)

    
    elif args.paralelization_type=="N":
        device = torch.device(default_cuda_id)
        torch.cuda.set_device(device)
        model = model.cuda()


    if args.model_path is not None:
        if args.paralelization_type=="N":
            data=torch.load(args.model_path)
            model.load_state_dict(data["model"])
        elif args.paralelization_type=="DP":
            data=torch.load(args.model_path)
            model.module.load_state_dict(data["model"])
        elif args.paralelization_type=="DDP":
            loc='cuda:{}'.format(device)
            data=torch.load(args.model_path, map_location=loc)
            model.module.load_state_dict(data["model"])
        print("Model loaded");


    ########################## Dataset and Optimizer ##########################

    labled_train, unlabeled_train = DATA_Getters(args)
    
    if args.paralelization_type=="DDP":
        labeled_sampler = torch.utils.data.distributed.DistributedSampler(labled_train)
        unlabeled_sampler = torch.utils.data.distributed.DistributedSampler(unlabeled_train)

    else:
        labeled_sampler = None
        unlabeled_sampler = None


    trainloader_labeled = torch.utils.data.DataLoader( labled_train, batch_size=args.batch_size,
               shuffle=(labeled_sampler is None), num_workers=args.num_workers, pin_memory=True,
               sampler=labeled_sampler, drop_last=True)

    trainloader_unlabeled = torch.utils.data.DataLoader( unlabeled_train, batch_size=args.batch_size * args.mu,
               shuffle=(unlabeled_sampler is None), num_workers=args.num_workers, pin_memory=True,
               sampler=unlabeled_sampler, drop_last=True)



    optimizer_type=args.optimizer
    
    if optimizer_type=="adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    elif optimizer_type=="sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,momentum=args.momentum,nesterov=args.nesterov)
    
    scheduler_type=args.scheduler
    
    if scheduler_type=="steplr": 
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=args.decay_step,gamma=args.learning_rate_decay)
    elif scheduler_type=="cosineWarmap":
        scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T0, T_mult=args.Tmult, eta_min=args.eta_min)
    elif scheduler_type=="cosine":
        scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.num_epoch, T_mult=args.Tmult, eta_min=args.eta_min)
        

    if args.LossFunction=="L2":
        lossFunction=torch.nn.MSELoss()
        masked_lossFunc=torch.nn.MSELoss(reduction='none')

    if args.LossFunction=="L1":
        lossFunction=torch.nn.L1Loss()
        masked_lossFunc=torch.nn.L1Loss(reduction='none')


    torch.backends.cudnn.benchmark = True

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
        print("fp16_scaler being used!")
        
    ########################## Main Loop ##########################
    
    Train(model,trainloader_labeled, trainloader_unlabeled, args,lossFunction,masked_lossFunc, optimizer,device,fp16_scaler, scheduler, rank)
    model_name="savedModel_E{}.pt".format(args.num_epoch)
    data={"model":(model.module.state_dict() if not args.paralelization_type=="N" else model.state_dict()) , "args":args,"optimizer":optimizer.state_dict()}
    torch.save(data, os.path.join(args.checkpoints_dir,model_name ))
    
    print('Finished Training')




################################################ Functions #####################

def Train(model,trainloader_labeled, trainloader_unlabeled, args,lossFunction,masked_lossFunc,optimizer,device,fp16_scaler, scheduler, rank):

    thresholds = torch.tensor([1.4 , 4.7 , 1.7 , 5.8, 1.3, 5.6, 1.8, 5.9, 2.3, 4.8, 8, 8, 11.2, 12.8])[None,...].cuda(device, non_blocking=True) # (1,num_joints)    
    ms=0
    model.train()
    
    if args.paralelization_type=="DDP":
        trainloader_unlabeled.sampler.set_epoch(0)
    
    unlabeled_iterator= iter(trainloader_unlabeled)
    lrs=[]
    
    for epoch in range(args.num_epoch):
        running_loss , psudo_loss, Unlabled_Loss, true_strongloss , confidence_stats= [],[], [], [], []

    
        if args.paralelization_type=="DDP":
            trainloader_labeled.sampler.set_epoch(epoch+1)
                

        start_time_iter = time.time()
        start_time_iter2 = time.time()

        for i, data in enumerate(trainloader_labeled, 0):
            
            if args.paralelization_type=="DDP":
                img_labeled, gt_uvd, com, cubesize , joint_mask, visible_mask = data[0], data[1].cuda(device, non_blocking=True), data[4].cuda(device, non_blocking=True), data[6].cuda(device, non_blocking=True), data[7].cuda(device, non_blocking=True), data[8].cuda(device, non_blocking=True)           
            else:
                img_labeled, gt_uvd, com, cubesize , joint_mask, visible_mask= data[0], data[1].to(device), data[4].to(device), data[6].to(device), data[7].to(device), data[8].to(device)
            
            try:
                img_weak , img_strong, gt2Dcrop_weak, M_weakToOrig, joint_mask_weak, M_OrigToStrong, gt2Dcrop_strong , randomScale, randomComJitter, M , cube_size_strong = unlabeled_iterator.next()
            except:
                if args.paralelization_type=="DDP":
                    trainloader_unlabeled.sampler.set_epoch(epoch+1)

                unlabeled_iterator= iter(trainloader_unlabeled)

                img_weak , img_strong, gt2Dcrop_weak, M_weakToOrig, joint_mask_weak, M_OrigToStrong, gt2Dcrop_strong , randomScale, randomComJitter, M , cube_size_strong = unlabeled_iterator.next()
                        

            gt_uvd=Normalize_depth(gt_uvd,sizes=cubesize,coms=com,add_com=False)

            inputs = interleave(torch.cat((img_labeled, img_weak, img_strong)), 2*args.mu+1).cuda(device, non_blocking=True)

            # forward + backward + optimize
            
            with torch.cuda.amp.autocast(fp16_scaler is not None):

                outputs, heatmaps = model(inputs,return_heatmap=True)
                outputs = de_interleave(outputs, 2*args.mu+1)
                heatmaps = de_interleave(heatmaps, 2*args.mu+1)

                out_labeled = outputs[:args.batch_size]

                # out_weak, out_strong = outputs[args.batch_size:].chunk(2)
                
                out_weak, out_strong = outputs[args.batch_size:((args.mu+1)*args.batch_size)] , outputs[((args.mu+1)*args.batch_size):((2*args.mu+1)*args.batch_size)]
                heatmaps = heatmaps[args.batch_size:((args.mu+1)*args.batch_size)]
                
                del outputs


                with torch.no_grad():
                    psudo_labels = WeakToStrong(out_weak , M_weakToOrig.cuda(device, non_blocking=True), M.cuda(device, non_blocking=True), M_OrigToStrong.cuda(device, non_blocking=True), randomScale.cuda(device, non_blocking=True), randomComJitter.cuda(device, non_blocking=True), cube_size_strong.cuda(device, non_blocking=True))
                    psudo_labels = Normalize_depth(psudo_labels,sizes=cube_size_strong.cuda(device, non_blocking=True),coms=com,add_com=False)
                    _, stds = compute_MeanSTD(heatmaps)
                    confident_predictions = (stds < thresholds).float()[...,None] #(B,num_joints,1)
                    stats = (torch.sum(confident_predictions,dim=0)/confident_predictions.shape[0]).squeeze()



                out_strong = Normalize_depth(out_strong,sizes=cube_size_strong.cuda(device, non_blocking=True),coms=com,add_com=False)

                out_labeled = Normalize_depth(out_labeled,sizes=cubesize.cuda(device, non_blocking=True),coms=com,add_com=False)

                loss_labeled = lossFunction(out_labeled, gt_uvd)
                unlabeled_loss = loss_masked(psudo_labels, out_strong, confident_predictions, masked_lossFunc)


                loss = loss_labeled + args.unlabeled_weight * unlabeled_loss

                with torch.no_grad():
                    strong_ = WeakToStrong(gt2Dcrop_weak.cuda(device, non_blocking=True) , M_weakToOrig.cuda(device, non_blocking=True), M.cuda(device, non_blocking=True), M_OrigToStrong.cuda(device, non_blocking=True), randomScale.cuda(device, non_blocking=True), randomComJitter.cuda(device, non_blocking=True), cube_size_strong.cuda(device, non_blocking=True))
                    strong_ = Normalize_depth(strong_,sizes=cube_size_strong.cuda(device, non_blocking=True),coms=com,add_com=False)
                    gt2Dcrop_strong = Normalize_depth(gt2Dcrop_strong.cuda(device, non_blocking=True),sizes=cube_size_strong.cuda(device, non_blocking=True),coms=com,add_com=False)

                    val=torch.max(torch.norm(strong_-gt2Dcrop_strong.cuda(device, non_blocking=True),dim=-1))

                    true_strong= lossFunction(gt2Dcrop_strong, out_strong)


                    if val>1e-4:
                        f= open("diog.txt","a+")
                        f.write(f"detected: {val}"+"\r\n")
                        f.close()

            optimizer.zero_grad()
            
            if fp16_scaler is None:
                loss.backward()
                if args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                if args.clip_max_norm > 0:
                    fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
            
            

            with torch.no_grad():
                gt2Dcrop_weak = Normalize_depth(gt2Dcrop_weak,sizes=cube_size_strong,coms=com,add_com=False)
                out_weak = Normalize_depth(out_weak,sizes=cube_size_strong.cuda(device, non_blocking=True),coms=com,add_com=False)
                psudo_labele_loss = lossFunction(out_weak, gt2Dcrop_weak.cuda(device, non_blocking=True))



            running_loss.append(loss.item())
            psudo_loss.append(psudo_labele_loss.item())
            Unlabled_Loss.append(unlabeled_loss.item())
            true_strongloss.append(true_strong.item())
            confidence_stats.append(stats)

            if i%200==0:
                message=f"average loss: {loss.item():.5f}  | time: {(time.time()-start_time_iter):.1f} Secs\n"
                print(message)
                start_time_iter = time.time()
                f= open("log.txt","a+")
                f.write(message+"\r\n")
                f.close()

        stats=print_tensor( torch.mean(torch.stack(confidence_stats),dim=0) )
        message=f"Epoch: {epoch+1} , Labeled Loss: {np.mean(running_loss):.3f}, Unlabled Loss: {np.mean(Unlabled_Loss):.3f}, Psudo_labelLoss: {np.mean(psudo_loss):.3f}, STD Psudo_labelLoss: {np.std(psudo_loss):.3f}, TrueStrong: {np.mean(true_strongloss):.3f}, Total Time: {(time.time()-start_time_iter2)/60:.2f} mins\nstats: {stats}\n"
        print(message)
        f= open("log.txt","a+")
        f.write(message+"\r\n")
        f.close()
        
        #lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

        # Save the model
        if ( args.paralelization_type!="DDP" or (args.paralelization_type=="DDP" and rank == 0) ) and epoch%2==0 and epoch!=0:
            model_name="savedModel_E{}.pt".format(epoch+1)
            data={"model":(model.module.state_dict() if not args.paralelization_type=="N" else model.state_dict()) , "args":args,"optimizer":optimizer.state_dict()}
            torch.save(data, os.path.join(args.checkpoints_dir,model_name ))

        
    
    #torch.save(lrs,"lrs.pt")
    return



def DATA_Getters(args):
    
    cubic_size=[args.cubic_size,args.cubic_size,args.cubic_size]
    image_size=(args.cropSize,args.cropSize)

    datase_length = DATASET_LENGTHS[args.dataset]
    
    if args.subsetLength==-1: #full dataset
        subset=None
    else:
        np.random.seed(args.randseed)
        labeled_subset = np.random.choice([i for i in range(datase_length)],args.subsetLength,replace=False)
        unlabeled_subset = list( set([i for i in range(datase_length)])-set(labeled_subset) )
        
        e=time.time()
        e=int((e-int(e))*1000000)
        np.random.seed(e)
    
    base_parameters=dict(basepath=args.datasetpath, train=True, cropSize=image_size, 
            doJitterRotation=args.RotAugment, doAddWhiteNoise=args.doAddWhiteNoise, sigmaNoise=args.sigmaNoise,
            rotationAngleRange=[-45.0, 45.0], comJitter=args.comjitter, RandDotPercentage=args.RandDotPercentage,
            indeces=labeled_subset, cropSize3D=cubic_size, do_norm_zero_one=False, 
            random_seed=args.randseed,drop_joint_num=args.drop_joint_num,center_refined=args.center_refined,
            horizontal_flip=args.horizontal_flip, scale_aug=args.scale_aug)


    if args.dataset=="nyu":
        print("NYU dataset will be used")
        specs=dict(doLoadRealSample=(args.dataset_type=="real"),camID=args.camid,basepath=os.environ.get('NYU_PATH', args.datasetpath))
        base_parameters.update(specs)
        labled_train=NYUHandPoseDataset(**base_parameters)
        
    elif args.dataset=="icvl":
        print("ICVL dataset will be used")
        specs=dict(basepath= os.environ.get('ICVL_PATH', args.datasetpath))
        base_parameters.update(specs)
        labled_train=ICVLHandPoseDataset(**base_parameters)


     
    unlabeled_train = UnlabledDataset( basepath=specs["basepath"], cropSize=image_size, indeces=unlabeled_subset,
                                    cropSize3D=cubic_size,random_seed=args.randseed,drop_joint_num=args.drop_joint_num,dataset_name=args.dataset,camid=args.camid)

    print("Total number of labeled samples to use for Training: %d"%(len(labled_train)))
    print("Total number of Unlabled samples to use for Training: %d"%(len(unlabeled_train)))
    
    return labled_train, unlabeled_train



####################################

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)	
    
    
    
    
