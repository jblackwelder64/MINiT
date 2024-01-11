import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from time import time
from sys import argv
import random
from warmup_scheduler import GradualWarmupScheduler

import comm
import dataset
from sam import SAM
import hyperparameters
from utils import collate_fn
from detectron2_launch import launch
from distributed_sampler_wrapper import DistributedSamplerWrapper
from runtime_augmentations import cutmix_data, mixup_data, mix_criterion

import minit

EPOCH = 101
DEBUG = True
MODEL_TYPE = argv[1].lower()
TASK = 'Sex'
DATA_DIR = '.' if len(argv) <= 2 else argv[2]
CKPT_DIR = '.' if len(argv) <= 3 else argv[3]
XLS_PATH = '/home/jwb2168/RANN_scores_demographics_final.xls'
NUM_GPUS_PER_MACHINE = 1
FINETUNE = True
TRAINABLE_PARAMS = [
    'vit.transformer.layers.5.0.fn.norm.weight', 'vit.transformer.layers.5.0.fn.norm.bias', 
    'vit.transformer.layers.5.0.fn.fn.to_qkv.weight', 'vit.transformer.layers.5.0.fn.fn.to_out.0.weight', 
    'vit.transformer.layers.5.0.fn.fn.to_out.0.bias', 'vit.transformer.layers.5.1.fn.norm.weight', 
    'vit.transformer.layers.5.1.fn.norm.bias', 'vit.transformer.layers.5.1.fn.fn.net.0.weight',
     'vit.transformer.layers.5.1.fn.fn.net.0.bias', 'vit.transformer.layers.5.1.fn.fn.net.3.weight', 
     'vit.transformer.layers.5.1.fn.fn.net.3.bias', 'vit.mlp_head.0.weight', 'vit.mlp_head.0.bias', 
     'vit.mlp_head.1.weight', 'vit.mlp_head.1.bias', 'linear.weight', 'linear.bias']

hyperparameter_fn = getattr(hyperparameters, f"{MODEL_TYPE}_hyperparameters")
model_hyperparameters = hyperparameter_fn()
model_hyperparameters['task'] = TASK
model_hyperparameters['model_name'] = MODEL_TYPE
batch_size = model_hyperparameters['batch_size']

num_workers = 0
pin_memory = True
last_time = 0

"""
Copy from detectron2
"""
# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# from detectron2.utils import comm

__all__ = ["DEFAULT_TIMEOUT", "launch"]

DEFAULT_TIMEOUT = timedelta(minutes=30)


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def launch_2(
    main_func,
    num_gpus_per_machine,
    num_machines=1,
    machine_rank=0,
    dist_url=None,
    args=(),
    timeout=DEFAULT_TIMEOUT,
):
    """
    Launch multi-gpu or distributed training.
    This function must be called on all machines involved in the training.
    It will spawn child processes (defined by ``num_gpus_per_machine``) on each machine.

    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_gpus_per_machine (int): number of GPUs per machine
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine
        dist_url (str): url to connect to for distributed jobs, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to "auto" to automatically select a free port on localhost
        timeout (timedelta): timeout of the distributed workers
        args (tuple): arguments passed to main_func
    """
    world_size = num_machines * num_gpus_per_machine
    if world_size >= 1:
        # https://github.com/pytorch/pytorch/pull/14391
        # TODO prctl in spawned processes

        if dist_url == "auto":
            assert num_machines == 1, "dist_url=auto not supported in multi-machine jobs."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"
        if num_machines > 1 and dist_url.startswith("file://"):
            logger = logging.getLogger(__name__)
            logger.warning(
                "file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://"
            )

        mp.spawn(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(
                main_func,
                world_size,
                num_gpus_per_machine,
                machine_rank,
                dist_url,
                args,
                timeout,
            ),
            daemon=False,
        )
    else:
        main_func(*args)



def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    dist_url,
    args,
    timeout=DEFAULT_TIMEOUT,
):
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL",
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            timeout=timeout,
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(dist_url))
        raise e

    # Setup the local process group (which contains ranks within the same machine)
    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    main_func(*args)


"""
End copy from detectron2
"""


def reproducibility(seed):
    torch.manual_seed(seed)
    random.seed(seed) # random
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Used https://stackoverflow.com/questions/67295494/correct-validation-loss-in-pytorch
def validate(loader, model, criterion, amp_enabled):
    correct = 0
    total = 0
    running_loss = 0.0
    model.eval()
    loss = 0
    denom = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.squeeze(1)
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            cuda = torch.cuda.is_available()

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(inputs.float())                
                labels = labels.squeeze(1).long()
                loss+=criterion(outputs, labels)
            denom+=len(labels)
    print('validation loss: ', loss/denom)
    print('validation dataset size == ', denom)



def train(model, load_pretrained, datasets, lr, alpha, warmup_epochs, multiplier, weight_decay, cutmix, mixup, ckpt_folder, amp_enabled, **kwargs):
    last_time = time()
    arguments = dict(locals(), **kwargs)

    reproducibility(seed=1)

    cur_rank = comm.get_local_rank()
    torch.cuda.set_device(cur_rank)
    
    net = model(**arguments)
    net.cuda(cur_rank)

    if comm.is_main_process():
        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Total Parameters: {pytorch_total_params}")
        
    # Loading pretrained model.
    if comm.is_main_process() and len(argv) >= 5:
        checkpoint = torch.load(f"{ckpt_folder}/{argv[4]}.pt") #, map_location='cur_rank')
        state = checkpoint['model_state_dict']
        net.load_state_dict(state, strict=True)

    
    # Freezing all layers not contained in TRAINABLE_PARAMS if finetuneing
    if FINETUNE:
        for name, param in net.named_parameters():
            if name in TRAINABLE_PARAMS:
                print('Will optimize layer: ', name)
                param.requires_grad = True
            else:
                print('Will not optimize layer: ', name)
                param.requires_grad = False

    net = DDP(net, device_ids=[cur_rank], broadcast_buffers=False, find_unused_parameters=True)
    
    criterion = nn.CrossEntropyLoss()
    criterion_valid = nn.CrossEntropyLoss(reduction='sum')
    optimizer = SAM(net.parameters(), optim.Adam, lr=lr, betas=(.9, .999), weight_decay=weight_decay) # SAM + Adam
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, EPOCH) #  SAM: LR scheduling should be applied to the base optimizer
    scheduler_warmup = GradualWarmupScheduler(optimizer.base_optimizer, multiplier=multiplier, total_epoch=warmup_epochs, after_scheduler=cosine)

    # Due to dataset access limitations, you would need to implement get_dataset yourself to return an MRIDataset object.
    # trainset_augment = dataset.get_dataset_JB(datasets, [TASK], DATA_DIR, augment=True)[1]
    _, trainset_augment, testset = dataset.get_dataset_JB(DATA_DIR, [TASK], XLS_PATH, train_augment=False, train_percent=.8)
    
    if comm.is_main_process():
        print('Number of examples in each fold')
        print('trainset augment:', trainset_augment.__len__())

    # Get class counts.
    positive = 0 
    negative = 0 
    for i in trainset_augment.list_IDs:
        print('i==',i)
        label = trainset_augment.labels[trainset_augment.list_IDs[i]]
        if label[TASK] == 1:
            positive += 1
        else:
            negative += 1

    # Construct weights for weighted random sampler.
    weights = []
    for i in trainset_augment.list_IDs:
        print('i==',i)
        label = trainset_augment.labels[trainset_augment.list_IDs[i]][TASK]
        print('label: ', label)
        weights.append(1.0/positive if label == 1 else 1.0/negative)
    
    sampler = DistributedSamplerWrapper(sampler=WeightedRandomSampler(weights, len(weights)), num_replicas=comm.get_world_size(), rank=comm.get_local_rank(), shuffle=False)
    
    # Dataloaders
    trainloader_augment = torch.utils.data.DataLoader(
        trainset_augment, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn, sampler=sampler)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn, sampler=None)

    if comm.is_main_process():
        print('Finished loading dataset', flush=True)

    if comm.is_main_process():
        print('Initiating training', flush=True)

    for epoch in range(EPOCH):
        trainloader_augment.sampler.set_epoch(epoch)
        net.train()
        epoch_loss = torch.tensor(0, device=cur_rank).float()
        
        print(f"Rank {comm.get_local_rank()} is beginning Epoch {epoch} after {(time() - last_time)/60} minutes.", flush=True)
        last_time = time()
        for i, (inputs, labels) in enumerate(trainloader_augment):
            net.train()

            inputs = inputs.squeeze(1)
            # print('inputs shape: ', inputs.shape)

            # get the inputs
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            mixing = False

            cuda = torch.cuda.is_available()
            random_result = random.random()
            if random_result < mixup:
                # print('Here!')
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha, cuda)
                # print('1:inputs, targets_a, targets_b, lam ',inputs, targets_a, targets_b,lam)
                mixing = True
            elif random_result < mixup + cutmix:
                # print('Here!!')
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, alpha, cuda)
                # print('2:inputs, targets_a, targets_b, lam: ',inputs, targets_a, targets_b,lam)
                mixing = True

            # forward + backward + optimize
            ## forward pass (autocasted)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = net(inputs.float())
                # print('outputs: ', outputs)
                
                if mixing:
                    targets_a = targets_a.squeeze(1).long()
                    targets_b = targets_b.squeeze(1).long()
                
                labels = labels.squeeze(1).long()

                loss = mix_criterion(criterion, outputs, targets_a, targets_b, lam) if mixing else criterion(outputs, labels)
            
            if DEBUG and comm.is_main_process():
                print(f"Rank {cur_rank}'s Epoch {epoch} Local Loss is {loss}.")

            ## Break out if NaN loss.
            # print('Mixing: ', mixing)
            # print('Random result, mixup, cutmix ', random_result, mixup, cutmix)
            # print('Me printing out loss: ', loss)
            # print('Me printing out type(loss): ', type(loss))
            if torch.isnan(loss).any():
                print(torch.isnan(inputs))
                print(inputs)
                print(outputs)
                print(f"Breaking out because of nan in GPU {comm.get_local_rank()}")
                return
    
            ## first backward pass (Note: GradScaler cannot be used here as it does not support SAM)
            with net.no_sync():  # To compute SAM while using multiple GPUs - unsupported in nn.DataParallel
                loss.backward()
            epoch_loss += loss.detach()
            optimizer.first_step(zero_grad=True)

            ## SAM: second forward-backward pass
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                second_step_loss = mix_criterion(criterion, net(inputs.float()), targets_a, targets_b, lam) if mixing else criterion(net(inputs.float()), labels)
            second_step_loss.backward()
            optimizer.second_step(zero_grad=True)

            ## To warm up within epoch
            if (i+1) % (len(trainloader_augment)//3) == 0:
                scheduler_warmup.step()
        validate(testloader, net, criterion_valid, amp_enabled)
        if epoch % 5 == 0:
            if comm.is_main_process():
                # validate(loader, net, criterion)
                print('==> Saving model ...')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, f"{ckpt_folder}/ckpt_main_{epoch}_v3.pt")
    print(f"Rank {comm.get_local_rank()} has finished the last Epoch ({epoch}) after {(time() - last_time)/60} minutes.", flush=True)

def main():
    model_hyperparameters['ckpt_folder'] = CKPT_DIR
    train(**model_hyperparameters)

if __name__ == '__main__':
    training_start_time = time()
    # launch(
    #     main,
    #     num_gpus_per_machine=NUM_GPUS_PER_MACHINE,
    #     num_machines=1,
    #     machine_rank=0,
    #     dist_url='auto',
    #     args=()
    # )

    launch_2(
        main,
        num_gpus_per_machine=NUM_GPUS_PER_MACHINE,
        num_machines=1,
        machine_rank=0,
        dist_url='auto',
        args=()
    )
    print(f"==> Finished Training after {(time() - training_start_time)/60} minutes.")
