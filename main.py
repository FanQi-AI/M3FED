import sys

sys.path.append("data")
from copy import deepcopy
import torch
import random
import os
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.aggregator import Aggregators
from rich.console import Console
from rich.progress import track
from utils import get_args, fix_random_seed
from model import get_model,LearnableProjection,meta
from Client import FedClient
from data.utils import get_client_id_indices
from data.utils import get_dataloader,affectnet_mead_seed_DataSet,aggregate_pro_parameters,data_deal,angular_distance,cluster_parameters

if __name__ == "__main__":
    args = get_args()
    fix_random_seed(args.seed)   
    if os.path.isdir("./log") == False: 
        os.mkdir("./log")
    
    device = torch.device("cuda:"+str(args.gpu))  #GPU
    
    logger = Console(record=args.log) 
    logger.log(f"Arguments:", dict(args._get_kwargs()))  
    client_num_in_total=[i for i in range(25)] #number clients


    
    data_fea = data_deal()
    train_dataloader=[]
    val_dataloader=[]
    for client_id in client_num_in_total:
        if client_id <10: 
            data=data_fea[0]
        elif client_id <20:
            data=data_fea[1]
        else:
            data=data_fea[2]
        client_dataset= affectnet_mead_seed_DataSet(datas=data)
        trainloader, valloader = get_dataloader(client_dataset,client_id, args.batch_size)
        train_dataloader.append(trainloader)
        val_dataloader.append(valloader)


    global_model = meta().cpu() 
    global_Projection = LearnableProjection(512,384).cpu()
    global_Projection_param=global_Projection.get_parameters()
    global_model_list = [deepcopy(global_model) for _ in range(4)] ##n_clusters
    cluster_global_model_id={client_id: 0 for client_id in range(25)} #number clients
    # init clients  
    clients = [
        FedClient(
            client_id=client_id,
            alpha=args.alpha,
            beta=args.beta,
            global_model=global_model_list[cluster_global_model_id[client_id]],
            criterion=torch.nn.CrossEntropyLoss(),
            batch_size=args.batch_size,
            local_epochs=args.local_epochs,
            valset_ratio=args.valset_ratio,
            logger=logger,
            gpu=args.gpu,
            train=train_dataloader[client_id],
            val=val_dataloader[client_id],
            device=device
            
        )
        for client_id in range(25)
    ]
    # training
    logger.log("=" * 20, "TRAINING", "=" * 20, style="bold red") 
    for round in track(  range(args.global_epochs), "Training...", console=logger, disable=args.log): 
        # select clients
        selected_clients = random.sample(client_num_in_total, args.client_num_per_round)   

        model_params_cache = [] 
        model_Projection_clients = []
        # client local training
        for client_id in selected_clients:  
            serialized_model_params,glo_Projection_clients = clients[client_id].train(
                global_model=global_model,
                eval_while_training=args.eval_while_training,
                global_Projection=global_Projection_param
            )
            model_params_cache.append(serialized_model_params)
            model_Projection_clients.append(glo_Projection_clients)

        # aggregate model parameters   
        aggregated_model_params = Aggregators.fedavg_aggregate(model_params_cache) 
        clustering_labels =angular_distance(model_params_cache,aggregated_model_params)
        global_model_list = cluster_parameters(model_params_cache,clustering_labels,global_model_list)
        for client_id, new_label in zip(selected_clients, clustering_labels):
            cluster_global_model_id[client_id] = new_label


        SerializationTool.deserialize_model(global_model, aggregated_model_params) 
        global_Projection_param = aggregate_pro_parameters(model_Projection_clients)
        logger.log('round:'+str(round)+"=" * 60) 


    
