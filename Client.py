import rich
import torch
import utils
from copy import deepcopy
from typing import Tuple, Union
from collections import OrderedDict
from data.utils import get_dataloader
from fedlab.utils.serialization import SerializationTool
from model import affectnet_model,mead_model,seed_model


def update_permodel(per_model,global_model):
    per_state=deepcopy(per_model.state_dict())
    glo_state=deepcopy(global_model.state_dict())
    for key in glo_state:
        per_state[key]=glo_state[key]
    per_model.load_state_dict(per_state,strict=False)
    return per_model


def get_client_model(id,global_model):
    if id<10:
        model=affectnet_model().cpu()
    elif id <20:
        model = mead_model().cpu()
    else :
        model=seed_model().cpu()
    # model=update_permodel(model,global_model)
    return model

class FedClient:
    def __init__(
        self,
        client_id: int,
        alpha: float,
        beta: float,
        global_model: torch.nn.Module,
        criterion: Union[torch.nn.CrossEntropyLoss, torch.nn.MSELoss],
        batch_size: int,
        local_epochs: int,
        valset_ratio: float,
        logger: rich.console.Console,
        gpu: int,
        train:str,
        val:str,
        device:str
    ):
        
        self.device = device
        self.logger = logger

        self.local_epochs = local_epochs
        self.criterion = criterion
        self.id = client_id
        
        self.alpha = alpha
        self.beta = beta
        self.trainloader=train
        self.valloader = val
        self.iter_trainloader = iter(self.trainloader)

        self.glo_model = deepcopy(global_model)
        self.per_model=get_client_model(client_id,global_model)
        self.glo_Projection=None
        return
    # def update_per_model(self):
        
        # return

    def get_data_batch(self):
        try:
            x, y = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            x, y = next(self.iter_trainloader)

        return x.to(self.device), y.to(self.device)

    def train(                                                   
        self,
        global_model: torch.nn.Module,
        
        eval_while_training=False,
        global_Projection=None,
    ):
        # self.model.load_state_dict(global_model.state_dict())    
        self.per_model=update_permodel(self.per_model,global_model) 
        # if eval_while_training:    
        #     loss_before, acc_before = utils.eval(
        #         self.model, self.valloader, self.criterion, self.device
        #     )
        self._train(global_Projection) #

        if eval_while_training:   
            loss_after, acc_after = utils.eval(
                self.per_model, self.valloader, self.criterion, self.device
            )
            self.logger.log(     
                "client [{}] [red]test_loss: {:.4f}    [blue]acc: {:.2f}% ".format(
                    self.id,
                    loss_after,
                    acc_after * 100.0,
                )
            )
        
        per_state=deepcopy(self.per_model.state_dict())
        glo_state=deepcopy(global_model.state_dict())
        for key in glo_state:
            glo_state[key]=per_state[key]
        global_model.load_state_dict(glo_state,strict=False)
        return SerializationTool.serialize_model(global_model) ,self.glo_Projection
        # return glo_state

    def _train(self,global_Projection=None):
        
        for _ in range(self.local_epochs):
            for _ in range(int(len(self.trainloader)/3)):
                temp_model = deepcopy(self.per_model) 
                data_batch_1 = self.get_data_batch()  
                grads = self.compute_grad(temp_model, data_batch_1) 
                for param, grad in zip(temp_model.parameters(), grads):
                    param.data.sub_(self.alpha * grad.to(self.device))
            for _ in range(len(self.trainloader)-int(len(self.trainloader)/3)):
                data_batch_2 = self.get_data_batch()
                grads_1st = self.compute_grad(temp_model, data_batch_2,Projection_param=global_Projection)   


                per_Projection = deepcopy(self.per_model.projection_layer.get_parameters()) #save_param
                self.per_model.projection_layer.set_parameters(global_Projection)
                data_batch_3 = self.get_data_batch()

                grads_2nd = self.compute_grad(
                    self.per_model, data_batch_3, v=grads_1st, second_order_grads=True,Projection_param=per_Projection    
                )
                self.glo_Projection = deepcopy(self.per_model.projection_layer.get_parameters())
                self.per_model.projection_layer.set_parameters(per_Projection)

                # NOTE: Go check https://github.com/KarhouTam/Per-FedAvg/issues/2 if you confuse about the model update.
                for param, grad1, grad2 in zip(
                    self.per_model.parameters(), grads_1st, grads_2nd
                ):
                    param.data.sub_(self.beta * grad1 - self.beta * self.alpha * grad2)

        

    def compute_grad(
        self,
        model: torch.nn.Module,
        data_batch: Tuple[torch.Tensor, torch.Tensor],
        v: Union[Tuple[torch.Tensor, ...], None] = None,
        second_order_grads=False,
        Projection_param=None
    ):
        x, y = data_batch
        
        model.to(self.device)

        
        if torch.isnan(x).any() or torch.isnan(y).any():
            raise ValueError("Input data contains NaN values")
        
        
        if second_order_grads:
            frz_model_params = deepcopy(model.state_dict())
            delta = 1e-5  
            
           
            def safe_update(params, delta_grad):
                new_params = OrderedDict()
                for (name, param), dg in zip(model.named_parameters(), delta_grad):
                   
                    update = delta * dg
                    if torch.isnan(update).any() or torch.isinf(update).any():
                        update = torch.zeros_like(update)
                    new_params[name] = param + update
                return new_params
            
            
            dummy_model_params_1 = safe_update(model.named_parameters(), v)
            dummy_model_params_2 = safe_update(model.named_parameters(), [-g for g in v])

            
            model.load_state_dict(dummy_model_params_1, strict=False)
            logit_1 = model(x)
            loss_1 = self.criterion(logit_1, y)
            
            
            grads_1 = list(torch.autograd.grad(loss_1, model.parameters()))
            torch.nn.utils.clip_grad_norm_(grads_1, max_norm=1.0)  

            
            model.load_state_dict(dummy_model_params_2, strict=False)
            logit_2 = model(x)
            loss_2 = self.criterion(logit_2, y)
            
            if Projection_param is not None:
                
                proj_params = model.projection_layer.get_parameters()
                if proj_params is not None:
                    loss_2_2 = self.calculate_regularization_loss(
                        Projection_param.to(self.device),
                        proj_params
                    )
                    
                    if not torch.isnan(loss_2_2) and not torch.isinf(loss_2_2):
            
                        loss_2 = loss_2 + 0.6 * loss_2_2
            
            grads_2 = list(torch.autograd.grad(loss_2, model.parameters()))
            torch.nn.utils.clip_grad_norm_(grads_2, max_norm=1.0)  

            
            model.load_state_dict(frz_model_params)

            
            grads = []
            for g1, g2 in zip(grads_1, grads_2):
                diff = g1 - g2
                
                if torch.isnan(diff).any() or torch.isinf(diff).any():
                    diff = torch.zeros_like(diff)
                grads.append(diff / (2 * delta))
            
            return grads

        else:
            
            logit = model(x)
            loss = self.criterion(logit, y)
            
            
            if torch.isnan(loss) or torch.isinf(loss):
               
                return [torch.zeros_like(p) for p in model.parameters()]
            
            if Projection_param is not None:
                proj_params = model.projection_layer.get_parameters()
                if proj_params is not None:
                    loss_2 = self.calculate_regularization_loss(
                        Projection_param.to(self.device),
                        proj_params
                    )
                    if not torch.isnan(loss_2) and not torch.isinf(loss_2):
                        loss = loss + 0.6 * loss_2
            
            grads = list(torch.autograd.grad(loss, model.parameters()))
        
        
        valid_grads = []
        for g in grads:
            if g is None: 
                continue
            if torch.isnan(g).any() or torch.isinf(g).any():
                valid_grads.append(torch.zeros_like(g))
            else:
                valid_grads.append(g)
                
        torch.nn.utils.clip_grad_norm_(valid_grads, max_norm=1.0)
        return valid_grads
        
    def calculate_regularization_loss(sekf,local_params, global_params):
        
        loss = torch.nn.functional.mse_loss(local_params, global_params)
        
        return loss
    def pers_N_eval(self, global_model: torch.nn.Module, pers_epochs: int):
        self.model.load_state_dict(global_model.state_dict())

        loss_before, acc_before = utils.eval(
            self.model, self.valloader, self.criterion, self.device
        )
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.alpha)
        for _ in range(pers_epochs):
            x, y = self.get_data_batch()
            logit = self.model(x)
            loss = self.criterion(logit, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_after, acc_after = utils.eval(
            self.model, self.valloader, self.criterion, self.device
        )
        self.logger.log(
            "client [{}] [red]loss: {:.4f} -> {:.4f}   [blue]acc: {:.2f}% -> {:.2f}%".format(
                self.id, loss_before, loss_after, acc_before * 100.0, acc_after * 100.0
            )
        )
        return {
            "loss_before": loss_before,
            "acc_before": acc_before,
            "loss_after": loss_after,
            "acc_after": acc_after,
        }
