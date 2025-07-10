import torch.nn as nn
import torch
import torch.nn.functional as F

def z_score_normalize(data):
    """Z-score """
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True)
    normalized_data = (data - mean) / (std + 1e-10)  
    return normalized_data  
class elu(nn.Module):
    def __init__(self) -> None:
        super(elu, self).__init__()

    def forward(self, x):
        return torch.where(x >= 0, x, 0.2 * (torch.exp(x) - 1)) 

class linear(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super(linear, self).__init__()
        self.w = nn.Parameter(
            torch.randn(out_c, in_c) * torch.sqrt(torch.tensor(2 / in_c))
        )
        self.b = nn.Parameter(torch.randn(out_c))

    def forward(self, x):
        return F.linear(x, self.w, self.b)


class MLP_MNIST(nn.Module):
    def __init__(self) -> None:
        super(MLP_MNIST, self).__init__()
        self.fc1 = linear(28 * 28, 80)
        self.fc2 = linear(80, 60)
        self.fc3 = linear(60, 10)
        self.flatten = nn.Flatten() 
        self.activation = elu()

    def forward(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)
        x = self.activation(x)

        return x


class MLP_CIFAR10(nn.Module):
    def __init__(self) -> None:
        super(MLP_CIFAR10, self).__init__()
        self.fc1 = linear(32 * 32 * 3, 80)
        self.fc2 = linear(80, 60)
        self.fc3 = linear(60, 10)
        self.flatten = nn.Flatten()
        self.activation = elu()

    def forward(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)
        x = self.activation(x)

        return x

class meta(nn.Module):
    def __init__(self) -> None:
        super(meta, self).__init__()

        self.fc1 = nn.Linear(512, 512)
        self.BN2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)

        # self.activation = elu()
        self.activation = nn.GELU()
    def forward(self,x):

        x = self.fc1(x)
        x = self.BN2(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x
    
class LearnableProjection(nn.Module):
    def __init__(self, input_dim,output_dim):
        super(LearnableProjection, self).__init__()
        self.projection_matrix = nn.Parameter(torch.randn(input_dim, output_dim))

    def forward(self, x):
        return x @ self.projection_matrix

    def get_parameters(self):
        
        return self.projection_matrix.data.clone()

    def set_parameters(self, params):
        
        self.projection_matrix.data.copy_(params)


class affectnet_model(nn.Module):
    def __init__(self) -> None:
        super(affectnet_model, self).__init__()
        
        self.BN = nn.BatchNorm1d(1408)
        self.fc_1 = nn.Linear(1408, 512)
        self.BN1 = nn.BatchNorm1d(512)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(0.3)

       
        self.fc1 = nn.Linear(512, 512)
        self.BN2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)

        self.BN3 = nn.BatchNorm1d(512)
        self.projection_layer = LearnableProjection(512, 384)  

        
        self.BN5 = nn.BatchNorm1d(384)
        self.fc_3 = nn.Linear(384, 256)
        self.BN6 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc_4 = nn.Linear(256, 8)
        self.softmax = nn.Softmax(dim=1) 
    def forward(self, x):
        x = self.BN(x)
        x = self.fc_1(x)
        x = self.BN1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        x = self.fc1(x)
        x = self.BN2(x)
        x = self.activation(x)
        x = self.fc2(x)

        x = self.BN3(x)
        x = self.activation(x)  
        x = self.projection_layer(x)

        x = self.BN5(x)
        x = self.fc_3(x)
        x = self.BN6(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.fc_4(x)
        x = self.softmax(x)
        return x
class mead_model(nn.Module):
    def __init__(self) -> None:
        super(mead_model, self).__init__()
        self.fc_1 = linear(2048, 512)
        self.BN = nn.BatchNorm1d(2048)
        self.BN1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        self.activation = nn.GELU()
       
        
        self.fc1 = nn.Linear(512, 512)
        self.BN2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)

        self.BN3 = nn.BatchNorm1d(512)
        self.projection_layer = LearnableProjection(512, 384)  

        self.fc_3 = linear(384, 256)
        self.fc_4 = linear(256, 8)
        self.BN5 = nn.BatchNorm1d(384)
        self.BN6 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.softmax = nn.Softmax(dim=1) 
    def forward(self,x):
        x = self.BN(x)
        x = self.fc_1(x)
        x = self.BN1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        x = self.fc1(x)
        x = self.BN2(x)
        x = self.activation(x)
        x = self.fc2(x)

        x = self.BN3(x)
        x = self.activation(x)  
        x = self.projection_layer(x)

        x = self.BN5(x)
        x = self.fc_3(x)
        x = self.BN6(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.fc_4(x)
        x = self.softmax(x)
        
        return x


class seed_model(nn.Module):
    def __init__(self) -> None:
        super(seed_model, self).__init__()
    
        self.fc_1 = linear(310, 512)
        self.BN = nn.BatchNorm1d(310)
        self.BN1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        self.activation = nn.GELU()
       
       
        self.fc1 = nn.Linear(512, 512)
        self.BN2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)

        self.BN3 = nn.BatchNorm1d(512)
        self.projection_layer = LearnableProjection(512, 384)  

        self.fc_3 = linear(384, 128)
        self.fc_4 = linear(128, 5)
        self.BN5 = nn.BatchNorm1d(384)
        self.BN6 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.softmax = nn.Softmax(dim=1) 
    def forward(self,x):
        x = self.BN(x)
        x = self.fc_1(x)
        x = self.BN1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        
        x = self.fc1(x)
        x = self.BN2(x)
        x = self.activation(x)
        x = self.fc2(x)

        x = self.BN3(x)
        x = self.activation(x)  
        x = self.projection_layer(x)

        x = self.BN5(x)
        x = self.fc_3(x)
        x = self.BN6(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.fc_4(x)
        x = self.softmax(x)
        return x
    

MODEL_DICT = {"mnist": MLP_MNIST, "cifar": MLP_CIFAR10}


def get_model(dataset, device):
    return MODEL_DICT[dataset]().to(device)

