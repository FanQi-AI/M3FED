
import torch
import numpy as np



def  dirichlet_split_noniid(train_labels, alpha, n_clients):
    
    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
  

    class_idcs = [np.argwhere(train_labels==y).flatten() 
           for y in range(n_classes)]
  
 
    client_idcs = [[] for _ in range(n_clients)]
   
    for c, fracs in zip(class_idcs, label_distribution):
       
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
  
    return client_idcs

if __name__ == "__main__":
    torch.manual_seed(2024)
    np.random.seed(2024)

    #mead
    data=torch.load('/data/MEAD/mead_2048.pt',map_location='cpu')
    savepath='/data/MEAD/mead_label.pt'
    label=[]

    for i in range(len(data)):
        label.append(data[i][1])

    torch.save(label,savepath)
    print()
    a_label=np.array(torch.load('/data/MEAD/mead_label.pt',map_location='cpu'))
    a_data_splits=dirichlet_split_noniid(a_label,alpha=1,n_clients=10)
    id=10
    for i in range(len(a_data_splits)):
        torch.save(a_data_splits[i],'/data/data_split/client_'+str(id)+'.pt')
        id +=1
    