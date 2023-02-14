import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import VGAE
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np

# dataset=QM9("/home/nesara/Project_CLESS/Dummy/MolCLR/data_pygeo", transform=T.NormalizeFeatures())
train_dataset = QM9("/home/nesara/Project_CLESS/Dummy/MolCLR/data_pygeo/data_pygeo_QM9", transform=T.NormalizeFeatures())[:70000]
test_dataset= QM9("/home/nesara/Project_CLESS/Dummy/MolCLR/data_pygeo/data_pygeo_QM9", transform=T.NormalizeFeatures())[70000:100000]
# dataset.data.train_mask = dataset.data.val_mask =dataset.data.test_mask = None
# data = dataset[0]
# data.train_mask = data.val_mask = data.test_mask = None
# data
# train_dataset, test_dataset = train_test_split(dataset)
# data = train_test_split_edges(dataset.data)
# data = train_test_split_edges(dataset.data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True) # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
    
     
from torch_geometric.nn import VGAE
class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
    
    
out_channels = 20
epochs = 30


model = VGAE(VariationalGCNEncoder(in_channels= train_dataset[0].x.shape[1], out_channels=out_channels))  # new line

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def run_one_epoch(data_loader, type, epoch):
    # Store per batch loss and accuracy 
    all_losses = []
    all_kldivs = []

    # Iterate over data loader
    for _, batch in enumerate(tqdm(data_loader)):
        # Some of the data points have invalid adjacency matrices 
        try:
            # Use GPU
            batch.to(device)  
            # Reset gradients
            optimizer.zero_grad() 
            # Call model
            # print(batch.edge_attr, 'See Kothi')
            # print(batch.x, 'Node Features')
            # print(batch.x.float().shape, batch.edge_attr.float().shape,batch.edge_index.shape, batch.batch.shape, 'Batch Feature Shapes')
            # break
            z = model.encode(batch.x.float(), batch.edge_index)
            loss = model.recon_loss(z, batch.edge_index)
            loss = loss + (1 / batch.num_nodes) * model.kl_loss()  # new line
            loss.backward()
            optimizer.step()
            # Store loss and metrics
            all_losses.append(loss.detach().cpu().numpy())
            #all_accs.append(acc)
            all_kldivs.append(model.kl_loss().data.numpy())
            
        except IndexError as error:
            # For a few graphs the edge information is not correct
            # Simply skip the batch containing those
            print("Error: ", error)
    all_losses=np.average(all_losses)
    all_kldivs=np.average(all_kldivs)
    return all_losses, all_kldivs
   



# def train():
#     model.train()
#     optimizer.zero_grad()
#     z = model.encode(x, train_pos_edge_index)
#     loss = model.recon_loss(z, train_pos_edge_index)
    
#     loss = loss + (1 / data_train.num_nodes) * model.kl_loss()  # new line
#     loss.backward()
#     optimizer.step()
#     return float(loss)


# def test(pos_edge_index, neg_edge_index):
#     model.eval()
#     with torch.no_grad():
#         z = model.encode(x, train_pos_edge_index)
#     return model.test(z, pos_edge_index, neg_edge_index)

writer = SummaryWriter('runs/VGAE_experiment_'+'2d_100_epochs')

for epoch in range(1, epochs + 1):
    loss, kl_loss = run_one_epoch(train_loader, type="Train", epoch=epoch)
    # auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    # print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
    writer.add_scalar('loss train',loss,epoch)
    writer.add_scalar('KL loss train',kl_loss,epoch)
    # writer.add_scalar('auc train',auc,epoch) # new line
    # writer.add_scalar('ap train',ap,epoch)   # new line
writer.close()