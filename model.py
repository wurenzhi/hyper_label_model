import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import coo_matrix
import torch.optim as optim

from loss import BCEMask, BCEMaskWeighted

def sparse_mean(index, value, expand=True):
    """obtain the mean of values that share the same index (e.g. same row or column in label matrix X) 

    Args:
        index: The indices of the elements, the first dimension is batch size 
        value : The values of the elements, the first dimension is batch size
        expand (bool, optional): if true expand the output to have the same size as index

    Returns:
        mean values
    """
    output_batch = []
    ind_max = int(index.max() + 1)
    for i_batch in range(value.shape[0]):
        output = torch.zeros((ind_max, value.shape[2])).float().to(value.device).index_add_(0,
                                                                                    index[i_batch],
                                                                                    value[i_batch])
        norm = torch.zeros(ind_max).to(value.device).float().index_add_(
            0, index[i_batch], torch.ones_like(index[i_batch]).float()) + 1e-9
        output = output / norm[:, None].float()
        if expand:
            output = torch.index_select(output, 0, index[i_batch])
        output_batch.append(output)
    return torch.stack(output_batch)


class GNNLayer(nn.Module):
    """One GNN layer
    Args:
        in_features: embedding dimension of the input
        out_features: embedding dimension of the output
    """
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.activation = nn.LeakyReLU()
        self.linear = nn.Linear(in_features * 4, out_features) #this corresponds to fk in Equation 5
        self.linear_row = nn.Linear(in_features, in_features) #this corresponds to W1 in Equation 5
        self.linear_col = nn.Linear(in_features, in_features) #this corresponds to W2 in Equation 5
        self.linear_global = nn.Linear(in_features, in_features) #this corresponds to W3 in Equation 5
        self.linear_self = nn.Linear(in_features, in_features) #this corresponds to W4 in Equation 5


    def forward(self, index, value):
        ## pool over values in the same column, same row, and the whole matrix
        pooled = [self.linear_row(sparse_mean(index[:, :, 0], value)),
                  self.linear_col(sparse_mean(index[:, :, 1], value)), 
                  self.linear_global(torch.mean(value, dim=1)).unsqueeze(1).expand_as(value)]
        
        stacked = torch.cat(
            [self.linear_self(value)] + pooled, dim=2)
        return index, self.activation(self.linear(stacked))

class SequentialMultiArg(nn.Sequential):
    """helper class to stack multiple GNNLayer
    """
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class LELAGNN(nn.Module):
    """The model architecture of LELA
    """
    def __init__(self):
        super(LELAGNN, self).__init__()
        #the multi-layer GNN network
        self.matrix_net = SequentialMultiArg(
            GNNLayer(1, 8),
            GNNLayer(8, 8),
            GNNLayer(8, 8),
            GNNLayer(8, 32),
        )
        col_embed_mixed_size = 32
        #MLP network
        self.classify = nn.Sequential(
            nn.Linear(col_embed_mixed_size, col_embed_mixed_size),
            nn.LeakyReLU(),
            nn.Linear(col_embed_mixed_size, col_embed_mixed_size),
            nn.LeakyReLU(),
            nn.Linear(col_embed_mixed_size, 1),
            nn.Sigmoid()
        )

    def forward(self, index, value):
        _, elementwise_embed_sparse = self.matrix_net(index, value.float().unsqueeze(2)) # encode matrix with GNNs
        example_embed = sparse_mean(
            index[:, :, 0], elementwise_embed_sparse, expand=False)# pool over elements in the same row to obtain embedding for each example
        mask = torch.sum(example_embed, dim=2) > 0 
        # mask examples where all LFs abstains, 
        # these examples have all zeros in the ebmeddings. 
        # mask will be used in loss function to skip these examples
        output = self.classify(example_embed)
        return torch.squeeze(output), mask



class LELAWrapper:
    """Wrapper for the trained lela model
    """
    def __init__(self, checkpoint_path): #initialize from a trained model checkpoint
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.net = LELAGNN()
        self.net.to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()
        self.name = checkpoint_path

    def predict(self,X):
        """predict hard labels for each data point

        Args:
            X: the label matrix, a nxm array where n is the number of data points and m is the number of LFs
            for each element in X, -1 denotes abstaintion and other numbers (e.g. 0, 1,2) denote each classes

        Returns:
            y: hard labels, a numpy array of size (n,)
        """
        probs = self.predict_prob(X)
        pred = np.argmax(probs, axis=1).flatten()
        return pred

    def predict_prob(self, X):
        """predict probabilities of each data point being each class

        Args:
            X: the label matrix, a nxm array where n is the number of data points and m is the number of LFs
            for each element in X, -1 denotes abstaintion and other numbers (e.g. 0, 1,2) denote the classes
        Returns:
            y: probabilities of each data point being each class, a nxk matrix where k is the number of classes
        """
        X_arr = np.array(X)
        preds = []
        n_class = int(np.max(X_arr)+1)
        if n_class == 2:
            mat = -1*np.ones_like(X_arr)
            mat[X_arr == 1] = 1
            mat[X_arr == -1] = 0 #note -1 denotes abstention in X and X_arr, but 0 denotes abstention in mat
            mat = np.array(mat)
            X_sparse = coo_matrix(np.squeeze(mat))
            index = np.array([X_sparse.row, X_sparse.col]).T
            value = X_sparse.data
            index = torch.from_numpy(index).to(self.device)
            value = torch.from_numpy(value).float().to(self.device)
            pred, _ = self.net(index.unsqueeze(0), value.unsqueeze(0))
            preds = torch.stack([1-pred, pred], dim=1)
        else:
            for label in range(n_class):
                mat = -1*np.ones_like(X_arr)
                mat[X_arr == label] = 1
                mat[X_arr == -1] = 0 #note -1 denotes abstention in X and X_arr, but 0 denotes abstention in mat
                mat = np.array(mat)
                X_sparse = coo_matrix(np.squeeze(mat))
                index = np.array([X_sparse.row, X_sparse.col]).T
                value = X_sparse.data
                index = torch.from_numpy(index).to(self.device)
                value = torch.from_numpy(value).float().to(self.device)
                pred, _ = self.net(index.unsqueeze(0), value.unsqueeze(0))
                preds.append(pred.unsqueeze(1))
            preds = torch.cat(preds, dim=1)
            preds = preds/torch.sum(preds, dim=1).unsqueeze(1)
        preds = preds.detach().cpu().numpy()

        ## scale the probs the have a maximum of 1 and minimum of 0
        tie_score = 1/n_class
        preds[preds>tie_score] = (preds[preds>tie_score]-tie_score)*tie_score/(np.max(preds[preds>tie_score])-tie_score)+tie_score
        preds[preds<tie_score] = (preds[preds<tie_score]-tie_score)*tie_score/(tie_score-np.min(preds[preds<tie_score]))+tie_score
        preds[preds>1]=1
        preds[preds<0]=0
        preds=preds/np.sum(preds,axis=1,keepdims=True)
        return preds


def LELASemi(LF_mat, y_partial, y_indices, checkpoint_path):
    """function to perform semisupervised label aggregation using LELA

    Args:
        LF_mat: label matrix, note "-1" denotes abstentions
        y_partial: provided labels, e.g [1, 0, 1, 1]. 
        y_indices: the corresponding indices of the provided labels, e.g. [100, 198, 222, 4213]

    Returns:
        predicted labels
    """
    preds = []
    n_class = int(np.max(y_partial)+1)
    if n_class == 2:
        lela = LELASemisupervisedHelper(checkpoint_path)
        X = np.zeros_like(LF_mat)-1
        X[LF_mat==1] = 1
        X[LF_mat==-1] = 0
        pos_ratio = np.sum(y_partial)/len(y_partial)
        weights = [pos_ratio, 1-pos_ratio]
        pred = lela.fit_predict(X, y_partial, y_indices, weights)
        preds = [1-pred.unsqueeze(1), pred.unsqueeze(1)] 
    else:
        for label in range(n_class):
            lela = LELASemisupervisedHelper(checkpoint_path)
            X = np.zeros_like(LF_mat)-1
            X[LF_mat==label] = 1
            X[LF_mat==-1] = 0
            y_binary = np.zeros_like(y_partial)
            y_binary[y_partial==label] = 1
            pred = lela.fit_predict(X, y_binary, y_indices)
            preds.append(pred.unsqueeze(1))
    preds = torch.cat(preds, dim=1)
    preds = preds/torch.sum(preds, dim=1).unsqueeze(1)
    return preds

class LELASemisupervisedHelper:
    """helper class to perform semisupervised label aggregation
    """
    def __init__(self, checkpoint_path):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.net = LELAGNN()
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=0.0001,
            amsgrad=True,
        )
        self.checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device(self.device)
        )

    def initialize_net(self):
        """initialize model as pretrained LELA
        """
        self.net.load_state_dict(self.checkpoint['model_state_dict'])
        self.net.to(self.device)
        self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        self.optimizer.zero_grad()

    def fit_predict(self,X, y_partial, y_indices,weights=None):
        """finetune LELA model by miminizing the loss on the provided labels
        """
        self.initialize_net()
        self.net.train()
        X_sparse = coo_matrix(np.squeeze(X))
        index = np.array([X_sparse.row, X_sparse.col]).T
        value = X_sparse.data
        index = torch.from_numpy(index).to(self.device).unsqueeze(0)
        value = torch.from_numpy(value).float().to(self.device).unsqueeze(0)
        y_complete = np.zeros(X.shape[0])
        y_complete[y_indices] = y_partial
        y_complete = torch.from_numpy(y_complete).float().to(self.device).unsqueeze(0)
        if weights:
            self.criterion = BCEMaskWeighted(weights)
        else:
            self.criterion = BCEMask()
        mask = np.zeros(X.shape[0])
        mask[y_indices] = 1
        mask = mask.astype(bool)
        mask = torch.from_numpy(mask).unsqueeze(0)
        for i in range(int(np.sqrt(len(y_partial)))):
            pred, _ = self.net(index, value)
            loss = self.criterion(pred.unsqueeze(0), y_complete, mask)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.net.eval()
        pred, _ = self.net(index, value)
        return pred


if __name__ == "__main__":
    lela = LELAWrapper("lela_checkpoint.pt")
    X = np.array([[1, 1, 0],
                [0,0,1],
                [1, 1, 1],
                [-1, -1, 0],
                [0,1, 0]])
    y = lela.predict_prob(X)
    print(y)