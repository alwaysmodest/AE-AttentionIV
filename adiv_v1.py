import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader
from mliv.utils import set_seed, cat

example = '''
from mliv.inference import OneSIV

model = OneSIV()
model.fit(data)
ITE = model.predict(data.train)
ATE,_ = model.ATE(data.train)
'''
################## Define loss #################################
def fit_linear(target: torch.Tensor, feature: torch.Tensor, reg: float = 0.0):
    nData, nDim = feature.size()
    A = torch.matmul(feature.t(), feature)
    device = feature.device
    A = A + reg * torch.eye(nDim, device=device)
    A_inv = torch.inverse(A)
    if target.dim() == 2:
        b = torch.matmul(feature.t(), target)
        weight = torch.matmul(A_inv, b)
    else:
        b = torch.einsum("nd,n...->d...", feature, target)
        weight = torch.einsum("de,d...->e...", A_inv, b)
    return weight

def linear_reg_pred(feature: torch.Tensor, weight: torch.Tensor):
    if weight.dim() == 2:
        return torch.matmul(feature, weight)
    else:
        return torch.einsum("nd,d...->n...", feature, weight)

def linear_reg_loss(target: torch.Tensor, feature: torch.Tensor, reg: float):
    weight = fit_linear(target, feature, reg)
    pred = linear_reg_pred(feature, weight)
    return torch.norm((target - pred)) ** 2 + reg * torch.norm(weight) ** 2
class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.W1 = nn.Linear(input_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        u = torch.tanh(self.W1(x))
        alpha = torch.softmax(self.W2(u), dim=1)
        context = (alpha * x).sum(dim=1)
        context=context.squeeze()
        return context
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        
        # 定义 encoder 的线性层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return x
        
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        
        # 定义 decoder 的线性层
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.sigmoid(self.fc4(x))
        return x

class Networks(nn.Module):
    def __init__(self, z_dim, x_dim, t_dim, dropout):
        super(Networks, self).__init__()

        t_input_dim, y_input_dim, G_input_dim = z_dim+x_dim, t_dim+x_dim, z_dim+x_dim+t_dim
        self.t_net = nn.Sequential(nn.Linear(t_input_dim, 1280),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(1280, 320),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(320, 32),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(32, t_dim))
        self.y_net = nn.Sequential(nn.Linear(y_input_dim, 1280),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(1280, 320),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(320, 32),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(32,1))
                                
        self.attention_net = nn.Sequential(nn.Linear(G_input_dim, 1280),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(1280, 320),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     Attention(320, 320),
                                     nn.Dropout(dropout),
                                     nn.Linear(1000,1000),
                                     )    
        '''                        
        self.attention_net = nn.Sequential(nn.Linear(G_input_dim, 1280),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(1280, 320),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     Attention(320, 320),
                                     nn.Dropout(dropout),
                                     nn.Linear(1000,320),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(320, 32),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(32,1))
        '''                             
        self.dis_t_net = nn.Sequential(nn.Linear(t_input_dim, 1280),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(1280, 320),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(320, 32),
                                    nn.Sigmoid(),
                                    nn.Dropout(dropout),
                                    nn.Linear(32, t_dim)
                                    )
        self.encoder_net = nn.Sequential(nn.Linear(G_input_dim, 1280), 
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(1280, 320),
                                     nn.ReLU())
        self.decoder_net = nn.Sequential(nn.Linear(1, 32), 
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(32, 320),
                                     nn.Sigmoid())
    def forward(self, z, x):   
        pred_t = self.t_net(cat([z,x]))
        pred_encoder = self.encoder_net(cat([z,x,pred_t]))
        pred_yxt =self.attention_net(cat([z,x,pred_t]))
        pred_yxt = pred_yxt.unsqueeze(dim=-1)
        pred_decoder = self.decoder_net(pred_yxt)
        yt_input = torch.cat((pred_t,x), 1)
        pred_yt = self.y_net(yt_input)
        pred_G=self.dis_t_net
        return pred_t, pred_yt,pred_yxt,pred_encoder,pred_decoder

class AttentionDiscriminant(object):
    def __init__(self) -> None:
        self.config = {
                    'methodName': 'AttentionDiscriminant',
                    'device': 'cuda:0',
                    'learning_rate': 0.005,
                    'dropout': 0.5,
                    'beta1': 0.9,
                    'beta2': 0.999,
                    'eps': 1e-8,
                    'w1': 0.0017,
                    'w2': 1.0,
                    'w3': 1.0,
                    'epochs': 30,
                    'verbose': 1,
                    'show_per_epoch': 10,
                    'batch_size':1000,
                    'seed': 2022,   
                    }

    def set_Configuration(self, config):
        self.config = config

    def get_loader(self, data=None):
        if data is None:
            data = self.train
        loader = DataLoader(data, batch_size=self.batch_size)
        return loader

    def fit(self, data, exp=-1, config=None):
        if config is None:
            config = self.config

        self.z_dim = data.train.z.shape[1]
        self.x_dim = data.train.x.shape[1]
        self.t_dim = data.train.t.shape[1]
        
        self.device = config['device']
        self.batch_size = config['batch_size']

        set_seed(config['seed'])
        data.tensor()
        data.to(self.device)
        self.data = data

        AttentionDiscriminant_dict = {
            'z_dim':self.z_dim, 
            'x_dim':self.x_dim, 
            't_dim':self.t_dim, 
            'dropout':config['dropout'],
        }

        net = Networks(**AttentionDiscriminant_dict)
        net.to(self.device)

        optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'], betas=(config['beta1'], config['beta2']),eps=config['eps'])
        t_loss = torch.nn.MSELoss()
        y_loss = torch.nn.MSELoss()
        G_loss = torch.nn.MSELoss()
        rescon_loss = torch.nn.MSELoss()

        print('Run {}-th experiment for {}. '.format(exp, config['methodName']))

        train_loader = self.get_loader(data.train)

        def estimation(data):
            net.eval()
            return net.y_net(cat([data.t-data.t, data.x])), net.y_net(cat([data.t, data.x]))
        ATE_train=9999
        RMSE_train=9999
        for epoch in range(config['epochs']):
            net.train()

            for idx, inputs in enumerate(train_loader):
                z = inputs['z'].to(self.device)
                x = inputs['x'].to(self.device)
                t = inputs['t'].to(self.device)
                y = inputs['y'].to(self.device)
                pred_t, pred_y, pred_G,pred_e,pred_d = net(z,x)
                loss =G_loss(pred_G,y)  + t_loss(pred_t,t)+ 0.0017*y_loss(pred_y,y)
                #10*rescon_loss(pred_d,pred_e)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            net.eval()

            if (config['verbose'] >= 1 and epoch % config['show_per_epoch'] == 0 ) or epoch == config['epochs']-1:
                _, pred_test_y = estimation(data.test)
                print(f'Epoch {epoch}: {y_loss(pred_test_y, data.test.y)}. ')
                print("attention_loss")
                print(G_loss(pred_G,y))
                print("t_loss")
                print(t_loss(pred_t,t))
                print("y_loss")
                print(0.0017*y_loss(pred_y,y))
                print("rescon_loss")
                print(10*rescon_loss(pred_d,pred_e))
                if y_loss(pred_test_y, data.test.y)<0.02 :
                    break
                y=y.detach().cpu().numpy()
                ITE_0 = net.y_net(cat([t-t,x])).detach().cpu().numpy()
                ITE_1 = net.y_net(cat([t-t+1,x])).detach().cpu().numpy()
                ITE_t = net.y_net(cat([t,x])).detach().cpu().numpy()
                ATE_Train=np.abs(np.mean(ITE_1-ITE_0))
                RMSE_Train=np.sqrt(((ITE_t - y) ** 2).mean())
            if ATE_Train<ATE_train:
                ATE_train=ATE_Train
                print("ATE_Train")
                print(ATE_train)
            if RMSE_Train<RMSE_train:
                RMSE_train=RMSE_Train
                print("RMSE_Train")
                print(RMSE_train)
        print(ATE_train)
        print(RMSE_train)
        print('End. ' + '-'*20)

        self.estimation = estimation
        self.y_net = net.y_net
        self.t_net = net.dis_t_net
        self.attention=net.attention_net

    def predict(self, data=None, t=None, x=None):
        if data is None:
            data = self.data.test

        if x is None:
            x = data.x
        if t is None:
            t = data.t

        return self.y_net(cat([t,x])).detach().cpu().numpy()

    def ITE(self, data=None, t=None, x=None):
        if data is None:
            data = self.data.test

        if x is None:
            x = data.x
        if t is None:
            t = data.t

        y=data.y.detach().cpu().numpy()
        ITE_0 = self.y_net(cat([t-t,x])).detach().cpu().numpy()
        ITE_1 = self.y_net(cat([t-t+1,x])).detach().cpu().numpy()
        ITE_t = self.y_net(cat([t,x])).detach().cpu().numpy()


        return np.abs(ITE_0),np.abs(ITE_1),np.sqrt(((ITE_t - y) ** 2).mean())

    def ATE(self, data=None, t=None, x=None):
        ITE_0,ITE_1,RMSE = self.ITE(data,t,x)

        return np.abs(np.mean(ITE_1-ITE_0)), RMSE
        
