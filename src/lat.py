import copy
import numpy as np
import pandas as pd
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from torch.nn.utils import weight_norm
from .algorithm_utils import Algorithm, PyTorchUtils, get_sub_seqs, get_train_data_loaders
from sklearn.decomposition import PCA
from tqdm import trange
import logging
import torch


class LAT(Algorithm, PyTorchUtils):
    def __init__(self, name: str='LAT', num_epochs: int=10, batch_size: int=32, lr: float=1e-3, sequence_length:
                 int=50, k: int=5, dropout: float=0.2, train_val_percentage: float=0.10, seed: int=None, gpu: int=None, details=False, patience: int=2,
                 stride: int=1, step: int=50, out_dir=None, pca_comp=None):
        Algorithm.__init__(self, __name__, name, seed, details=details, out_dir=out_dir)
        PyTorchUtils.__init__(self, seed, gpu)
        np.random.seed(seed)
        self.torch_save = True
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.patience = patience
        self.sequence_length = sequence_length
        self.k = k
        self.stride = stride
        self.train_val_percentage = train_val_percentage
        self.trans = None
        self.pca_comp = pca_comp
        self.step = step
        self.additional_params = dict()


    def fit(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        if self.pca_comp is not None:
            # Project input data on a limited number of principal components
            pca = PCA(n_components=self.pca_comp, svd_solver='full')
            pca.fit(data)
            self.additional_params["pca"] = pca
            data = pca.transform(data)
        sequences, res_seqs = get_sub_seqs(data, seq_len=self.sequence_length, stride=self.stride)
        train_loader, train_val_loader = get_train_data_loaders(sequences, batch_size=self.batch_size,
                                                                splits=[1 - self.train_val_percentage,
                                                                        self.train_val_percentage], seed=self.seed)
        self.trans = Trans(seed=self.seed, gpu=self.gpu, num_inputs=data.shape[1], feature_size=512, dropout=self.dropout, num_layers=1, heads=8)
        self.trans = fit(train_loader, train_val_loader, self.trans, patience=self.patience, num_epochs=self.num_epochs, lr=self.lr, step=self.step, k=self.k)

    def predict(self, X: pd.DataFrame) -> np.array:
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        if self.pca_comp is not None:
            data = self.additional_params["pca"].transform(data)
        sequences, res_seqs = get_sub_seqs(data, seq_len=self.sequence_length, stride=self.step)
        test_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=False, pin_memory=True,
                                 shuffle=False)
        reconstr_errors, outputs_array = predict_score(self.trans, test_loader, res_seqs, return_output=False, seq_len=self.sequence_length, step=self.step)
        if len(data) != len(reconstr_errors):
            multivar = (len(reconstr_errors.shape) > 1)
            if multivar:
                padding = np.zeros((len(data) - reconstr_errors.shape[0], reconstr_errors.shape[-1]))
            else:
                padding = np.zeros(len(data) - reconstr_errors.shape[0])
            reconstr_errors = np.concatenate([padding, reconstr_errors])

            print('padding', reconstr_errors.shape, padding.shape)
        predictions_dic = {'score_t': None,
                           'score_tc': None,
                           'error_t': None,
                           'error_tc': reconstr_errors,
                           'recons_tc': outputs_array,
                           }
        return predictions_dic

def my_kl_loss(p, q):
    res = p * (torch.log(p + 1/(p.shape[-1] * p.shape[-2])) - torch.log(q + 1/(q.shape[-1] * q.shape[-2])))
    return torch.mean(res, dim=0)

def fit(train_loader, val_loader, pytorch_module, patience, num_epochs, lr, step=1, k=5):
    pytorch_module = pytorch_module.cuda()
    optimizer = torch.optim.Adam(pytorch_module.parameters(), lr=lr)
    epoch_wo_improv = 0
    pytorch_module.train()
    train_loss_by_epoch = []
    val_loss_by_epoch = []
    best_val_loss = None
    best_params = pytorch_module.state_dict()
    # assuming first batch is complete
    for epoch in trange(num_epochs):
        if epoch_wo_improv < patience:
            logging.debug(f'Epoch {epoch + 1}/{num_epochs}.')
            pytorch_module.train()
            train_loss = []
            for n, ts_batch in enumerate(train_loader):
                ts_batch = ts_batch.float().cuda()
                predict, dim_corrs, ji_mats = pytorch_module(ts_batch)
                for i in range(len(dim_corrs)):
                    if i == 0:
                        loss1 = torch.mean(my_kl_loss(dim_corrs[i], ji_mats[i]))
                    else:
                        loss1 += torch.mean(my_kl_loss(dim_corrs[i], ji_mats[i]))
                loss1 = loss1 / (len(dim_corrs))
                loss2 = nn.MSELoss(reduction="mean")(predict[:, -step:, :], ts_batch[:, -step:, :])
                loss = -loss1 * k + loss2
                pytorch_module.zero_grad()
                loss.backward()
                optimizer.step()
                # multiplying by length of batch to correct accounting for incomplete batches
                train_loss.append(loss.item() * len(ts_batch[:, -step:, :]))
            train_loss = np.mean(train_loss) / train_loader.batch_size
            train_loss_by_epoch.append(train_loss)

            # Get Validation loss
            pytorch_module.eval()
            val_loss = []
            loss1_list = []
            loss2_list = []
            with torch.no_grad():
                for n, ts_batch in enumerate(val_loader):
                    ts_batch = ts_batch.float().cuda()
                    predict, dim_corrs, ji_mats = pytorch_module(ts_batch)
                    for i in range(len(dim_corrs)):
                        if i == 0:
                            loss1 = torch.mean(my_kl_loss(dim_corrs[i], ji_mats[i])) + torch.mean(
                                my_kl_loss(dim_corrs[i], ji_mats[i]))
                        else:
                            loss1 += torch.mean(my_kl_loss(dim_corrs[i], ji_mats[i])) + torch.mean(
                                my_kl_loss(dim_corrs[i], ji_mats[i]))
                    loss1 = loss1 / (len(dim_corrs))
                    loss2 = nn.MSELoss(reduction="mean")(predict[:, -step:, :], ts_batch[:, -step:, :])
                    loss = -loss1 * k + loss2
                    loss1_list.append(loss1.item() * len(ts_batch[:, -step:, :]))
                    loss2_list.append(loss2.item() * len(ts_batch[:, -step:, :]))
                    val_loss.append(loss.item() * len(ts_batch[:, -step:, :]))


            val_loss = np.mean(val_loss) / val_loader.batch_size
            val_loss_by_epoch.append(val_loss)
            loss1 = np.mean(loss1_list) / val_loader.batch_size
            loss2 = np.mean(loss2_list) / val_loader.batch_size

            print(f'{epoch}/{num_epochs}', f'dim_loss: {loss1}, re_loss: {loss2}, val_loss: {val_loss}')


            best_val_loss_epoch = np.argmin(val_loss_by_epoch)
            if best_val_loss_epoch == epoch:
                # any time a new best is encountered, the best_params will get replaced
                best_params = pytorch_module.state_dict()
            # Check for early stopping by counting the number of epochs since val loss improved
            if epoch > 0 and val_loss >= val_loss_by_epoch[-2]:
                epoch_wo_improv += 1
            else:
                epoch_wo_improv = 0
        else:
            # early stopping is applied
            pytorch_module.load_state_dict(best_params)
            break
    return pytorch_module

def predict_score(netG, test_loader, res_seqs, return_output=False, seq_len=100, step=1):
    netG.eval()
    reconstr_scores = []
    outputs_array = []
    with torch.no_grad():
        for ts_batch in test_loader:
            ts_batch = ts_batch.float().cuda()
            predict, dim_corrs, ji_mats = netG(ts_batch)
            for i in range(len(dim_corrs)):
                if i == 0:
                    metric = dim_corrs[i]
                else:
                    metric += dim_corrs[i]
            metric = torch.mean(torch.softmax((metric/len(dim_corrs)) * 50, dim=-1), dim=0)
            error = nn.L1Loss(reduction='none')(predict[:, -step:, :], ts_batch[:, -step:, :]) * metric
            reconstr_scores.append(error.cpu().numpy())
        if res_seqs is not None:
            pred = torch.zeros(res_seqs.shape[0], seq_len - res_seqs.shape[1], res_seqs.shape[2]).float().cuda()
            res_seqs = torch.from_numpy(res_seqs).float().cuda()
            pre = torch.cat([res_seqs, pred], dim=1).float().cuda()
            predict, dim_corrs, ji_mats = netG(pre)
            for i in range(len(dim_corrs)):
                if i == 0:
                    metric = dim_corrs[i]
                else:
                    metric += dim_corrs[i]
            metric = torch.mean(torch.softmax((metric/len(dim_corrs)) * 50, dim=-1), dim=0)
            error = nn.L1Loss(reduction='none')(predict, pre) * metric
            reconstr_scores.append(error[:, :res_seqs.shape[1], ].cpu().numpy())

    if res_seqs is not None:
        reconstr = np.concatenate(reconstr_scores[:-1])
        reconstr_scores = np.append(reconstr.reshape(-1, reconstr.shape[-1]), reconstr_scores[-1].reshape(-1, reconstr.shape[-1]), axis=0)
    else:
        reconstr_scores = np.concatenate(reconstr_scores)
        reconstr_scores = reconstr_scores.reshape(-1, reconstr_scores.shape[-1])
    if return_output:
        return_vars = (reconstr_scores, outputs_array)
    else:
        return_vars = (reconstr_scores, None)

    return return_vars

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Trans(nn.Module, PyTorchUtils):
    def __init__(self, seed:int, gpu:int, num_inputs, dropout=0.2, feature_size=512, num_layers=1, heads=8):
        super(Trans, self).__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.embedding = nn.Embedding(num_inputs, feature_size//heads)
        self.encoder = Trans_encoder(num_inputs, feature_size, num_layers, dropout, heads)

    def forward(self, x):
        x_out, dim_corrs, ji_mats = self.encoder(x)
        return x_out.permute(1, 0, 2), dim_corrs, ji_mats

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x)
        return x.permute(2, 0, 1)

class Cross_Correlation(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, kdim=None, vdim=None, scale=None):
        super(Cross_Correlation, self).__init__()
        self.dropout = nn.Dropout(dropout)

        kdim = kdim if kdim is not None else d_model
        vdim = vdim if vdim is not None else d_model

        self.drop = dropout
        self.q_proj = nn.Linear(d_model, kdim)
        self.k_proj = nn.Linear(d_model, kdim)
        self.v_proj = nn.Linear(d_model, vdim)
        self.out_proj = nn.Linear(2 * vdim, d_model)
        self.n_heads = nhead
        self.scale = scale or float(kdim//nhead) ** -0.5
        self.head_dim = kdim//nhead if kdim is not None else d_model//nhead
        self.embedding = nn.Embedding(self.head_dim, vdim)

    def forward(self, query, key, value, attn_mask, key_padding_mask=None):

        bsz = query.shape[1]
        tgt_len = query.shape[0]

        q = self.q_proj(query).contiguous().view(-1, bsz * self.n_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(key).contiguous().view(-1, bsz * self.n_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(value).contiguous().view(-1, bsz * self.n_heads, self.head_dim).transpose(0, 1)

        weights_arr = nn.Embedding(self.head_dim, self.n_heads * bsz).cuda()(torch.arange(self.head_dim).to(q.device))

        weights = weights_arr.view(self.head_dim, -1)

        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
        cos_ji_mat = cos_ji_mat / normed_mat
        ji_mat = torch.softmax(cos_ji_mat, dim=-1)

        mat = cos_ji_mat.detach().clone()
        ji_num = round(math.sqrt(self.head_dim)) * round(math.log(self.head_dim))
        topk_weights_ji = torch.softmax(torch.topk(mat, ji_num, dim=-1)[0], dim=-1)
        topk_indices_ji = torch.topk(mat, ji_num, dim=-1)[1]

        q_fft = torch.fft.rfft(q.permute(0, 2, 1).contiguous().unsqueeze(1).repeat(1, self.head_dim, 1, 1), dim=-1)
        k_fft = [torch.roll(k.permute(0, 2, 1).contiguous(), shifts=-i, dims=1).unsqueeze(1) for i in
                 range(self.head_dim)]
        k_fft = torch.cat(k_fft, dim=1)
        k_fft = torch.fft.rfft(k_fft, dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        cor = [torch.roll(corr.permute(0, 2, 1, 3).contiguous()[:, i, :, :].unsqueeze(1), i, -2) for i in range(self.head_dim)]
        cor = torch.cat(cor, dim=1)

        len_corr = [torch.sum(torch.index_select(cor[:, i].unsqueeze(1), -2, topk_indices_ji[i]) * topk_weights_ji[i].unsqueeze(1).repeat(1, tgt_len), dim=-2) for i in range(self.head_dim)]
        len_corr = torch.cat(len_corr, dim=1)
        len_corr = torch.mean(len_corr, dim=1)

        top_k = int(math.log(tgt_len))

        v = torch.cat([v, weights_arr.T.unsqueeze(1).repeat(1, tgt_len, 1)], dim=-1)
        weight = torch.topk(torch.mean(len_corr, dim=0), top_k, dim=-1)[0]
        index = torch.topk(torch.mean(len_corr, dim=0), top_k, dim=-1)[1]
        tmp_corr = torch.softmax(weight, dim=-1)
        intervals_agg = torch.zeros_like(v).float()
        dilation = 1
        for i in range(top_k):
            ind = int(index[i].cpu().numpy())
            if ind >= tgt_len//2:
                ind = tgt_len - ind
            if ind == 0 or ind == 1:
                intervals_agg = intervals_agg + tmp_corr[i] * v
            else:
                intervals_agg = intervals_agg + tmp_corr[i] * TemporalBlock(2 * self.head_dim, 2 * self.head_dim, ind, stride=1, dilation=dilation, padding=(ind-1) * dilation, dropout=self.drop).cuda()(v.permute(0, 2, 1)).permute(0, 2, 1)

        intervals_agg = intervals_agg.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        cor = cor.view(-1, self.n_heads, self.head_dim, self.head_dim, tgt_len)
        dim_cor = torch.sum(torch.index_select(torch.mean(torch.mean(cor, dim=0), dim=0), -1, index) * tmp_corr, dim=-1)
        dim_cor = torch.softmax(self.scale * dim_cor, dim=-1)

        return self.out_proj(intervals_agg), dim_cor, ji_mat

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, c_in, dim_feedforward=16, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = Cross_Correlation(d_model, nhead, dropout=dropout, kdim=c_in * nhead, vdim=c_in * nhead)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=dim_feedforward, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=dim_feedforward, out_channels=d_model, kernel_size=1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, dim_corr, ji_mat = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, dim_corr, ji_mat

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, encoder_layers, num_layers, norm=None):
        super(Encoder, self).__init__()
        self.layers = _get_clones(encoder_layers, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, x, src_mask=None):
        # x [B, L, D]
        dim_corrs = []
        ji_mats = []
        for layer in self.layers:
            x, dim_corr, ji_mat = layer(x, src_mask=src_mask)
            dim_corrs.append(dim_corr)
            ji_mats.append(ji_mat)
        if self.norm is not None:
            x = self.norm(x)
        return x, dim_corrs, ji_mats

class Trans_encoder(nn.Module):
    def __init__(self, num_inputs, feature_size=512, num_layers=1, dropout=0.1, heads=8):
        super(Trans_encoder, self).__init__()

        self.src_mask = None
        self.embedding = TokenEmbedding(c_in=num_inputs, d_model=feature_size)
        self.pos_encoder = PositionalEncoding(feature_size, dropout=0.1)
        self.encoder_layer = EncoderLayer(d_model=feature_size, nhead=heads, c_in=num_inputs, dropout=dropout)
        self.transformer_encoder = Encoder(self.encoder_layer, num_layers=num_layers, norm=torch.nn.LayerNorm(feature_size))
        self.decoder = nn.Linear(feature_size, num_inputs)

    def forward(self, src):
        src = src.permute(0, 2, 1)
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output, dim_corrs, ji_mats = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = self.decoder(output)
        return output, dim_corrs, ji_mats