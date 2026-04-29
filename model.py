import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class net(nn.Module):
    def __init__(self, args):
        super(net, self).__init__()
        self.args = args

        self.tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        self.bert_shared = BertModel.from_pretrained(args.bert_path)

        hidden_size = 768

        # ===== 超参数 =====
        self.balance_mode = getattr(args, "balance_mode", "coral")   # "mmd" or "coral"

        # ---------------- 分解 ----------------
        self.shared_proj = nn.Linear(hidden_size, hidden_size)
        self.gate_layer = nn.Linear(hidden_size, hidden_size * 3)

        self.REP_I = nn.Sequential(*self.rep_layer(input_dims=hidden_size, out_dims=hidden_size, layer=2))
        self.REP_C = nn.Sequential(*self.rep_layer(input_dims=hidden_size, out_dims=hidden_size, layer=2))
        self.REP_A = nn.Sequential(*self.rep_layer(input_dims=hidden_size, out_dims=hidden_size, layer=2))
        self.map_t = nn.Sequential(*self.rep_layer(input_dims=1, out_dims=64, layer=2))

        # ---------------- 输出头 ----------------
        self.t_regress_c = nn.Sequential(*self.output_layer(input_dims=hidden_size, out_dims=2, layer=4))
        self.t_regress_i = nn.Sequential(*self.output_layer(input_dims=hidden_size, out_dims=2, layer=4))
        self.y_regress = nn.Sequential(*self.output_layer(input_dims=hidden_size + hidden_size + 64, out_dims=2, layer=4))
        self.y_regress_a = nn.Sequential(*self.output_layer(input_dims=hidden_size, out_dims=2, layer=4))
        self.y_regress_c = nn.Sequential(*self.output_layer(input_dims=hidden_size, out_dims=2, layer=4))

    def forward(self, input_data):
        text_emb = self.bert(input_data['text_a'], input_data['text_b'])
        rep_t = self.map_t(input_data['xiaolei'].float().unsqueeze(1))

        shared, gate, rep_c, rep_a, rep_i, recon = self.decompose_with_gate(text_emb)
        y_input = torch.cat((rep_c, rep_a, rep_t), dim=1)

        pred_y = self.y_regress(y_input)[:, 1]
        a_pred_y = self.y_regress_a(rep_a)[:, 1]
        c_pred_y = self.y_regress_c(rep_c)[:, 1]
        c_pred_t = self.t_regress_c(rep_c)[:, 1]
        i_pred_t = self.t_regress_i(rep_i)[:, 1]

        output_data = {
            'text_emb': text_emb,
            'shared': shared,
            'gate': gate,
            'recon': recon,
            'rep_i': rep_i,
            'rep_c': rep_c,
            'rep_a': rep_a,
            'pred_y': pred_y,
            'a_pred_y': a_pred_y,
            'c_pred_y': c_pred_y,
            'c_pred_t': c_pred_t,
            'i_pred_t': i_pred_t,
        }

        self.output = output_data
        self.input = input_data

        return output_data

    def bert(self, text_a, text_b):
        encode_text = self.tokenizer(
            text_a,
            text_b,
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128
        )
        encoded_dict = encode_text.to(self.args.device)
        outputs = self.bert_shared(**encoded_dict)
        result = outputs.last_hidden_state[:, 0, :]
        return result

    def decompose_with_gate(self, text_emb):
        shared = self.shared_proj(text_emb)  # [B, H]

        B, H = shared.shape
        gates_logits = self.gate_layer(shared).view(B, H, 3)  # [B, H, 3]
        gates = torch.softmax(gates_logits, dim=-1)  # [B, H, 3]

        w_c = gates[:, :, 0]
        w_a = gates[:, :, 1]
        w_i = gates[:, :, 2]

        rep_c_base = w_c * shared
        rep_a_base = w_a * shared
        rep_i_base = w_i * shared

        rep_c = self.REP_C(rep_c_base)
        rep_a = self.REP_A(rep_a_base)
        rep_i = self.REP_I(rep_i_base)

        recon = rep_c + rep_a + rep_i

        return shared, gates, rep_c, rep_a, rep_i, recon

    def loss_func(self):
        out = self.output
        inp = self.input

        label = inp['label'].float()
        t = inp['xiaolei'].float()

        # ===== treatment 组逆频率重加权 =====
        sample_weight = self.compute_treatment_reweight(t)  # [B]

        # ===== outcome losses =====
        loss_y = self.weighted_bce(out['pred_y'], label, sample_weight)
        a_loss_y = self.weighted_bce(out['a_pred_y'], label, sample_weight)
        c_loss_y = self.weighted_bce(out['c_pred_y'], label, sample_weight)

        # ===== treatment losses =====
        i_loss_t = self.weighted_bce(out['i_pred_t'], t, sample_weight=None)
        c_loss_t = self.weighted_bce(out['c_pred_t'], t, sample_weight=None)

        # ===== reconstruction =====
        loss_recon = F.mse_loss(out['recon'], out['shared'])

        # ===== orthogonality =====
        loss_orth = (
            self.orthogonal_feature_loss(out['rep_c'], out['rep_a']) +
            self.orthogonal_feature_loss(out['rep_c'], out['rep_i']) +
            self.orthogonal_feature_loss(out['rep_a'], out['rep_i'])
        ) / 3.0

        # ===== rep_c balancing loss =====
        if self.balance_mode.lower() == "mmd":
            loss_balance_c = self.mmd_loss(out['rep_c'], t)
        elif self.balance_mode.lower() == "coral":
            loss_balance_c = self.coral_loss(out['rep_c'], t)
        else:
            raise ValueError("balance_mode must be 'mmd' or 'coral'")

        if self.input['tr']:
            loss = (
                loss_y
                + a_loss_y
                + c_loss_y
                + i_loss_t
                + c_loss_t
                + loss_recon
                + loss_orth
                + loss_balance_c
            )
        else:
            loss_y = self.weighted_bce(out['pred_y'], label, sample_weight=None)
            loss = loss_y

        logs = {
            'loss_y': loss_y.detach(),
            'a_loss_y': a_loss_y.detach(),
            'c_loss_y': c_loss_y.detach(),
            'i_loss_t': i_loss_t.detach(),
            'c_loss_t': c_loss_t.detach(),
            'loss_recon': loss_recon.detach(),
            'loss_orth': loss_orth.detach(),
            'loss_balance_c': loss_balance_c.detach(),
            'weight_mean': sample_weight.mean().detach(),
            'loss_total': loss.detach(),
        }

        return loss, logs

    def rep_layer(self, input_dims, out_dims, layer):
        dim = np.around(np.linspace(input_dims, out_dims, layer + 1)).astype(int)
        layers = []
        for i in range(layer):
            layers.append(nn.Linear(dim[i], dim[i + 1]))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=self.args.dropout))
        return layers

    def output_layer(self, input_dims, out_dims, layer):
        dim = np.around(np.linspace(input_dims, out_dims, layer + 1)).astype(int)
        layers = []
        for i in range(layer):
            layers.append(nn.Linear(dim[i], dim[i + 1]))
            if i < layer - 1:
                layers.append(nn.Tanh())
                layers.append(nn.Dropout(p=self.args.dropout))
        layers.append(nn.Softmax(dim=1))
        return layers

    # ===== weighted BCE =====
    def weighted_bce(self, pred, target, sample_weight=None):
        loss = F.binary_cross_entropy(pred, target, reduction='none')
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()

    def compute_treatment_reweight(self, t):
        """
        treatment 组逆频率重加权
        """
        t = t.view(-1)
        device = t.device

        n = t.size(0)
        n1 = (t >= 0.5).float().sum()
        n0 = (t < 0.5).float().sum()

        n1 = torch.clamp(n1, min=1.0)
        n0 = torch.clamp(n0, min=1.0)

        w1 = n / (2.0 * n1)
        w0 = n / (2.0 * n0)

        sample_weight = torch.where(t >= 0.5, w1, w0).to(device)

        sample_weight = 1.0 + (sample_weight - 1.0)
        sample_weight = sample_weight / (sample_weight.mean().detach() + 1e-8)

        return sample_weight

    # ===== MMD balancing =====
    def mmd_loss(self, rep, t, kernel_mul=2.0, kernel_num=5):
        t = t.view(-1)
        rep0 = rep[t < 0.5]
        rep1 = rep[t >= 0.5]

        if rep0.size(0) < 2 or rep1.size(0) < 2:
            return torch.tensor(0.0, device=rep.device)

        total = torch.cat([rep0, rep1], dim=0)
        total0 = total.unsqueeze(0)
        total1 = total.unsqueeze(1)
        L2_distance = ((total0 - total1) ** 2).sum(2)

        bandwidth = torch.sum(L2_distance.detach()) / (total.size(0) ** 2 - total.size(0) + 1e-8)
        bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / (bw + 1e-8)) for bw in bandwidth_list]
        kernels = sum(kernel_val)

        n0 = rep0.size(0)
        XX = kernels[:n0, :n0]
        YY = kernels[n0:, n0:]
        XY = kernels[:n0, n0:]
        YX = kernels[n0:, :n0]

        loss = XX.mean() + YY.mean() - XY.mean() - YX.mean()
        return loss

    # ===== CORAL balancing =====
    def coral_loss(self, rep, t):
        t = t.view(-1)
        rep0 = rep[t < 0.5]
        rep1 = rep[t >= 0.5]

        if rep0.size(0) < 2 or rep1.size(0) < 2:
            return torch.tensor(0.0, device=rep.device)

        mean0 = rep0.mean(dim=0, keepdim=True)
        mean1 = rep1.mean(dim=0, keepdim=True)

        xc0 = rep0 - mean0
        xc1 = rep1 - mean1

        cov0 = torch.matmul(xc0.t(), xc0) / (rep0.size(0) - 1 + 1e-8)
        cov1 = torch.matmul(xc1.t(), xc1) / (rep1.size(0) - 1 + 1e-8)

        mean_loss = ((mean0 - mean1) ** 2).mean()
        cov_loss = ((cov0 - cov1) ** 2).mean()

        return mean_loss + cov_loss

    def orthogonal_feature_loss(self, rep_c, rep_a):
        rc = rep_c - rep_c.mean(dim=0, keepdim=True)
        ra = rep_a - rep_a.mean(dim=0, keepdim=True)

        B = rc.size(0)
        cov = torch.matmul(rc.t(), ra) / (B + 1e-8)

        loss_orth = (cov ** 2).mean()
        return loss_orth
