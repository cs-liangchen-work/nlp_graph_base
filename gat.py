import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        # 参数说明：
        # in_features：  输入的维度
        # out_features：  输出的维度
        # dropout：  dropout的比率。
        # alpha：  GAT中使用的激活函数：LeakyReLU，  alpha是其一个参数的设定。
        # concat：   是否连接，
        #            对于多头注意力机制，GAT的论文建议，中间层使用拼接，而最后一层采取平均操作。
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features, out_features, bias=False)
        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        # 使得 W的参数 服从 一个设定的均匀分布。
        # https://blog.csdn.net/dss_dssssd/article/details/83959474
        # xavier_uniform_ 均匀分布
        # self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))

        self.a1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        # https://blog.csdn.net/qq_36955294/article/details/88117170
        self.a2 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
        # a1 a2 为两个可以训练的矩阵。
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = self.W(input)
        # [batch_size, N, out_features]
        # 分别获取到三个维度。   批量   句子的长度，即词的个数  每个词的维度。
        batch_size, N, _ = h.size()

        middle_result1 = torch.matmul(h, self.a1).expand(-1, -1, N)
        # h的形状：[batch_size, N, out_features]
        # a1的形状：[out_features，1]
        # matmul后： [batch_size, N, 1]
        middle_result2 = torch.matmul(h, self.a2).expand(-1, -1, N).transpose(1, 2)
        # [batch_size, N, 1] -> transpose -> [batch_size, 1, N]

        e = self.leakyrelu(middle_result1 + middle_result2)
        # 广播机制！ [batch_size, N, 1] + [batch_size, 1, N]
        # [batch_size, N, N]

        attention = e.masked_fill(adj == 0, -1e9)
        # 没有连接的，直接给一个很大的负的偏执项。
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)
            # elu：  一个激活函数。
            # https://pytorch.org/docs/stable/generated/torch.nn.ELU.html#torch.nn.ELU
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, layer):
        # 参数解释：
        # nfeat：  输入的维度，
        # nhid,
        # nclass：   n 分类任务，即输出的维度。
        # dropout：  dropout的比率。
        # alpha：  GAT中使用的激活函数：LeakyReLU，  alpha是其一个参数的设定。
        # nheads：  多头的头的个数。
        # layer ： int，表示层数。
        super(GAT, self).__init__()
        self.dropout = dropout
        self.layer = layer

        if self.layer == 1:
            self.attentions = [GraphAttentionLayer(nfeat, nclass, dropout=dropout, alpha=alpha, concat=True) for _ in
                               range(nheads)]
        else:
            self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                               range(nheads)]
            self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        if self.layer == 1:
            x = torch.stack([att(x, adj) for att in self.attentions], dim=2)
            x = x.sum(2)
            x = F.dropout(x, self.dropout, training=self.training)
            return F.log_softmax(x, dim=2)
        else:
            x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.elu(self.out_att(x, adj))
            return F.log_softmax(x, dim=2)


model = GAT(nfeat=5, nhid=7, nclass=3, dropout=0.6, alpha=0.1, nheads=3, layer=2)
a = torch.rand(10, 12, 5)
b = torch.rand(10, 12, 12)
c = model(a, b)
print(c.shape)


