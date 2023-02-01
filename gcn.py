class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """
    def __init__(self, mem_dim, layers, dropout_rate, self_loop = False):
        super(GraphConvLayer, self).__init__()
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = torch.nn.Dropout(dropout_rate)
        # linear transformation
        self.linear_output = nn.Linear(self.mem_dim, self.mem_dim)
        # dcgcn block
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))
        self.weight_list = self.weight_list.cuda()
        self.linear_output = self.linear_output.cuda()
        self.self_loop = self_loop

    def forward(self, adj, gcn_inputs):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []
        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            if self.self_loop:
                AxW = AxW  + self.weight_list[l](outputs)  # self loop
            else:
                AxW = AxW
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))
        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs
        out = self.linear_output(gcn_outputs)
        return out


class Bert(torch.nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.model = BertModel.from_pretrained('').cuda()  # , config=modelConfig
        embedding_dim = self.model.config.hidden_size
        self.dropout = torch.nn.Dropout(0.5)
        self.gcn = GraphConvLayer(768, 2, 0.5)  # 词向量的维度，  层数，  dropout比率。
        self.linear_1 = torch.nn.Linear(embedding_dim, 1)
        self.linear_2 = torch.nn.Linear(embedding_dim, 1)

    def forward(self, tokens, seg_embedding, attention_mask, adj_matrix):
        output = self.model(tokens, token_type_ids=seg_embedding, attention_mask=attention_mask)
        output = output[0]
        output = self.gcn(adj_matrix, output)  # 邻接矩阵，输入向量。
        output = self.dropout(output)
        output_1 = self.linear_1(output)
        output_2 = self.linear_2(output)
        return output_1.squeeze(-1), output_2.squeeze(-1)
