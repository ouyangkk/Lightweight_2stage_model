
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, stack_num, n_freq, BN=False):
        super(DNN, self).__init__()

        n_hid = 2048
        self.BN = BN
        self.fc1 = nn.Linear(stack_num * n_freq, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_hid)
        self.fc4 = nn.Linear(n_hid, 257)

        if BN:
            self.bn1 = nn.BatchNorm1d(n_hid)
            self.bn2 = nn.BatchNorm1d(n_hid)
            self.bn3 = nn.BatchNorm1d(n_hid)

    def forward(self, x):
        drop_p = 0.2
        (_, stack_num, n_freq) = x.size()
        x = x.view(-1, stack_num * n_freq)

        if self.BN:
            x2 = F.dropout(F.relu(self.bn1(self.fc1(x))), p=drop_p, training=self.training)
            x3 = F.dropout(F.relu(self.bn2(self.fc2(x2))), p=drop_p, training=self.training)
            x4 = F.dropout(F.relu(self.bn3(self.fc3(x3))), p=drop_p, training=self.training)
        else:
            x2 = F.dropout(F.relu(self.fc1(x)), p=drop_p, training=self.training)
            x3 = F.dropout(F.relu(self.fc2(x2)), p=drop_p, training=self.training)
            x4 = F.dropout(F.relu(self.fc3(x3)), p=drop_p, training=self.training)
        x5 = self.fc4(x4)

        return x5



class LSTM(nn.Module):
    def __init__(self, stack_num, n_freq, BN=False):
        super(LSTM, self).__init__()

    #     fc_hid = 257
    #     lstm_hid = 512
    #     self.BN = BN
    #     self.lstm = nn.LSTM(input_size=n_freq, hidden_size=lstm_hid, batch_first=True, dropout=0.4,
    #                          num_layers=3)
    #     # self.fc1 = nn.Sequential(
    #     #     nn.Linear(lstm_hid,fc_hid),
    #     #     nn.LeakyReLU(),
    #     #     nn.Linear(fc_hid,257),
    #     #     nn.LeakyReLU()
    #     # )
    #     self.linear = nn.Linear(in_features=lstm_hid,out_features=fc_hid)
    #     self.activation  = nn.Sigmoid()
    #     # self.fc1 = nn.Linear(lstm_hid, 257)
    # def forward(self, x):
    #     drop_p = 0.2
    #     lstm_output, _ = self.lstm(x)
    #     # x1 =  F.dropout(F.relu(lstm_output), p=drop_p, training=self.training)
    #
    #     x2 = self.linear(lstm_output)
    #     x2 = self.activation(x2)
    #     return x2
        self.lstm = nn.LSTM(input_size=257, hidden_size=512, num_layers=2, batch_first=True,bidirectional=False)

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 257),
            nn.Sigmoid()
        )
    def forward(self, input):

        lstm_out, (h_0, c_0) = self.lstm(input)

        mask = self.fc(lstm_out)


        return mask