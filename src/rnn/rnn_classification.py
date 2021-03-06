import numpy as np
from tqdm import tqdm
from module.layer import TimeRNNClassification


class RNNClassificationModel:
    """
    回帰問題を解く
    """

    def __init__(self, eta, batch_size, D, T, H):
        """
        eta        : 学習率
        batch_size : ミニバッチ学習のバッチサイズ
        D          : データの次元
        T          : Trancated BPTT を行う際の RNN ブロック数
        H          : RNN 隠れ状態のノード数
        """
        self.eta = eta
        self.batch_size = batch_size
        self.D = D
        self.T = T
        self.H = H

        self.x_ave = None  # 平均値
        self.x_sd = None  # 標準偏差
        self.layer = TimeRNNClassification(D, H)

        self.loss = []
        self.perplexity = []

    def train(self, x, epochs):
        """
        x      : 次元D,長さLの時系列データ
        epochs :
        """
        # 学習の入力時系列長
        L_in = x.shape[0] - 1

        # 標準化
        self.x_ave = x.mean(axis=0)
        self.x_sd = x.std(axis=0)
        x_std = (x - self.x_ave) / self.x_sd

        # 入力データと正解データ
        x_2cycle = np.concatenate([x_std, x_std])  # 末尾に到達したら先頭に戻る
        x_prev = x_2cycle[:-1]
        x_next = x_2cycle[1:]

        jump = L_in // self.batch_size
        for e in tqdm(range(epochs)):
            # バッチデータ生成
            offset = np.random.randint(jump)
            batch_x_prev = np.empty((self.batch_size, L_in, self.D), dtype='f')
            batch_x_next = np.empty((self.batch_size, L_in, self.D), dtype='f')
            for i in range(self.batch_size):
                batch_x_prev[i] = x_prev[offset:offset + L_in]
                batch_x_next[i] = x_next[offset:offset + L_in]
                offset += jump
            # 学習
            base = 0
            while base < L_in:
                t_start = base
                t_end = t_start + self.T
                if t_end >= L_in:
                    t_end = L_in
                base += self.T

                # forward propagation
                xs = batch_x_prev[:, t_start:t_end, :]
                ys = batch_x_next[:, t_start:t_end, :]
                loss = self.layer.forward(xs, ys, is_training=True)
                perplexity = np.exp(loss)

                # back propagation
                self.layer.backward()

                # update params
                for i in range(len(self.layer.params)):
                    self.layer.params[i] -= self.eta * self.layer.grads[i]

            self.loss.append(loss)
            self.perplexity.append(perplexity)

    def predict(self, x):
        """
        x : D次元のデータが時系列にL個並んだもの
        """
        # RNN の隠れ状態をリセット
        self.layer.reset_state()

        L, D = x.shape
        x_std = (x - self.x_ave) / self.x_sd
        x_std = x_std.reshape(1, L, D)

        out = x_std
        for l in self.layer.layers:
            out = l.forward(out)
        out = out.reshape(L, D)
        result = out * self.x_sd + self.x_ave
        return result[-1]



# 学習
N = 1000
T = 20
A, B, C = 10.0, 2.5, 5.0    # 振幅
An, Bn, Cn = 0.1, 0.2, 0.1  # ノイズの振幅
#An, Bn, Cn = 0, 0, 0
x = np.concatenate([
    A*np.sin(np.linspace(0, 40*np.pi, N)) + An*np.random.randn(N),
    B*np.sin(np.linspace(0, 20*np.pi, N)) + Bn*np.random.randn(N),
    C*np.sin(np.linspace(0, 80*np.pi, N)) + Cn*np.random.randn(N)
]).reshape(3, N).T

#model = RNNRegression(eta=1.0e-4, batch_size=10, T=T, H=50)
model = RNNClassificationModel(eta=1.0e-6, batch_size=10, D=x.shape[1], T=T, H=20)
model.train(x, 1000)
print(model.loss[-1])

# 学習曲線
from matplotlib import pyplot as plt
plt.plot(range(len(model.loss)), model.loss)
plt.show()



start = 0
T_corr = 300
T_pred = 100

x_in = x[start:start+T_corr].copy()
x_pred = []
for _ in range(T_pred):
    x_next = model.predict(x_in)
    x_pred.append(x_next)
    x_in[:-1] = x_in[1:]
    x_in[-1] = x_next
x_pred = np.array(x_pred)

for d in range(x.shape[1]):
    plt.plot(range(T_corr), x[start:start+T_corr, d], label='a')
    plt.plot(range(T_corr, T_corr+x_pred.shape[0]), x_pred[:, d], label='b')
    plt.legend()
    plt.show()