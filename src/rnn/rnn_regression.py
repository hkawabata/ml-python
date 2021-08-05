import numpy as np
from tqdm import tqdm
from module.layer import TimeRNNRegression


class RNNRegressionModel:
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
        self.layer = TimeRNNRegression(D, H)

        self.loss = []

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
        for _ in tqdm(range(epochs)):
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

                # back propagation
                self.layer.backward()

                # update params
                for i in range(len(self.layer.params)):
                    self.layer.params[i] -= self.eta * self.layer.grads[i]

            self.loss.append(loss)

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
T = 10
theta = np.array(range(N)) * np.pi / 50
A1, A2, A3 = 10.0, 2.5, 5.0    # 振幅
An1, An2, An3 = 1.0, 0.5, 1.0  # ノイズの振幅
F1, F2, F3 = 5, 1, 3           # 振動数
x = np.concatenate([
    A1 * np.sin(F1 * theta) + An1 * np.random.randn(N),
    A2 * np.sin(F2 * theta) + An2 * np.random.randn(N),
    A3 * np.sin(F3 * theta) + An3 * np.random.randn(N)
]).reshape(3, N).T

model = RNNRegressionModel(eta=1.0e-4, batch_size=10, D=x.shape[1], T=T, H=40)
model.train(x, 2000)
print(model.loss[-1])

# 学習曲線
from matplotlib import pyplot as plt
fig = plt.figure()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(range(len(model.loss)), model.loss)
plt.axhline(y=model.loss[-1], linewidth=0.5, color='red', label=f'loss = {model.loss[-1]:.4f}')
plt.legend()
fig.savefig('rnn_regression_loss.png')


start = 30
T_corr = 50
T_pred = 500

x_in = x[start:start+T_corr].copy()
x_pred = []
for _ in range(T_pred):
    x_next = model.predict(x_in)
    x_pred.append(x_next)
    x_in[:-1] = x_in[1:]
    x_in[-1] = x_next
x_pred = np.array(x_pred)

for d in range(x.shape[1]):
    fig = plt.figure()
    plt.title(f'variable {d}')
    plt.xlabel('theta')
    plt.ylabel('value')
    plt.plot(theta[start:start+T_corr+T_pred], x[start:start+T_corr+T_pred, d], label='actual')
    plt.plot(theta[start+T_corr:start+T_corr+T_pred], x_pred[:, d], label='predicted')
    plt.legend()
    fig.savefig(f'rnn_regression_variable{d:02d}.png')
