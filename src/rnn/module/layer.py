import numpy as np

class RNN:
    def __init__(self, Wx, Wh, b):
        """
        Wh : hの重み
        Wx : xの重み
        b  : バイアス
        """
        self.Wh = Wh  # 全てのRNNブロックで同一の重みを共有
        self.Wx = Wx  # 全てのRNNブロックで同一の重みを共有
        self.b = b  # 全てのRNNブロックで同一の重みを共有
        self.grads = [
            np.zeros_like(Wx),
            np.zeros_like(Wh),
            np.zeros_like(b)
        ]  # 各RNNブロックが個別に保有
        self.cache = None

    def forward(self, X, Hprev, is_training=False):
        """
        X     : その時刻の入力（行ベクトルを積んだ行列）
        Hprev : 1つ前の時刻の出力（行ベクトルを積んだ行列）
        """
        H = np.tanh(np.dot(X, self.Wx) + np.dot(Hprev, self.Wh) + self.b)
        if is_training:
            self.cache = (X, H, Hprev)
        return H

    def backward(self, dH):
        X, H, Hprev = self.cache
        self.cache = 0
        tmp = dH * (1.0 - H * H)

        dX = np.dot(tmp, self.Wx.T)
        dHprev = np.dot(tmp, self.Wh.T)
        dWx = np.dot(X.T, tmp)
        dWh = np.dot(Hprev.T, tmp)
        db = tmp.sum(axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dX, dHprev


class TimeRNN:
    """
    Trancated BPTT を行う際の1セットを表すレイヤ
    """

    def __init__(self, Wx, Wh, b, stateful=False):
        """
        stateful : 次の TimeRNN レイヤへ隠れ状態Hを引き継ぐかどうか
        """
        self.params = [Wx, Wh, b]
        self.grads = [
            np.zeros_like(Wx),
            np.zeros_like(Wh),
            np.zeros_like(b)
        ]
        self.layers = None
        self.h, self.dh = None, None
        self.stateful = stateful

    def forward(self, xs, is_training=False):
        """
        Xs : データサンプル
            - D次元のデータサンプルを
            - 時系列にT個並べたものを
            - N個のバッチにまとめた3次元配列
        """
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')
        for t in range(T):
            layer = RNN(Wx, Wh, b)
            self.h = layer.forward(xs[:, t, :], self.h, is_training)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        grads = [0, 0, 0]

        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs

    def reset_state(self):
        """
        隠れ状態をリセット
        学習済みモデルで予測を行うときに使う
        """
        self.h = None


class Affine:
    def __init__(self, W, b):
        # モデルの重み
        self.params = [W, b]
        # 重みの勾配
        self.grads = [np.empty_like(W), np.empty_like(b)]
        # 誤差逆伝播用の中間データ
        self.cache = None

    def forward(self, X, is_training=False):
        W, b = self.params
        out = np.dot(X, W) + b

        if is_training:
            self.cache = X
        return out

    def backward(self, dout):
        W, b = self.params
        X = self.cache
        self.cache = None

        dX = np.dot(dout, W.T)
        dW = np.dot(X.T, dout)
        db = dout.sum(axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dX


class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.empty_like(W), np.empty_like(b)]
        self.cache = None

    def forward(self, xs, is_training=False):
        W, b = self.params
        N, T, D = xs.shape
        D, H = W.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        for t in range(T):
            layer = Affine(W, b)
            h = layer.forward(xs[:, t, :], is_training)
            hs[:, t, :] = h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        W, b = self.params
        N, T, H = dhs.shape
        D, H = W.shape

        dxs = np.empty((N, T, D), dtype='f')
        grads = [0, 0]

        for t in reversed(range(T)):
            layer = self.layers[t]
            dx = layer.backward(dhs[:, t, :])
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        return dxs


class LeastSquareLoss:
    def __init__(self):
        self.cache = None

    def forward(self, X_pred, X_corr, is_training=False):
        X_err = X_pred - X_corr
        cost = (X_err * X_err).sum()
        if is_training:
            self.cache = X_err
        return cost

    def backward(self, dout=1.0):
        X_err = self.cache
        batch_size = X_err.shape[0]
        self.cache = None
        return 2 * X_err * dout / batch_size


class TimeLeastSquareLoss:
    def __init__(self):
        self.cache = None

    def forward(self, xs, xs_corr, is_training=False):
        N, T, D = xs.shape

        cost = 0
        self.layers = []
        for t in range(T):
            layer = LeastSquareLoss()
            cost += layer.forward(xs[:, t, :], xs_corr[:, t, :], is_training)
            self.layers.append(layer)

        if is_training:
            self.cache = (N, T, D)

        return cost

    def backward(self):
        N, T, D = self.cache
        dxs = np.empty((N, T, D), dtype='f')
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx = layer.backward()
            dxs[:, t, :] = dx

        return dxs


class SoftmaxWithLoss:
    def __init__(self):
        self.cache = None
        self.eps =1.0e-8

    def forward(self, X, P_corr, is_training=False):
        P_pred = self.__softmax(X)
        cost = self.__calc_cost(P_pred, P_corr)
        if is_training:
            self.cache = (P_pred, P_corr)
        return cost

    def backward(self, dout=1.0):
        P_pred, P_corr = self.cache
        self.cache = None
        batch_size = P_corr.shape[0]
        dX = (P_pred - P_corr) * dout / batch_size
        return dX

    def __softmax(self, X):
        """
        X : 行ベクトルのデータサンプルを積み重ねた行列
        """
        expX = np.exp(X)
        return (expX.T / expX.sum(axis=1)).T

    def __calc_cost(self, P_pred, P_corr):
        """
        P_pred : 予測所属確率の行ベクトルを積み重ねた行列
        P_corr : 正解所属確率
        """
        #cost = - np.sum(P_corr * np.log(P_pred) + (1.0 - P_corr) * np.log(1.0 - P_pred), axis=0)
        cost = - np.sum(P_corr * np.log(P_pred + self.eps), axis=1)
        cost = np.average(cost)
        return cost


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.cache = None

    def forward(self, xs, P_corr, is_training=False):
        N, T, D = xs.shape

        cost = 0
        self.layers = []
        for t in range(T):
            layer = SoftmaxWithLoss()
            cost += layer.forward(xs[:, t, :], P_corr[:, t, :], is_training)
            self.layers.append(layer)

        if is_training:
            self.cache = (N, T, D)

        return cost

    def backward(self):
        N, T, D = self.cache
        dxs = np.empty((N, T, D), dtype='f')
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx = layer.backward()
            dxs[:, t, :] = dx

        return dxs


class TimeRNNRegression:
    """
    回帰問題を解く
    """

    def __init__(self, D, H):
        """
        D : データの次元
        H : RNN 隠れ状態のノード数
        """
        self.D = D
        self.H = H

        # レイヤ初期化
        sigma = 0.01
        self.Wx_rnn = sigma * np.random.randn(D, H)
        self.Wh_rnn = sigma * np.random.randn(H, H)
        self.b_rnn = sigma * np.random.randn(H)
        self.W_aff = sigma * np.random.randn(H, D)
        self.b_aff = sigma * np.random.randn(D)
        self.layers = [
            TimeRNN(self.Wx_rnn, self.Wh_rnn, self.b_rnn, stateful=True),
            TimeAffine(self.W_aff, self.b_aff)
        ]
        self.loss_layer = TimeLeastSquareLoss()

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, ys, is_training=False):
        """
        xs : 入力。T個の時系列データ（標準化済）
        ys : 主力の正解。T個の時系列データ（標準化済）
        """
        out = xs
        for layer in self.layers:
            out = layer.forward(out, is_training)
        loss = self.loss_layer.forward(out, ys, is_training)
        return loss

    def backward(self):
        din = self.loss_layer.backward()
        for layer in reversed(self.layers):
            din = layer.backward(din)
        return din

    def reset_state(self):
        self.layers[0].reset_state()


class TimeRNNClassification:
    """
    分類問題を解く
    """

    def __init__(self, D, H):
        """
        D : データの次元
        H : RNN 隠れ状態のノード数
        """
        self.D = D
        self.H = H

        # レイヤ初期化
        sigma = 0.01
        self.Wx_rnn = sigma * np.random.randn(D, H)
        self.Wh_rnn = sigma * np.random.randn(H, H)
        self.b_rnn = sigma * np.random.randn(H)
        self.W_aff = sigma * np.random.randn(H, D)
        self.b_aff = sigma * np.random.randn(D)
        self.layers = [
            TimeRNN(self.Wx_rnn, self.Wh_rnn, self.b_rnn, stateful=True),
            TimeAffine(self.W_aff, self.b_aff)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, ys, is_training=False):
        """
        xs : 入力。T個の時系列データ（標準化済）
        ys : 主力の正解。T個の時系列データ（標準化済）
        """
        out = xs
        for layer in self.layers:
            out = layer.forward(out, is_training)
        loss = self.loss_layer.forward(out, ys, is_training)
        return loss

    def backward(self):
        din = self.loss_layer.backward()
        for layer in reversed(self.layers):
            din = layer.backward(din)
        return din

    def reset_state(self):
        self.layers[0].reset_state()
