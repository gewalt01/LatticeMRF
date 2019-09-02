"""
ハンズオン用

パフォーマンスについては一切考慮していない
入出力は{-1, +1}の値を取る行列
"""

import numpy as np
import math
import random
import copy


class LatticeMRF:
    """
    Attributes:
        burnin_time (int): バーンイン期間
        sumpling_num (int): サンプリング数

    NOTE:
        ROW, COLの向き要確認
    """

    @property
    def width(self):
        """int: Lattice width"""
        return self._width

    @property
    def height(self):
        """int: Lattice height."""
        return self._height

    lr = 0.1  #: float: 学習率

    def __init__(self, width: int, height: int):
        """
        Args:
            width: lattice width
            height: lattice height
        """
        self._width = width
        self._height = height
        self._node = np.zeros((width + 2, height + 2), dtype="float32")
        self._weight_horizontal = np.random.uniform(-0.01, +0.01, (width + 1, height)).astype("float32")
        self._weight_horizontal[0, :] = 0  # パディングノードに対するウェイトは0
        
        self._weight_vertical = np.random.uniform(-0.01, +0.01, (width, height + 1)).astype("float32")
        self._weight_vertical[:, 0] = 0  # パディングノードに対するウェイトは0

        self.burnin_time = 15
        self.sumpling_num = 30

    def _sum_around_weight_node(self, row: int, col: int):
        """
        最近接の情報を取り込む

        Args:
            row: サンプリングするノードの行番号
            col: サンプリングするノードの列番号

        XXX:
            パディング関係を要確認
        """

        sum_value = self._weight_horizontal[1 + col - 1, row] * self._node[1 + col - 1, 1 + row]
        sum_value += self._weight_horizontal[1 + col, row] * self._node[1 + col + 1, 1 + row]
        sum_value += self._weight_vertical[col, 1 + row - 1] * self._node[1 + col, 1 + row - 1]
        sum_value += self._weight_vertical[col, 1 + row] * self._node[1 + col, 1 + row + 1]

        return sum_value

    def gibbs_sampling(self, row: int, col: int):
        """
        ノード x[row][col]のギブスサンプリング

        Args:
            row: サンプリングするノードの行番号
            col: サンプリングするノードの列番号
        """
        u = random.uniform(0, 1)

        z = math.exp(self._sum_around_weight_node(row, col)) + math.exp(-self._sum_around_weight_node(row, col))
        p = math.exp(-self._sum_around_weight_node(row, col)) / z

        value = -1.0 if u < p else 1.0

        return value

    def burnin(self):
        """
        バーンイン実行

        XXX: 変数のインスタンスに注意. deepcopyしている
        """

        for n in range(self.burnin_time):
            self._node = self._sampling_all_node()

    def _sampling_all_node(self):
        """
        全ノードサンプリング

        Returns:
            * np.ndarray - サンプル結果(※内部計算用のパディングしたノード含む)
        """

        nodes = copy.deepcopy(self._node)

        for i in range(self.height):
            for j in range(self.width):
                nodes[j + 1, i + 1] = self.gibbs_sampling(i, j)

        return nodes

    def update_params(self, x: np.ndarray):
        """
        バッチで1回パラメータ更新する

        Args:
            x: 学習データ　<b, h, w>
        """
        batch_size = x.shape[0]
        mean_w_h_data = np.zeros((self.width + 1, self.height), dtype="float32")
        mean_w_v_data = np.zeros((self.width, self.height + 1), dtype="float32")

        # データ平均の計算
        for n in range(batch_size):
            data = np.zeros((x[n].shape[0] + 2, x[n].shape[1] + 2))
            data[1:-1, 1:-1] = x[n]

            for i in range(self.height):
                for j in range(self.width):
                    mean_w_h_data[1 + j, i] = data[1 + j, 1 + i] * data[1 + j + 1, 1 + i] / batch_size
                    mean_w_v_data[j, 1 + i] = data[1 + j, 1 + i] * data[1 + j, 1 + i + 1] / batch_size

        # モデルの期待値計算 (モンテカルロ積分)
        mean_w_h_model = np.zeros((self.width + 1, self.height), dtype="float32")
        mean_w_v_model = np.zeros((self.width, self.height + 1), dtype="float32")
        for n in range(self.sumpling_num):
            self.burnin()
            nodes = self._sampling_all_node()

            # サンプル平均
            for i in range(self.height):
                for j in range(self.width):
                    mean_w_h_model[1 + j, i] = nodes[1 + j, 1 + i] * nodes[1 + j + 1, 1 + i] / self.sumpling_num
                    mean_w_v_model[j, 1 + i] = nodes[1 + j, 1 + i] * nodes[1 + j, 1 + i + 1] / self.sumpling_num

        # パラメータ更新 (勾配上昇法)
        for i in range(self.height):
            for j in range(self.width):
                grad_h = LatticeMRF.lr * (mean_w_h_data[1 + j, i] - mean_w_h_model[1 + j, i])
                grad_v = LatticeMRF.lr * (mean_w_v_data[j, 1 + i] - mean_w_v_model[j, 1 + i])
                self._weight_horizontal[1 + j, i] += grad_h
                self._weight_vertical[j, 1 + i] += grad_v

    def fit(self, x: np.ndarray, epoch: str = 50):
        """
        学習する.

        Args:
            x: 学習データ　<b, h, w>
        """

        for e in range(epoch):
            self.update_params(x)

    def transform(self, x: np.ndarray):
        """
        Args:
            x: 入力データ <w, h>

        Returns:
            * np.ndarray - 1回推論後のデータ

        NOTICE:
            乱数サンプリングするか最大確率をとるか未定
        """
        self._node = np.array(self._node, dtype="float32")
        self._node[1:-1, 1:-1] = x
        return self._sampling_all_node()[1:-1, 1:-1]
