# %%
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

    def __init__(self, width: int, height: int):
        """
        Args:
            width: lattice width
            height: lattice height
        """
        self._width = width
        self._height = height
        self._node = np.zeros((width + 2, height + 2), dtype="float32").tolist()
        self._weight_horizontal = np.zeros((width + 1, height), dtype="float32").tolist()
        self._weight_vertical = np.zeros((width, height + 1), dtype="float32").tolist()
        
        self.burnin_time = 100
        self.sumpling_num = 100
        
    def _sum_around_weight_node(self, row: int, col: int):
        """
        最近接の情報を取り込む
        
        Args:
            row: サンプリングするノードの行番号
            col: サンプリングするノードの列番号        
        """
        
        sum_value = self._weight_horizontal[col - 1][row] * self._node[col - 1][row]
        sum_value += self._weight_horizontal[col][row] * self._node[col + 1][row]
        sum_value += self._weight_vertical[col][row - 1] * self._node[col][row - 1]
        sum_value += self._weight_vertical[col][row] * self._node[col][row + 1]
        
        return sum_value
        
    def gibbs_sampling(self, row: int, col: int):
        """
        ノード x[row][col]のギブスサンプリング
        
        Args:
            row: サンプリングするノードの行番号
            col: サンプリングするノードの列番号
        """
        u = random.uniform(0, 1)
        
        z = math.exp(self._sum_around_weight_node(row, col)) + 1.0
        p = 1.0 / z
        
        value = 1.0 if p < u else 0.0
        
        return value
    
    def burnin(self):
        """
        バーンイン実行
        
        XXX: 変数のインスタンスに注意. deepcopyしている
        """
        
        for n in range(self.burnin_time):
            self._node = _sampling_all_node()
                
    def _sampling_all_node():
        """
        全ノードサンプリング
        
        Returns:
            *list - サンプル結果(※内部計算用のパディングしたノード含む)
        """
        
        nodes = deep_copy(self._node)

        for i in range(self.height):
            for j in range(self.width):
                nodes[j + 1][i + 1] = self.gibbs_sampling(i, j)
                    
        return nodes
    
    
    def update_params(self, x: np.ndarray):
        """
        バッチで1回パラメータ更新する
        
        Args:
            x: 学習データ　<b, h, w>
        """
        # データの平均
        
        
        # モデルの期待値計算 (モンテカルロ積分)
        exp_value = np.zeros((self.width, self.height)).tolist()   
        for n in self.sumpling_num:
            self.burnin()
            nodes = self._sampling_all_node()

            # サンプル平均
            for i in range(self.height):
                for j in range(self.width):
                    # exp_value[i][j]
        
    def fit(self, x: np.ndarray, epoch: str = 1):
        """
        学習する.
        
        Args:
            x: 学習データ　<b, h, w>
        """
        
        pass
    
    def transform(self, x):
        """
        """
        pass
        

#%%
