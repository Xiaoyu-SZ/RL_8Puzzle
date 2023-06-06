# coding:UTF-8
# 八数码问题的环境
# import gym
import numpy as np
# from sklearn.model_selection import train_test_split

# from TestCase import testCase
import cv2
import time

# import seaborn as sns

# sns.set_style('whitegrid')
np.set_printoptions(suppress=True, linewidth=500, edgeitems=8, precision=4)



# 上下左右四个移动方向
dRow = [-1, 1, 0, 0]
dCol = [0, 0, -1, 1]

def generate(steps=4):
    target = np.array([[1,2,3], [4, 5,6], [7, 8, 0]])
    state = target.copy().reshape(3, 3)
    a = 0
    last_a = -1
    pos = [2,2]
    for setp in range(steps):
        nowRow = pos[0]
        nowCol = pos[1]
        a = np.random.randint(4)
        nextRow = nowRow + dRow[a]
        nextCol = nowCol + dCol[a]
        while (not checkBounds(nextRow, nextCol, 3, 3)) or\
                (a == 0 and last_a == 1) or (a == 1 and last_a == 0) or\
                (a == 2 and last_a == 3) or (a == 3 and last_a == 2):
            a = np.random.randint(4)
            nextRow = nowRow + dRow[a]
            nextCol = nowCol + dCol[a]
        last_a = a
        nextState = state.copy()
        # 移动方格
        swap(nextState, nowRow, nowCol, nextRow, nextCol)
        pos = np.array([nextRow, nextCol])
        state = nextState
    return pos,state



def swap(matrix, row1, col1, row2, col2):
    '''
       交换矩阵的两个元素
    '''
    t = matrix[row1, col1]
    matrix[row1, col1] = matrix[row2, col2]
    matrix[row2, col2] = t


def checkBounds(i, j, m, n):
    '''
        检测下标是否越界
    '''
    if i >= 0 and i < m and j >= 0 and j < n:
        return True
    else:
        return False


def arrayToMatrix(arr, m, n):
    '''
    数组转矩阵
    '''
    return np.resize(arr, [m, n])


def matirxToArray(matrix):
    '''
    矩阵转数组
    '''
    return matrix.ravel()




def findZeroPos(board, m, n):
    '''
    找到0所在位置
    '''
    startRow = -1
    startCol = -1
    flag = True
    for i in range(0, m):
        if not flag:
            break
        for j in range(0, n):
            if board[i, j] == 0:
                startRow = i
                startCol = j
                flag = False
    return np.array([startRow, startCol])


def A_star_dist(target, now):
    length = len(now)
    cnt = 0
    for i in range(0, length):
        if target[i] != now[i]:
            cnt += 1
    return cnt


class EightPuzzleEnv:  # gym.Env
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 20
    }
    # 动作空间
    ActionDim = 4
    # 默认行数
    DefaultRow = 3
    # 默认列数
    DefaultCol = 3
    # 成功回报
    SuccessReward = 10.0

    def __init__(self, m=DefaultRow, n=DefaultCol):
        self.viewer = None
        # 一轮episode最多执行多少次step
        if m == 2:
            self._max_episode_steps = 100  # 2048
        else:
            self._max_episode_steps = 30
        self.target = np.array([1,2,3,4,5,6,7,8,0])
        self.m = m
        self.n = n
        self.last_step = -1


    def reset(self, step=4,board=None):
        if not board is None:
            self.state = board
            self.pos = findZeroPos(board,3,3)
        else:
            self.pos,self.state = generate(step)
        return self.state
    
    def step(self, a):
        nowRow = self.pos[0]
        nowCol = self.pos[1]
        nextRow = nowRow + dRow[a]
        nextCol = nowCol + dCol[a]
        nextState = self.state.copy()
        # 检查越界
        if not checkBounds(nextRow, nextCol, self.m, self.n):
            return self.state, -2.0, False, {'info': -1, 'MSG': 'OutOfBounds!'}
        # 移动方格
        swap(nextState, nowRow, nowCol, nextRow, nextCol)
        self.pos = np.array([nextRow, nextCol])
        # 获得奖励
        re = self.reward(self.state, nextState, a)
        self.last_step = a
        self.state = nextState
        if re == EightPuzzleEnv.SuccessReward:
            return self.state, re, True, {'info': 2, 'MSG': 'Finish!'}
        return self.state, re, False, {'info': 1, 'MSG': 'NormalMove!'}

    def isFinish(self, s):
        '''
        检查是否到达终点
        '''
        if np.array_equal(s.ravel(), self.target):
            return True
        else:
            return False

    def reward(self, nowState, nextState, a):
        re = 0
        '''
        奖励函数
        '''
        if self.isFinish(nextState):
            # 到达终点，给予最大奖励
            return EightPuzzleEnv.SuccessReward
        else:
            # 对移动前的棋盘、移动后的棋盘分别进行估价A_star_dist
            # lastDist = Manhattan(self.target.reshape(self.m, self.n), nowState)
            # nowDist = Manhattan(self.target.reshape(self.m, self.n), nextState)
            lastDist = A_star_dist(self.target, nowState.ravel())
            nowDist = A_star_dist(self.target, nextState.ravel())
            if (a == 0 and self.last_step == 1) or (a == 1 and self.last_step == 0) or\
                    (a == 2 and self.last_step == 3) or (a == 3 and self.last_step == 2):
                re -= 0.5
            # 距离减小，给予较小惩罚
            if nowDist < lastDist:
                re -= 0.1
            # 距离不变，给予中等惩罚
            elif nowDist == lastDist:
                re -= 0.2
            # 距离增大，给予较大惩罚
            else:
                re -= 0.5
            return re

    def render(self, mode='human', close=False):
        '''
                渲染
        '''
        time.sleep(1)
        print('----------')
        for i in range(0, self.m):
            print('|', end='')
            for j in range(0, self.n):
                print(self.state[i][j], end='')
                print('|', end='')
            print()
        print('----------')

    @property
    def max_episode_steps(self):
        return self._max_episode_steps
