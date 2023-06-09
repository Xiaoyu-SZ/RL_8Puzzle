from exp import *

dRow = [-1, 1, 0, 0]
dCol = [0, 0, -1, 1]


# function for generating a puzzle that could be solved in N steps

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
    return state



sr = []
for i in range(20):
    # 测试1-10的难度
    print(f"==={i}===")
    board = [[0,4,2],[7,1,3],[8,5,6]] # 随便举个例子，不会用到
    board = generate(i+1)
    exp_name='dqn_embed=4_nstep=3' # 指定version
    done, sequence = inference(board,exp_name)
    print(done,sequence)


