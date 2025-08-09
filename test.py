import numpy as np

cc = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])

print(len(cc))
print(cc.shape[0])
print(cc.shape[1])

prior = [0.5,0.5]
# 這段程式碼沒有錯誤，會正確印出一個 shape 為 (2,) 的全 1 陣列
posterior = np.ones(len(prior))
print(posterior)