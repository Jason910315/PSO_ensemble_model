import numpy as np
import pandas as pd
from collections import Counter,defaultdict
from sklearn.model_selection import KFold
import os, time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import copy

# 隨機生成 NB 的先驗機率、似然機率
def random_naive_model(feature_nums, class_nums):
    # 使用 dirichlet() 生成隨機的先驗機率，其總和為 1
    random_prior = np.random.dirichlet(np.ones(class_nums))
    # 隨機生成的 likelihood，格式為 feature_{feature_index} : 類別 0 [feature_index 特徵所有可能值的機率],...,類別 n [feature_index 特徵所有可能值的機率]
    random_likelihood = {
                f"feature_{feature_index}": np.array([np.random.dirichlet(np.ones(10)) for _ in range(class_nums)])
                for feature_index in range(feature_nums) }
    return random_prior,random_likelihood

# 根據隨機生成的機率，預測樣本類別
def prediction(X, prior, likelihood):
    predictions = []
    # 預測每個樣本
    for x in X:
        # 針對單一樣本的每個特徵 j，取出 int(x[j]) 的機率 (特徵值)
        posterior = prior * np.prod([likelihood[f"feature_{j}"][:, int(x[j])] for j in range(len(x))], axis = 0)
        predicted_class = np.argmax(posterior)
        predictions.append(predicted_class)
    return np.array(predictions)

# 將正確率當作 PSO 的適應值，因此要越大越好
def cal_accuracy_fitness(X, y, prior, likelihood):
    predictions = prediction(X, prior, likelihood)
    return accuracy_score(y, predictions)

# 粒子群優化演算法，bounds 設定速度與空間的上下界
# 每個粒子都代表一個模型，帶有兩個參數 - prior、likelihood
def PSO(X, y, num_particles, feature_nums, class_nums, bounds, max_iter):

    # 所有粒子位置、速度、個體最佳位置、個體最佳適應值
    positions = []
    velocities = []
    personal_best_positions = []
    personal_best_fitness = []

    # 初始化每個粒子的位置、速度
    for i in range(num_particles):
        # 隨機初始化粒子先驗機率、似然機率
        particle_prior, particle_likelihood = random_naive_model(feature_nums, class_nums)
        positions.append((particle_prior, particle_likelihood))

        # 隨機初始化先驗機率、似然機率的「速度」，總和不需為 1
        prior_veclocity = np.random.uniform(bounds[0], bounds[1], size = class_nums) * 0.1  # 產生一個大小為 size 的隨機數陣列
        likelihood_velocity = {
            f"feature_{feature_index}":np.random.uniform(bounds[0], bounds[1], size = (class_nums, 10)) * 0.1
            for feature_index in range(feature_nums)
        }
        velocities.append((prior_veclocity, likelihood_velocity))

    positions = np.array(positions, dtype = object)    # 轉為 NumPy 陣列，確保支持複雜結構
    velocities = np.array(velocities, dtype = object)  

    # 計算每個粒子的初始適應值
    fitness = np.array([cal_accuracy_fitness(X, y, particle_prior, particle_likelihood) for (particle_prior, particle_likelihood) in positions])
    personal_best_positions = copy.deepcopy(positions)  # 初始化個體最佳位置、適應值
    personal_best_fitness = fitness

    best_fitness_index = np.argmax(fitness)  # 求出最佳適應值的粒子的索引
    global_best_positions = personal_best_positions[best_fitness_index]     # 初始化群體最佳位置與適應值
    global_best_fitness = personal_best_fitness[best_fitness_index]

    # 只要不超過迭代次數就一直跑
    for i in range(max_iter):
    
        # 設定權重、速度因子
        w = 0.9
        c1, c2 = 2        
        # 更新每個粒子
        for j in range(num_particles):
            prior, likelihood = positions[j]
            pbest_prior, pbest_likelihood = personal_best_positions[j]
            gbest_prior, gbest_likelihood = global_best_positions

