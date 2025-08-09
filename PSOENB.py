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
    random_likelihood = np.random.dirichlet(np.ones(10), size = (feature_nums, class_nums))
    return random_prior,random_likelihood

# 根據隨機生成的機率，預測樣本類別
def prediction(X, prior, likelihood):
    predictions = []
    # 預測每個樣本
    N = len(X)
    F, C, V = likelihood.shape
    # 預測每個樣本
    posterior = np.tile(prior, (N, 1))   # 形成一個 (N, C) 的矩陣

    for f in range(F):
        posterior *= likelihood[f][:, X[:,f]].T
    predictions = np.argmax(posterior, axis = 1)
    return np.array(predictions)

# 將正確率當作 PSO 的適應值，因此要越大越好
def cal_accuracy_fitness(X, y, prior, likelihood):
    predictions = prediction(X, prior, likelihood)
    return accuracy_score(y, predictions)

# 粒子群優化演算法，bounds 設定速度與空間的上下界
# 每個粒子都代表一個模型，帶有兩個參數 - prior、likelihood
def PSO(X, y, num_particles, feature_nums, class_nums, bounds, max_iter, stop_time = None):

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
        likelihood_velocity = np.random.uniform(bounds[0],bounds[1], size = (feature_nums, class_nums, 10)) * 0.1
        velocities.append((prior_veclocity, likelihood_velocity))

    # 轉為 numpy 結構 (object dtype，支援 tuple 結構)
    positions = np.array(positions, dtype = object)
    velocities = np.array(velocities, dtype = object)

    # 計算每個粒子的初始適應值
    fitness = np.array([cal_accuracy_fitness(X, y, particle_prior, particle_likelihood) for (particle_prior, particle_likelihood) in positions])
    personal_best_positions = copy.deepcopy(positions)  # 初始化個體最佳位置、適應值
    personal_best_fitness = fitness

    best_fitness_index = np.argmax(fitness)  # 求出最佳適應值的粒子的索引
    global_best_positions = personal_best_positions[best_fitness_index]     # 初始化群體最佳位置與適應值
    global_best_fitness = personal_best_fitness[best_fitness_index]

     # 設定權重、速度因子 (自適應參數)
    w = 0.9
    c1 = 2 
    c2 = 2   
    m = 0   # 停滯次數 (超過 m 次群體最佳位置都不變，停止)
    stop_iter = max_iter
    stop_iteration = []
    # 只要不超過迭代次數就一直跑

    for i in range(max_iter):
        # 更新每個粒子
        for j in range(num_particles):
            prior, likelihood = positions[j]
            pbest_prior, pbest_likelihood = personal_best_positions[j]
            gbest_prior, gbest_likelihood = global_best_positions

            r1, r2 = np.random.random(size = 2)  # 產生兩個 [0,1] 的隨機值

            # 更新先驗機率的速度
            velocities[j][0] = w * velocities[j][0] + c1 * r1 * (pbest_prior - prior) + c2 * r2 * (gbest_prior - prior)
            # 更新似然機率的速度，velocities[j][1][feature] 代表第 j 個粒子的似然機率中的 feature 的速度
            velocities[j][1] = w * velocities[j][1] + c1 * r1 * (pbest_likelihood - likelihood) + c2 * r2 * (gbest_likelihood - likelihood)
     
            # 更新粒子的先驗機率
            prior_j = prior + velocities[j][0]  # prior_j 包含所有類別的機率值
            # 避免超出空間
            prior_j[prior_j < bounds[0]] = bounds[0]
            prior_j[prior_j > bounds[1]] = bounds[1]    
            
            # 更新粒子的似然機率，每個 feature 都要更新
            likelihood_j = likelihood + velocities[j][1]
            # 避免超出空間
            for index in range(len(likelihood)):
                likelihood_j[index][likelihood_j[index] < bounds[0]] = bounds[0]
                likelihood_j[index][likelihood_j[index] > bounds[1]] = bounds[1]
            # 對機率正規化

            # 簡化並正確正規化 likelihood_j
            for f in range(len(likelihood_j)):
                for c in range(len(likelihood_j[f])):
                    if likelihood_j[f][c].sum() != 0:
                        likelihood_j[f][c] = likelihood_j[f][c] / likelihood_j[f][c].sum()
          
            # 更新粒子位置
            positions[j] = (prior_j, likelihood_j)

            new_fitness_j = cal_accuracy_fitness(X, y, prior_j, likelihood_j)

            if new_fitness_j > personal_best_fitness[j]:
                personal_best_fitness[j] = new_fitness_j
                personal_best_positions[j] = (prior_j, likelihood_j)

        # 找出目前最佳群體適應值
        current_best_index = np.argmax(personal_best_fitness)
        current_best_fitness = personal_best_fitness[current_best_index]

        # 若目前最佳群體適應值有大於歷史最佳群體適應值，更新
        if current_best_fitness > global_best_fitness:
            global_best_fitness = current_best_fitness
            global_best_positions = personal_best_positions[current_best_index]
            m = 0   # 群體最佳值變動，重置停滯次數
        else:
            m += 1  # 停滯次數加 1

        # 超過一定次數群體最佳值都不變，代表已收斂，停止粒子群位置更新
        if m > stop_time:
            stop_iter = i + 1
            stop_iteration.append(stop_iter)
            break

    return global_best_positions[0], global_best_positions[1], global_best_fitness, stop_iteration  # 返回最佳粒子的先驗機率與概似機率

def cv_with_PSO_model(X, y, feature_nums, class_nums, k, num_particles, bounds, max_iter, stop_time = None):

    # 分割 fold 訓練與測試
    kf = KFold(n_splits = k, shuffle = True, random_state = 42)    

    fold_test_accuracies = [[] for _ in range(k)]       # 儲存每個 fold 做 test set 的預測準確率
    fold_base_models = [[] for _ in range(k)]  # 儲存每個 fold 中通過門檻的模型的先驗機率

    # 儲存每個 fold 的 test set，做為之後集成模型預測使用
    fold_X_test = [[] for _ in range(k)]  
    fold_y_test = [[] for _ in range(k)]  

    start_time = time.time()
    for fold, (train_index, test_index) in enumerate(kf.split(X)):  # kf.split(X) 返回訓練及測試樣本的索引
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        fold_X_test[fold] = X_test
        fold_y_test[fold] = y_test

        base_models = 0
        # 生成 25 個模型
        with tqdm(total = 25, desc = f"Fold {fold+1} - Building Models") as pbar:
            while base_models < 25:
                prior, likelihood, global_best_fitness, stop_iteration = PSO(X, y, num_particles, feature_nums, class_nums, bounds, max_iter, stop_time)
                ave_stop_iteration = np.mean(stop_iteration)
                pbar.set_postfix_str(f"ave stop iter = {ave_stop_iteration}")  # 不會打亂進度列
                fold_base_models[fold].append((prior, likelihood))
                base_models += 1
                pbar.update(1)
    end_time = time.time()
    exec_time = end_time - start_time
    log_path = "log/PSO_accuracies_log.txt"
    write_log(log_path, f"隨機生成基本模型執行時間: {exec_time:.4f} 秒")
    print(f"隨機生成基本模型執行時間: {exec_time:.4f} 秒")  # 輸出總執行時間
        # 記錄每個 fold 做 test data 準確率
    for fold in range(k):
        y_test = fold_y_test[fold]
        X_test = fold_X_test[fold]
        fold_predictions = np.zeros((len(y_test),len(fold_base_models[fold])))
        print(fold_predictions.shape)

        for i, (prior, likelihood) in enumerate(fold_base_models[fold]):
            predictions = prediction(X_test, prior, likelihood)
            fold_predictions[:,i] = predictions

        fold_predictions = np.apply_along_axis(
            lambda x : np.bincount(x.astype(int)).argmax(),
            axis = 1,
            arr = fold_predictions
        )
        fold_accuracy = np.mean(y_test == fold_predictions)
        fold_test_accuracies[fold] = fold_accuracy
   
    return fold_test_accuracies, fold_base_models

# 建立開日誌檔後記錄 log 內容
def write_log(log_path,msg):
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"{msg}\n")

if __name__ == "__main__":
    np.random.seed(42)   # 設定隨機種子
    # 建立 log 日誌儲存資訊
    log_path = "log/PSO_accuracies_log.txt"
    k = 5    # k 折交叉驗證次數
    base_dir = Path.cwd()             
    parent_path = base_dir.parent   # 取得上層路徑
    datasets = ["Algerian","Banknote","Climate","Diabetes","Electrical","German"]
    for dataset in datasets:
        data_path = os.path.join(parent_path, "datasets", "離散化資料集","二類別",f"{dataset}.csv")
        df = pd.read_csv(data_path)
        
        target_column = 'class'
        X = df.drop(columns = [target_column]).values
        y = df[target_column].values
        feature_nums = len(X[0])
        class_nums = len(set(y))
        bounds = (1e-10, 1)
        num_particles = 50  # 粒子數
        max_iter = 100
        stop_time = 5

        # 使用 PSO 生成基本模型
        fold_test_accuracies, fold_base_models = cv_with_PSO_model(X, y, feature_nums, class_nums, k, num_particles, bounds, max_iter, stop_time)

        # 記錄 five-fold 裡共 125 個基本模型的 c1 機率
        five_fold_prior_c1 = []
        for fold in fold_base_models:
            for i, (prior, likelihood) in enumerate(fold):
                five_fold_prior_c1.append(prior[0])

            # 畫垂直 Boxplot 圖
         # 畫垂直 Boxplot 圖
        plt.figure(figsize = (5, 8))
        sns.boxplot(y = five_fold_prior_c1, color = "skyblue")  # 使用 y= 垂直顯示

        plt.title(f"Class 1 Probabilities of {dataset} PSO")
        plt.ylabel("Probability for Class 1")

        # 刻度精細化
        plt.yticks(np.arange(0, 1.1, 0.1))

        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig(os.path.join(parent_path, "charts", f"{dataset}_PSO_prior_c1_boxplot.jpg"))
        plt.close()

        avg_accuracy = np.mean(fold_test_accuracies)
        msg = f"{dataset} TRENB 五折平均準確率: {avg_accuracy}\n"
        msg += "----------------------------------------"
        write_log(log_path, msg)
