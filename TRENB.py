import numpy as np
import pandas as pd
from collections import Counter,defaultdict
from sklearn.model_selection import KFold
import os, time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

def random_naive_model(feature_nums, class_nums):
    # 使用 dirichlet() 生成隨機的先驗機率，其總和為 1
    random_prior = np.random.dirichlet(np.ones(class_nums))
    # 隨機生成的 likelihood
    random_likelihood = np.random.dirichlet(np.ones(10), size = (feature_nums, class_nums))
    return random_prior,random_likelihood

# 根據隨機生成的機率，預測樣本類別
def prediction(X, prior, likelihood):
    predictions = []

    N = len(X)
    F, C, V = likelihood.shape
    # 預測每個樣本
    posterior = np.tile(prior, (N, 1))   # 形成一個 (N, C) 的矩陣

    for f in range(F):
        posterior *= likelihood[f][:, X[:,f]].T
    predictions = np.argmax(posterior, axis = 1)
    return np.array(predictions)

# 用原始資料集不斷產生模型計算準確率，挑出門檻
def generate_ensemble_threshold(X, y, feature_nums, class_nums, threshold_nums, models_nums):
    threshold_accuracy = np.zeros(threshold_nums)  # 儲存 threshold_nums 次最高的準確率
    
    for iter in range(threshold_nums):
        models_accuracy = np.zeros(models_nums)  # 儲存單一迭代中，models_nums 個模型的準確率
        for model in range(models_nums):
            prior,likelihood = random_naive_model(feature_nums, class_nums)
            predictions = prediction(X, prior, likelihood)
            # 計算準確率 (正確預測的樣本數 / 總樣本數)
            accuracy = np.mean(predictions == y)
            models_accuracy[model] = accuracy
        
        # 儲存這次迭代中最高的準確率
        threshold_accuracy[iter] = np.max(models_accuracy)
    return np.mean(threshold_accuracy)   # 將所有迭代產生的最高準確率平均做為門檻

def cv_with_random_model(X, y, feature_nums, class_nums, threshold_nums, models_nums, k):
    threshold = generate_ensemble_threshold(X, y, feature_nums, class_nums, threshold_nums, models_nums)
    print(f"threshold = {threshold}")
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
        with tqdm(total = 25, desc = f"Fold {fold + 1} - Building Models") as pbar:
            while base_models < 25:
                prior, likelihood = random_naive_model(feature_nums, class_nums)
                predictions = prediction(X_train, prior, likelihood)
                accuracy = float(np.mean(predictions == y_train))
                if accuracy >= threshold:   # 此次模型準確率超過門檻
                    fold_base_models[fold].append((prior, likelihood))
                    base_models += 1
                    pbar.update(1)
    end_time = time.time()
    exec_time = end_time - start_time
    log_path = "log/TRENB_accuracies_log.txt"
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
    log_path = "log/TRENB_accuracies_log.txt"
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

        # 計算門檻時迭代 20 次,每次跑 1000 個模型，最後在生成 125 (一個 fold 25 個)個基本模型
        fold_test_accuracies, fold_base_models = cv_with_random_model(X, y, feature_nums, class_nums, 20, 1000, k)

        # 記錄 five-fold 裡共 125 個基本模型的 c1 機率
        five_fold_prior_c1 = []
        for fold in fold_base_models:
            for i, (prior, likelihood) in enumerate(fold):
                five_fold_prior_c1.append(prior[0])
    
        # 畫垂直 Boxplot 圖
        plt.figure(figsize = (5, 8))
        sns.boxplot(y = five_fold_prior_c1, color = "skyblue")  # 使用 y= 垂直顯示

        plt.title(f"Class 1 Probabilities of {dataset} TRENB")
        plt.ylabel("Probability for Class 1")

        # 刻度精細化
        plt.yticks(np.arange(0, 1.1, 0.1))

        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig(os.path.join(parent_path, "charts", f"{dataset}_TRENB_prior_c1_boxplot.jpg"))
        plt.close()

        avg_accuracy = np.mean(fold_test_accuracies)
        msg = f"{dataset} TRENB 五折平均準確率: {avg_accuracy}\n"
        msg += "----------------------------------------"
        write_log(log_path, msg)
