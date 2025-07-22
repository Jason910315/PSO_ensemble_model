import numpy as np
import pandas as pd
from collections import Counter,defaultdict
from sklearn.model_selection import KFold
import os, time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# 隨機生成 NB 的先驗機率、似然機率
def random_naive_model(df, feature_nums, class_nums):
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

# 用原始資料集不斷產生模型計算準確率，挑出門檻
def generate_ensemble_threshold(df, target_column, feature_nums, class_nums, threshold_nums, models_nums):
    # 取出特徵與類別
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values

    threshold_accuracy = np.zeros(threshold_nums)  # 儲存 threshold_nums 次最高的準確率
    

    for iter in range(threshold_nums):
        models_accuracy = np.zeros(models_nums)  # 儲存單一迭代中，models_nums 個模型的準確率
        for model in range(models_nums):
            prior,likelihood = random_naive_model(df, feature_nums, class_nums)
            predictions = prediction(X, prior, likelihood)
            # 計算準確率 (正確預測的樣本數 / 總樣本數)
            accuracy = np.mean(predictions == y)
            models_accuracy[model] = accuracy
        
        # 儲存這次迭代中最高的準確率
        threshold_accuracy[iter] = np.max(models_accuracy)
    return np.mean(threshold_accuracy)   # 將所有迭代產生的最高準確率平均做為門檻

def cv_with_random_model(X, y, feature_nums, class_nums, threshold_nums, models_nums, k):
    threshold = generate_ensemble_threshold(df, target_column, feature_nums, class_nums, threshold_nums, models_nums)

    # 分割 fold 訓練與測試
    kf = KFold(n_splits = k, shuffle = True, random_state = 42)    

    fold_accuracies = [[] for _ in range(k)]       # 儲存每個 fold 通過門檻的模型預測準確率
    fold_nb_models_prior = [[] for _ in range(k)]  # 儲存每個 fold 中通過門檻的模型的先驗機率
    fold_nb_models_likelihood = [[] for _ in range(k)]  # 儲存每個 fold 中通過門檻的模型的似然機率

    start_time = time.time()
    for fold, (train_index, test_index) in enumerate(kf.split(X)):  # kf.split(X) 返回訓練及測試樣本的索引
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        base_models = 0
        # 生成 25 個模型
        with tqdm(total = 25, desc = f"Fold {fold+1} - Building Models") as pbar:
            while base_models < 25:
                prior, likelihood = random_naive_model(df, feature_nums, class_nums)
                predictions = prediction(X_train, prior, likelihood)
                accuracy = float(np.mean(predictions == y_train))
                if accuracy >= threshold:   # 此次模型準確率超過門檻
                    fold_accuracies[fold].append(accuracy)
                    fold_nb_models_prior[fold].append(prior)
                    fold_nb_models_likelihood[fold].append(likelihood)
                    base_models += 1
                    pbar.update(1)
    end_time = time.time()
    exec_time = end_time - start_time
    print(f"隨機生成基本模型執行時間: {exec_time:.4f} 秒")  # 輸出總執行時間
    return fold_accuracies, fold_nb_models_prior, fold_nb_models_likelihood


if __name__ == "__main__":
    k = 5    # k 折交叉驗證次數
    base_dir = Path.cwd()             
    data_path = os.path.join(base_dir, "datasets", "離散化資料集","二類別","Heart Failure.csv")
    df = pd.read_csv(data_path)

    target_column = 'class'
    X = df.drop(columns = [target_column]).values
    y = df[target_column].values
    feature_nums = len(X[0])
    class_nums = len(set(y))

    # 計算門檻時迭代 20 次,每次跑 1000 個模型，最後在生成 125 (一個 fold 25 個)個基本模型
    fold_accuracies, fold_nb_models_prior, fold_nb_models_likelihood = cv_with_random_model(X, y, feature_nums, class_nums, 20, 1000, k)

    # 記錄 five-fold 裡共 125 個基本模型的 c1 機率
    five_fold_prior_c1 = []
    for fold in fold_nb_models_prior:
        for prior_c1,prior_c2 in fold:
            five_fold_prior_c1.append(prior_c1)
    
     # 畫 KDE 機率密度分布圖
    plt.figure(figsize=(8, 5))
    sns.kdeplot(five_fold_prior_c1, fill=True, color="skyblue", linewidth=2)
    plt.title("Kernel Density Estimation of Class 1 Probabilities")
    plt.xlabel("Probability for Class 1")
    plt.ylabel("Density")
    plt.grid(True)
    plt.savefig(os.path.join(base_dir, "charts", "TRENB_prior_c1.jpg"))
    plt.close()