import pandas as pd
import numpy as np
import traceback
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy import stats
import os
import random
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import precision_score, recall_score, f1_score
import math
import csv

folder_path='D:/Desktop1/cuda/glass/'

# 文件集合
filename='glass.csv'

# 基因名称
gene_list = ['gene1', 'gene2', 'gene3',
             'gene4','gene5', 'gene6',
             'gene7', 'gene8', 'gene9']


def data_split(folder_path, filename):
    # 读取数据
    data = pd.read_csv(folder_path + filename)
    header = data.columns.tolist()  # 获取列名

    # 训练集
    train_glass = pd.DataFrame(columns=header)

    # 测试集
    test_glass = pd.DataFrame(columns=header)

    Tag = data['tag'].tolist()

    glass1 = []
    glass2 = []

    for i in range(len(Tag)):
        if Tag[i] == 1:
            glass1.append(i)
        elif Tag[i]==2:
            glass2.append(i)

    # 分配训练集
    for i in range(math.floor(len(glass1) * 0.8)):
        train_glass = pd.concat([train_glass, data.iloc[[glass1[i]]]])  # 使用iloc选取一行

    for i in range(math.floor(len(glass2) * 0.8)):
        train_glass = pd.concat([train_glass, data.iloc[[glass2[i]]]])

    # 分配测试集
    for i in range(math.floor(len(glass1) * 0.2)):
        test_glass = pd.concat([test_glass, data.iloc[[glass1[i + math.floor(len(glass1) * 0.8)]]]])

    for i in range(math.floor(len(glass2) * 0.2)):
        test_glass = pd.concat([test_glass, data.iloc[[glass2[i + math.floor(len(glass2) * 0.8)]]]])

    # 保存文件
    train_glass.to_csv('D:/Desktop1/cuda/glass/'+'train_glass.csv', index=False)
    test_glass.to_csv('D:/Desktop1/cuda/glass/'+'test_glass.csv', index=False)

def normalize_all_columns(folder_path, filename):
    # 读取数据
    data = pd.read_csv(folder_path + filename)

    # 获取所有的数值型列名
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

    # 数据标准化
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data[numeric_columns])

    # 将标准化后的数据转换回 DataFrame，并保留原有的列名
    normalized_df = pd.DataFrame(normalized_data, columns=numeric_columns)

    # 将非数值型列（如果有）添加回 DataFrame
    non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_columns:
        normalized_df = pd.concat([normalized_df, data[non_numeric_columns]], axis=1)

    return normalized_df
# 求解种群中每个个体的基因值
def initial_groups(folder_path, filename, target_tag, gene_list):
    #读取数据
    data=normalize_all_columns(folder_path, filename)
    # data=pd.read_csv(folder_path+filename)
    #基因1数值
    gene1=data['gene1'].tolist()
    #基因2数值
    gene2=data['gene2'].tolist()
    #基因3数值
    gene3=data['gene3'].tolist()
    #基因4数值
    gene4=data['gene4'].tolist()
    #基因5数值
    gene5=data['gene5'].tolist()
    #基因6数值
    gene6=data['gene6'].tolist()
    #基因7数值
    gene7=data['gene7'].tolist()
    #基因8数值
    gene8=data['gene8'].tolist()
    #基因9数值
    gene9=data['gene9'].tolist()
    #目标种群
    label=pd.read_csv(folder_path+filename)
    tags=label['tag'].tolist()
    initial_group=[]
    id = 0
    for i in range(len(tags)):
        if tags[i]==target_tag:
            initial_group.append(np.zeros([1, len(gene_list)]))
            initial_group[id][0][0]=gene1[i]
            initial_group[id][0][1]=gene2[i]
            initial_group[id][0][2]=gene3[i]
            initial_group[id][0][3]=gene4[i]
            initial_group[id][0][4] = gene5[i]
            initial_group[id][0][5] = gene6[i]
            initial_group[id][0][6] = gene7[i]
            initial_group[id][0][7] = gene8[i]
            initial_group[id][0][8] = gene9[i]
            id+=1

    return initial_group

def normal_groups(folder_path, filename, target_tag, gene_list):
    #读取数据
    data=normalize_all_columns(folder_path, filename)
    # data=pd.read_csv(folder_path+filename)
    #基因1数值
    gene1=data['gene1'].tolist()
    #基因2数值
    gene2=data['gene2'].tolist()
    #基因3数值
    gene3=data['gene3'].tolist()
    #基因4数值
    gene4=data['gene4'].tolist()
    #基因5数值
    gene5=data['gene5'].tolist()
    #基因6数值
    gene6=data['gene6'].tolist()
    #基因7数值
    gene7=data['gene7'].tolist()
    #基因8数值
    gene8=data['gene8'].tolist()
    #基因9数值
    gene9=data['gene9'].tolist()
    #目标种群
    label=pd.read_csv(folder_path+filename)
    tags=label['tag'].tolist()
    normal_group=[]
    id = 0
    for i in range(len(tags)):
        if tags[i] == target_tag:
            normal_group.append(np.zeros([1, len(gene_list)]))
            normal_group[id][0][0]=gene1[i]
            normal_group[id][0][1]=gene2[i]
            normal_group[id][0][2]=gene3[i]
            normal_group[id][0][3]=gene4[i]
            normal_group[id][0][4] = gene5[i]
            normal_group[id][0][5] = gene6[i]
            normal_group[id][0][6] = gene7[i]
            normal_group[id][0][7] = gene8[i]
            normal_group[id][0][8] = gene9[i]
            id+=1

    return normal_group

def fit(normal_group, initial_group, gene_list):
    fitness = np.zeros([len(initial_group), 1])
    for i in range(len(initial_group)):
        for j in range(len(normal_group)):
            # 求解欧氏距离
            fitness[i] += np.sqrt(np.sum(np.square(initial_group[i] - normal_group[j])))
        # 求解加权平均值
        fitness[i] = fitness[i] / (len(normal_group) * len(gene_list))
    # for i in range(len(fitness)):
    #     fitness[i] =1- fitness[i] / np.sum(fitness)

    return fitness

#个体选择函数，使用轮盘赌算法选择优良的个体
def selection(population, fitness):
    total_fitness=sum(fitness)
    probabilities = [f / total_fitness for f in fitness]
    parent1, parent2=random.choices(population, weights=probabilities, k=2)
    return parent1, parent2

#交叉操作
def crossover(parent1, parent2, gene_list):
    child1=parent1.copy()
    child2=parent2.copy()
    point=random.randint(1, len(parent1[0])-1)
    child1[0][point:], child2[0][point:] = child2[0][point:], child1[0][point:]
    return child1, child2

#变异操作
def mutation(individual, gene_list, direct):
    uniform=random.uniform
    for col in range(len(gene_list)):
        if random.random()<0.5:
                individual[0][col]+=uniform(-0.1*direct, -0.0*direct)
    return individual

#正常种群淘汰
def normal_eliminate(normal_group, gene_list):
    temp=[]
    idx=[]
    for i in range(len(gene_list)):
        temp.append([])
        for j in range(len(normal_group)):
            temp[i].append(normal_group[j][0][i])
        #找到标准差
        med=np.median(temp[i])
        ma=stats.median_abs_deviation(temp[i])
        for k in range(len(temp[i])):
            z=(0.6745*(temp[i][k]-med))/np.median(ma)
            if np.abs(z)>3:
                idx.append(k)
    idx=sorted(list(set(idx)))
    print(idx)
    for i in range(len(idx)):
        del normal_group[idx[i]-i]
    return normal_group

#异常种群淘汰
def initial_eliminate(initial_group, gene_list):
    temp=[]
    idx=[]
    for i in range(len(gene_list)):
        temp.append([])
        for j in range(len(initial_group)):
            temp[i].append(initial_group[j][0][i])
        #找到标准差
        med=np.median(temp[i])
        ma=stats.median_abs_deviation(temp[i])
        for k in range(len(temp[i])):
            z=(0.6745*(temp[i][k]-med))/np.median(ma)
            if np.abs(z)>3:
                idx.append(k)
    idx=sorted(list(set(idx)))
    print(idx)
    for i in range(len(idx)):
        del initial_group[idx[i]-i]
    return initial_group


# 主遗传算法流程

def genetic_algorithm(initial_group, normal_group, gene_list, generations):
    #正向进化
    population = initial_group.copy()
    new_population =initial_group.copy()
    # 记录适应都变化
    fitness_record = []
    append=fitness_record.append
    extend=new_population.extend
    for generation in range(generations):
        print('-----------正向进化', generation, '代---------------------')
        fitness = fit(normal_group, population, gene_list)
        parent1, parent2 = selection(population, fitness)
        child1, child2 = crossover(parent1, parent2, gene_list)
        child1 = mutation(child1, gene_list,1)
        child2 = mutation(child2, gene_list, 1)
        extend([child1, child2])
        append(np.mean(fit(normal_group, new_population, gene_list)))

    population = new_population

    #反向进化
    population1=normal_group.copy()
    new_population1=normal_group.copy()

    #记录反向进化适应度变化
    fitness_record1=[]
    append=fitness_record1.append
    extend=new_population1.extend

    for generation in range(generations):
        print('-----------反向进化', generation, '代---------------------')
        fitness = fit(initial_group, population1, gene_list)
        parent1, parent2 = selection(population1, fitness)
        child1, child2 = crossover(parent1, parent2, gene_list)
        child1 = mutation(child1, gene_list, -1)
        child2 = mutation(child2, gene_list, -1)
        extend([child1, child2])
        append(np.mean(fit(initial_group, new_population1, gene_list)))

    population1 = new_population1




    return population, population1, fitness_record, fitness_record1

# 绘制适应度进化过程的函数
def plot_fitness_evolution(fitness_record):
    generations = list(range(1, len(fitness_record) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(generations, fitness_record, label='Average Fitness', color='blue')
    plt.title('Evolution of Average Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.legend()
    plt.grid(True)
    plt.savefig('fitness1.png')


def feature_visualization(normal_group_path, population_path, normal_max_range,  abnormal_max_range,  gene_id, generations):
    # Load the data from the .npz files
    loaded_data = np.load(normal_group_path)
    normal_group = [loaded_data[f'arr_{i}'] for i in range(len(loaded_data.files))]

    loaded_data = np.load(population_path)
    population = [loaded_data[f'arr_{i}'] for i in range(len(loaded_data.files))]
    #正常数据
    normal=[]
    for i in range(len(normal_group)):
        normal.append(normal_group[i][0])

    normal_target_feature=[]
    for i in range(len(normal)):
        normal_target_feature.append(normal[i][gene_id])


    #异常数据
    abnormal=[]
    for i in range(len(population)):                                                                                                                                                     
        abnormal.append(population[i][0])
    abnormal_target_feature=[]
    for i in range(len(abnormal)):
        abnormal_target_feature.append(abnormal[i][gene_id])
    

    if normal_max_range[gene_id] > abnormal_max_range[gene_id]:
        marked_value = normal_max_range[gene_id]
    else:
        marked_value = abnormal_max_range[gene_id]
    
    marked_value=round(marked_value, 2)
 
    #定义图像和三维格式坐标轴
    fig=plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)
    ax.scatter3D(range(len(abnormal_target_feature)), abnormal_target_feature, np.zeros([1, len(abnormal_target_feature)]), color='red', label='abnormal')
    ax.scatter3D(range(len(normal_target_feature)), normal_target_feature, np.zeros([1, len(normal_target_feature)]), color='green', label='normal')
    ax.axhline(y=marked_value, color='blue', linestyle='None', linewidth=3, label=f'Threshold: {marked_value}')
    # ax.text2D(0.5, 0.8, f'Threshold: {marked_value}', transform=ax.transAxes, fontsize=15, ha='center')
    ax.set_xlabel('num', fontsize=15)  # X-axis label
    ax.set_ylabel('value', fontsize=15, labelpad=15)  # Y-axis label
    ax.tick_params(axis='x', which='major', labelsize=15)  # X轴数值字体大小
    ax.tick_params(axis='y', which='major', labelsize=15)  # Y轴数值字体大小
    ax.tick_params(axis='z', which='major', labelsize=15, pad=5)  # Z轴数值字体大小
    # ax.text(0.5, 0.01, s='(a) Wine Dataset', ha='center', fontsize=15)
    ax.legend(fontsize=15, loc='upper right', bbox_to_anchor=(0.85, 0.85))
    plt.savefig('D:/Desktop1/cuda/glass/'+f'{generations}.png', dpi=600, bbox_inches='tight') # 紧凑边框
    # plt.show()
    plt.close()

def statics_range(population1, population, gene_list):
    #最大值范围
    normal_max_range=[]
    abnormal_max_range=[]
    #最小值范围
    normal_min_range=[]
    abnormal_min_range=[]

    for i in range(len(gene_list)):
        temp=[]
        for j in range(len(population1)):
            temp.append(population1[j][0][i])
        normal_max_range.append(np.max(np.array(temp)))
        normal_min_range.append(np.min(np.array(temp)))

    for i in range(len(gene_list)):
        temp=[]
        for j in range(len(population)):
            temp.append(population[j][0][i])
        abnormal_max_range.append(np.max(np.array(temp)))
        abnormal_min_range.append(np.min(np.array(temp)))
    return normal_max_range, normal_min_range, abnormal_max_range, abnormal_min_range

def feature_2D(population, population1, gene_list):
    #正常数据
    normal=[]
    for i in range(len(population1)):
        normal.append(population1[i][0])
    #异常数据
    abnormal=[]
    for i in range(len(population)):
        abnormal.append(population[i][0])
    #正常松弛变量
    theta=[]
    #异常松弛变量
    for i in range(len(gene_list)):
        temp=[]
        for j in range(len(normal)):
            temp.append(normal[j][i])
        theta.append(np.std(temp))
    theta1=[]
    for i in range(len(gene_list)):
        temp=[]
        for j in range(len(abnormal)):
            temp.append(abnormal[j][i])
        theta1.append(np.std(temp))

        # plt.scatter(range(len(temp)), sorted(temp))
    
    
    return theta, theta1


def train_svm_classifier(population, population1,  id, parameters, test_size=0.2, random_state=42):

    # 将种群转换为numpy数组
    abnormal_data = np.array([ind[0][id] for ind in population]).reshape(-1, 1)  # 调整为二维数组
    normal_data = np.array([ind[0][id] for ind in population1]).reshape(-1, 1)   # 调整为二维数组


    # 创建标签(1表示异常，0表示正常)
    abnormal_labels = np.ones(abnormal_data.shape[0])
    normal_labels = np.zeros(normal_data.shape[0])



    # 合并数据
    X = np.vstack((abnormal_data, normal_data))
    y = np.hstack((abnormal_labels, normal_labels))

    #合并原始数据

    # 将数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 初始化和训练SVM分类器
    clf = SVC(kernel='rbf', C=1.0, gamma='auto', probability=True)
    clf.fit(X_train, y_train)

  

    # Create grid for plotting
    x_min = min(X_train)[0] - 1
    x_max = max(X_train)[0] + 1
    xx = np.linspace(x_min, x_max, 500).reshape(-1, 1)

    # Get decision function values
    decision_values = clf.decision_function(xx)

        # 在SVM初始化部分添加gamma计算
    n_features = X_train.shape[1]  # 获取特征维度
    auto_gamma = 1.0 / n_features

    # Find decision boundary (where decision function crosses zero)
    zero_crossings = np.where(np.diff(np.sign(decision_values)))[0]
    boundaries = [xx[crossing][0] for crossing in zero_crossings]

    # 提取正常和异常数据用于绘图
    normal_values = normal_data
    abnormal_values = abnormal_data

    plt.figure(figsize=(12, 6))

    # 获取主坐标轴对象
    ax1 = plt.gca()

     # Plot histograms
    ax1.hist(normal_values, bins=30, alpha=0.3, color='blue', label='Normal Population')
    ax1.hist(abnormal_values, bins=30, alpha=0.3, color='red', label='Abnormal Population')

    # 新增决策函数曲线绘制
    ax2 = plt.gca().twinx()  # 创建第二个y轴

    # 显式设置主坐标轴标签
    ax1.set_ylabel('Frequency', fontsize=12)
    ax2.set_ylabel('Decision Value', fontsize=12)

    # 调整布局边距
    plt.subplots_adjust(left=0.15, right=0.85)  # 为双y轴调整左右边距

    ax2.plot(xx, decision_values, 
           color='green', 
           linewidth=2.5, 
           linestyle='-',
           label='Decision Function')
    
    # Plot decision boundary
    for i, boundary in enumerate(boundaries):
        plt.axvline(x=boundary, color='purple', linestyle='--',
                  linewidth=2, label='Boundary value' if i == 0 else None)

    global gene_list  # 确保可以访问全局变量 gene_list
    gene_idx = id  # 用 id 作为 gene_idx
    plt.xlabel(f'{gene_list[gene_idx]} Expression Level', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    ax2.set_ylabel('Decision Value', fontsize=12)
    # 合并图例
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, fontsize=10, loc='upper left')

    plt.title(f'Distribution of {gene_list[gene_idx]} with Decision Boundary', fontsize=14)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)

    # Save figure
    plt.savefig(f'D:/Desktop1/会议/glass/{parameters}/auto_RBF_gene_{gene_list[gene_idx]}_decision_boundary.png', 
               dpi=300, bbox_inches='tight')
    plt.close()

    # # Print full decision function information
    # print(f"\nSVM Decision Function for {gene_list[gene_idx]}:")
    # print("========================================")
    # print(f"Kernel: RBF (gamma={clf.gamma})")  # 使用 clf 代替未定义的 svm
    # # print(f"Number of support vectors: {len(clf.support_vectors_)}")  # 使用 clf 代替未定义的 svm
    # print(f"Intercept (b): {clf.intercept_[0]:.4f}")  # 使用 clf 代替未定义的 svm
    # print("\nDecision function formula:")
    # print("f(x) = sum(α_i * y_i * K(x_i, x)) + b")
    # print("where K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)")

    # # Print support vectors and coefficients
    # print("\nSupport vectors and coefficients:")
    # for i, (sv, coef) in enumerate(zip(clf.support_vectors_, clf.dual_coef_.T)):  # 使用 clf 代替未定义的 svm
    #     print(f"SV {i+1}: x={sv[0]:.4f}, α*y={coef[0]:.4f}")

    # 在测试集上评估
    y_pred = clf.predict(X_test)
    print("SVM分类报告:")
    print(classification_report(y_test, y_pred))
    # 计算多个指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # 记录到CSV文件部分
    with open(f'D:/Desktop1/会议/glass/{parameters}/auto_RBF_accuracy_records.csv', 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['gene_id', 'accuracy', 'precision', 'recall', 'f1', 'decision_function']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if csvfile.tell() == 0:
            writer.writeheader()
            
        writer.writerow({
            'gene_id': id,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'decision_function': f"f(x) = sum(α_i*y_i*exp(-{auto_gamma:.4f}*||x_i-x||²)) + {clf.intercept_[0]:.4f}"
        })

    return clf, X_test, y_test


# data_split(folder_path, filename)

if os.path.exists('D:/Desktop1/cuda/glass/initial_group.npz') == 0:
    initial_group = initial_groups(folder_path, 'train_glass.csv', 2, gene_list)
    np.savez('D:/Desktop1/cuda/glass/initial_group.npz', *initial_group)
else:
    loaded_data=np.load('D:/Desktop1/cuda/glass/initial_group.npz')
    initial_group = [loaded_data[f'arr_{i}'] for i in range(len(loaded_data.files))]


if os.path.exists('D:/Desktop1/cuda/glass/normal_group.npz') == 0:
    normal_group = normal_groups(folder_path, 'train_glass.csv',1,  gene_list)
    np.savez('D:/Desktop1/cuda/glass/normal_group.npz', *normal_group)
else:
    loaded_data=np.load('D:/Desktop1/cuda/glass/normal_group.npz')
    normal_group = [loaded_data[f'arr_{i}'] for i in range(len(loaded_data.files))]


# initial_group=initial_eliminate(initial_group, gene_list)
# normal_group=normal_eliminate(normal_group, gene_list)

generations=[600]
# #
for i in range(len(generations)):
    population, population1, fitness_record, fitness_record1=genetic_algorithm(initial_group, normal_group, gene_list, generations[i])

    for id in range(len(gene_list)):
        clf, X_test, y_test=train_svm_classifier(population, population1, id,  generations[i], test_size=0.2, random_state=42)


