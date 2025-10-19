import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy import stats
import os
import random
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
# import numba
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde
from sklearn.metrics import precision_score, recall_score, f1_score
import math
import csv
from keras.models import Model, Sequential
from keras.layers import Dense, LeakyReLU, Input
import tensorflow as tf
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  


filename= 'iris.csv'


gene_list = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']


def data_split( filename):
    
    data = pd.read_csv(filename)
    header = data.columns.tolist()  

    train_iris = pd.DataFrame(columns=header)

    test_iris = pd.DataFrame(columns=header)

    Species = data['Species'].tolist()

    setosa_id = []
    other_id = []

    for i in range(len(Species)):
        if Species[i] == 'setosa':
            setosa_id.append(i)
        else:
            other_id.append(i)

    for i in range(math.floor(len(setosa_id) * 0.8)):
        train_iris = pd.concat([train_iris, data.iloc[[setosa_id[i]]]])  # 使用iloc选取一行

    for i in range(math.floor(len(other_id) * 0.8)):
        train_iris = pd.concat([train_iris, data.iloc[[other_id[i]]]])


    for i in range(math.floor(len(setosa_id) * 0.2)):
        test_iris = pd.concat([test_iris, data.iloc[[setosa_id[i + math.floor(len(setosa_id) * 0.8)]]]])

    for i in range(math.floor(len(other_id) * 0.2)):
        test_iris = pd.concat([test_iris, data.iloc[[other_id[i + math.floor(len(other_id) * 0.8)]]]])

    train_iris.to_csv('train_iris.csv', index=False)
    test_iris.to_csv('test_iris.csv', index=False)



def initial_groups( filename, target_tag, gene_list):
 
    data=pd.read_csv(filename)
    
    gene1=data['Sepal.Length'].tolist()
    
    gene2=data['Sepal.Width'].tolist()
    
    gene3=data['Petal.Length'].tolist()
    
    gene4=data['Petal.Width'].tolist()
    
   

    tags=data['Species'].tolist()
    initial_group=[]
    id = 0
    for i in range(len(tags)):
        if tags[i] == target_tag:
            initial_group.append(np.zeros([1, len(gene_list)]))
            initial_group[id][0][0]=gene1[i]
            initial_group[id][0][1]=gene2[i]
            initial_group[id][0][2]=gene3[i]
            initial_group[id][0][3]=gene4[i]
        

    return initial_group

def normal_groups(filename, target_tag, gene_list):
    
    data=pd.read_csv(filename)

    gene1=data['Sepal.Length'].tolist()
    
    gene2=data['Sepal.Width'].tolist()
    
    gene3=data['Petal.Length'].tolist()

    gene4=data['Petal.Width'].tolist()
    
    tags=data['Species'].tolist()
    normal_group=[]
    id = 0
    for i in range(len(tags)):
        if tags[i] == target_tag:
            normal_group.append(np.zeros([1, len(gene_list)]))
            normal_group[id][0][0]=gene1[i]
            normal_group[id][0][1]=gene2[i]
            normal_group[id][0][2]=gene3[i]
            normal_group[id][0][3]=gene4[i]
    
            id+=1

    return normal_group

def fit(normal_group, initial_group, gene_list):
    fitness = np.zeros([len(initial_group), 1])
    for i in range(len(initial_group)):
        for j in range(len(normal_group)):
            
            fitness[i] += np.sqrt(np.sum(np.square(initial_group[i] - normal_group[j])))
        
        fitness[i] = fitness[i] / (len(normal_group) * len(gene_list))


    return fitness


def selection(population, fitness):
    total_fitness=sum(fitness)
    probabilities = [f / total_fitness for f in fitness]
    parent1, parent2=random.choices(population, weights=probabilities, k=2)
    return parent1, parent2


def crossover(parent1, parent2):
    child1=parent1.copy()
    child2=parent2.copy()
    point=random.randint(1, len(parent1[0])-1)
    child1[0][point:], child2[0][point:] = child2[0][point:], child1[0][point:]
    return child1, child2


def mutation(individual, gene_list, direct, rate):
    uniform=random.uniform
    for col in range(len(gene_list)):
        if random.random()<0.5:
                individual[0][col]+=uniform(rate*direct, 0)
    return individual



def build_generator(input_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim),
        LeakyReLU(alpha=0.2),
        Dense(64),
        LeakyReLU(alpha=0.2),
        Dense(input_dim, activation='tanh')
    ])
    return model

def build_discriminator(input_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim),
        LeakyReLU(alpha=0.2),
        Dense(64),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')
    ])
    return model

def pre_evaluate(population, discriminator):
    
    samples = np.array([ind[0] for ind in population])
    max_value, min_value=samples.max(), samples.min()
    samples=(samples-min_value)/(max_value-min_value)
    return np.min(discriminator.predict(samples, verbose=0))

class GAN():
    def __init__(self, input_dim, dp1, dp2, gp1, gp2):
        self.input_dim = input_dim
        self.generator = build_generator(input_dim)
        self.discriminator = build_discriminator(input_dim)
        self.dp1=dp1
        self.dp2=dp2
        self.gp1=gp1
        self.gp2=gp2
        

        self.discriminator.compile(loss='binary_crossentropy',
                                  optimizer=tf.keras.optimizers.Adam(self.dp1, self.dp2))
        
        z = Input(shape=(input_dim,))
        generated = self.generator(z)
        validity = self.discriminator(generated)
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy',
                             optimizer=tf.keras.optimizers.Adam(self.gp1, self.gp2))
        

def normal_eliminate(normal_group, discriminator, prob):
    

    real_samples = np.array([x[0] for x in normal_group])
    real_min_val, real_max_val = real_samples.min(), real_samples.max()
    real_samples = (real_samples - real_min_val) / (real_max_val - real_min_val)  


    probs = discriminator.predict(real_samples, verbose=0)
    print('-----------max_probs----------------')
    print(np.max(probs))
    print('-----------min_probs----------------')
    print(np.min(probs))
    keep_indices = np.where(probs.flatten() > prob)[0]  
    
    return [normal_group[i] for i in keep_indices]

def initial_eliminate(initial_group, discriminator, prob):

    real_samples = np.array([x[0] for x in initial_group])
    min_val, max_val = real_samples.min(), real_samples.max()
    real_samples = (real_samples - min_val) / (max_val - min_val) 
    

    probs = discriminator.predict(real_samples, verbose=0)
    print('-----------max_probs----------------')
    print(np.max(probs))
    print('-----------min_probs----------------')
    print(np.min(probs))
    keep_indices = np.where(probs.flatten() > prob)[0]  
    return [initial_group[i] for i in keep_indices]


def genetic_algorithm(initial_group, normal_group, gene_list, generations, rate):

    population = initial_group.copy()
    new_population =initial_group.copy()

    fitness_record = []
    append=fitness_record.append
    extend=new_population.extend
    for generation in range(generations):
        print('-----------forward', generation, '---------------------')
        fitness = fit(normal_group, population, gene_list)
        parent1, parent2 = selection(population, fitness)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutation(child1, gene_list, 1, rate)
        child2 = mutation(child2, gene_list, 1, rate)
        extend([child1, child2])
        append(np.mean(fit(normal_group, new_population, gene_list)))

    population = new_population

    population1=normal_group.copy()
    new_population1=normal_group.copy()


    fitness_record1=[]
    append=fitness_record1.append
    extend=new_population1.extend

    for generation in range(generations):
        print('-----------backward', generation, '---------------------')
        fitness = fit(initial_group, population1, gene_list)
        parent1, parent2 = selection(population1, fitness)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutation(child1, gene_list, -1, rate)
        child2 = mutation(child2, gene_list,-1, rate)
        extend([child1, child2])
        append(np.mean(fit(initial_group, new_population1, gene_list)))

    population1 = new_population1


    return population, population1, fitness_record, fitness_record1

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




def train_svm_classifier(population, population1,  id, rate, generations, test_size=0.2, random_state=42):
    
    abnormal_data = np.array([ind[0][id] for ind in population]).reshape(-1, 1)  
    normal_data = np.array([ind[0][id] for ind in population1]).reshape(-1, 1)   


   
    abnormal_labels = np.ones(abnormal_data.shape[0])
    normal_labels = np.zeros(normal_data.shape[0])

    
    X = np.vstack((abnormal_data, normal_data))
    y = np.hstack((abnormal_labels, normal_labels))

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

   
    clf = SVC(kernel='rbf', C=1.0, gamma='auto', probability=True)
    clf.fit(X_train, y_train)


    
    n_features = X_train.shape[1]  
    auto_gamma = 1.0 / n_features
    

    # Create grid for plotting
    x_min = min(X_train)[0] - 1
    x_max = max(X_train)[0] + 1
    xx = np.linspace(x_min, x_max, 500).reshape(-1, 1)

    # Get decision function values
    decision_values = clf.decision_function(xx)

    # Find decision boundary (where decision function crosses zero)
    zero_crossings = np.where(np.diff(np.sign(decision_values)))[0]
    boundaries = [xx[crossing][0] for crossing in zero_crossings]

    
    normal_values = normal_data
    abnormal_values = abnormal_data

    
    y_pred = clf.predict(X_test)
    
    print(classification_report(y_test, y_pred))
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    with open('GAN_T_RBF_accuracy_records.csv', 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['gene_id','generation',  'accuracy', 'precision', 'recall', 'f1', 'decision_function']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if csvfile.tell() == 0:
            writer.writeheader()
            
        writer.writerow({
            'gene_id': id,
            'generation': generations,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'decision_function': f"f(x) = sum(α_i*y_i*exp(-{auto_gamma:.4f}*||x_i-x||²)) + {clf.intercept_[0]:.4f}"
        })

    return clf, X_test, y_test


def train_gan(gan, real_samples, noise_source, epochs, batch_size=32, save_path='gan_loss_curve.png'):
    
    real_min, real_max = real_samples.min(), real_samples.max()
    real_samples = (real_samples - real_min) / (real_max - real_min)

    noise_samples = np.array([ind[0] for ind in noise_source])
    noise_min, noise_max = noise_samples.min(), noise_samples.max()
    noise_samples = (noise_samples - noise_min) / (noise_max - noise_samples.min())

    
    d_losses, g_losses = [], []

    for epoch in range(epochs):
        
        idx = np.random.randint(0, real_samples.shape[0], batch_size)
        real = real_samples[idx]

        noise_idx = np.random.randint(0, noise_samples.shape[0], batch_size)
        noise = noise_samples[noise_idx]
        fake = gan.generator.predict(noise, verbose=0)

        d_loss_real = gan.discriminator.train_on_batch(real, np.ones((batch_size, 1)))
        d_loss_fake = gan.discriminator.train_on_batch(fake, np.zeros((batch_size, 1)))
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        
        gan.discriminator.trainable = False
        g_loss = gan.combined.train_on_batch(noise, np.ones((batch_size, 1)))
        gan.discriminator.trainable = True

        d_losses.append(d_loss)
        g_losses.append(g_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | D_loss: {d_loss:.4f} | G_loss: {g_loss:.4f}")

    
    plt.figure(figsize=(6, 4))
    plt.plot(d_losses, label='D loss')
    plt.plot(g_losses, label='G loss')
    plt.title('GAN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Loss curve saved to {os.path.abspath(save_path)}")

    return gan.discriminator

data_split(filename)


if os.path.exists('initial_group.npz') == 0:
    initial_group = initial_groups('train_iris.csv', 'setosa', gene_list)
    np.savez('initial_group.npz', *initial_group)
else:
    loaded_data=np.load('initial_group.npz')
    initial_group = [loaded_data[f'arr_{i}'] for i in range(len(loaded_data.files))]

if os.path.exists('normal_group.npz') == 0:
    normal_group = normal_groups('train_iris.csv', 'setosa', gene_list)
    np.savez('normal_group.npz', *normal_group)
else:
    loaded_data=np.load('normal_group.npz')
    normal_group = [loaded_data[f'arr_{i}'] for i in range(len(loaded_data.files))]




generations=[600]
rate=[0.1]


print(">>> Training GAN for initial_group evaluation ...")
gan_initial = GAN(len(gene_list), 0.0004, 0.7, 0.000125, 0.7)
real_for_initial = np.array([x[0] for x in normal_group])
discriminator_initial = train_gan(gan_initial, real_for_initial, initial_group, epochs=30, save_path='initial_gan_loss_curve.png')


print(">>> Training GAN for normal_group evaluation ...")
gan_normal = GAN(len(gene_list), 0.0003, 0.5, 0.0001, 0.5)
real_for_normal = np.array([x[0] for x in initial_group])
discriminator_normal = train_gan(gan_normal, real_for_normal, normal_group, epochs=50, save_path='normal_gan_loss_curve.png')

init_prob   = pre_evaluate(initial_group, discriminator_initial)
normal_prob = pre_evaluate(normal_group, discriminator_normal)

print(f"init_prob threshold: {init_prob:.4f}")
print(f"normal_prob threshold: {normal_prob:.4f}")

for i in range(len(rate)):
    for j in range(len(generations)):
        print("before", len(initial_group)/len(normal_group))
        population, population1, fitness_record, fitness_record1=genetic_algorithm(initial_group, normal_group, gene_list, generations[j], rate[i])
        print("after", len(population)/len(population1))
        

        #  写入 CSV
        with open('fitness_record.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for item in fitness_record:
                writer.writerow([item]) 


        #  写入 CSV
        with open('fitness_record1.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for item in fitness_record1:
                writer.writerow([item]) 

        print('---------------------before_T_cell---------------------')
        print("initial_population:", len(population))
        print("normal_population:", len(population1))

        population=initial_eliminate(population, gan_initial.discriminator, init_prob)
        population1=normal_eliminate(population1, gan_normal.discriminator, normal_prob)
        print('---------------------after_T_cell---------------------')
        print("initial_population:", len(population))
        print("normal_population:", len(population1))

        clf, X_test, y_test=train_svm_classifier(population, population1, 3, rate[i], generations[j], test_size=0.2, random_state=42)

        for id in range(len(gene_list)):
                
                clf, X_test, y_test=train_svm_classifier(population, population1, id, rate[i], generations[j], test_size=0.2, random_state=42)

    

        
      
         
        

            



