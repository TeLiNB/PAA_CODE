import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy import stats
import os
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import precision_score, recall_score, f1_score
import csv
from keras.models import Model, Sequential
from keras.layers import Dense, LeakyReLU, Input
import tensorflow as tf

gene_list = ['Action', 'Adventure', 'Animation', 'Children\'s',
             'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
             'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
             'Sci-Fi', 'Thriller', 'War', 'Western']

def data_split(filename):
    
    data = pd.read_csv(filename)
    header = data.columns.tolist()  

    
    train_movie = pd.DataFrame(columns=header)

    
    test_movie = pd.DataFrame(columns=header)

    Tag = data['tag'].tolist()

    like = []
    unlike = []

    for i in range(len(Tag)):
        if Tag[i] >= 3:
            like.append(i)
        else:
            unlike.append(i)

    
    for i in range(math.floor(len(like) * 0.8)):
        train_movie = pd.concat([train_movie, data.iloc[[like[i]]]])  

    for i in range(math.floor(len(unlike) * 0.8)):
        train_movie = pd.concat([train_movie, data.iloc[[unlike[i]]]])

    
    for i in range(math.floor(len(like) * 0.2)):
        test_movie = pd.concat([test_movie, data.iloc[[like[i + math.floor(len(like) * 0.8)]]]])

    for i in range(math.floor(len(unlike) * 0.2)):
        test_movie = pd.concat([test_movie, data.iloc[[unlike[i + math.floor(len(unlike) * 0.8)]]]])

    
    train_movie.to_csv('train_movie.csv', index=False)
    test_movie.to_csv('test_movie.csv', index=False)

def normalize_all_columns(filename):
    
    data =pd.read_csv(filename)

    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

    
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data[numeric_columns])

    
    normalized_df = pd.DataFrame(normalized_data, columns=numeric_columns)

    
    non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_columns:
        normalized_df = pd.concat([normalized_df, data[non_numeric_columns]], axis=1)

    return normalized_df

def data_collection( rating, movie, gene_list):
    
    movies = pd.read_csv( movie)

    
    ratings = pd.read_csv( rating)

    header = ['user'] + gene_list + ['tag']
    data = pd.DataFrame(columns=header)

    
    mid = movies['movieId'].tolist()
    mid_index = {k: v for v, k in enumerate(mid)}

    
    genres = movies['genres'].tolist()
    for i, genre in enumerate(genres):
        genres[i] = genre.replace('|', ',')


    
    uid = ratings['userId'].tolist()
    
    fid = ratings['movieId'].tolist()
    
    scores = ratings['rating'].tolist()

    for i in range(15000):
        print('-----------', i, '--------------')
        if uid[i]==3:
            temp = [uid[i]] + [0] * len(gene_list) + [0]
            if fid[i] in mid_index:
                midx = mid_index[fid[i]]
                for genre in genres[midx].split(','):
                    if genre in gene_list:
                        gidx = gene_list.index(genre)
                        temp[gidx + 1] += 1
                if scores[i] >= 3:
                    temp[-1] = 1
                data.loc[len(data)] = temp

    data.to_csv('ml-25m.csv', index=False)


def initial_groups(filename, target_species, gene_list):
    
    data=normalize_all_columns(filename)
    
    gene1=data['Action'].tolist()
    
    gene2=data['Adventure'].tolist()
    
    gene3=data['Animation'].tolist()
    
    gene4=data['Children\'s'].tolist()
    
    gene5=data['Comedy'].tolist()
    
    gene6=data['Crime'].tolist()
    
    gene7=data['Documentary'].tolist()
    
    gene8=data['Drama'].tolist()
    
    gene9=data['Fantasy'].tolist()
    
    gene10 = data['Film-Noir'].tolist()
    
    gene11 = data['Horror'].tolist()
    
    gene12 = data['Musical'].tolist()
    
    gene13 = data['Mystery'].tolist()
    
    gene14 = data['Romance'].tolist()
    
    gene15 = data['Sci-Fi'].tolist()
    
    gene16 = data['Thriller'].tolist()
    
    gene17 = data['War'].tolist()
    
    gene18 = data['Western'].tolist()
    
    tag_data=pd.read_csv(filename)
    target=tag_data['tag'].tolist()
    initial_group=[]
    id = 0
    for i in range(len(target)):
        if target[i] == target_species:
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
            initial_group[id][0][9] = gene10[i]
            initial_group[id][0][10] = gene11[i]
            initial_group[id][0][11] = gene12[i]
            initial_group[id][0][12] = gene13[i]
            initial_group[id][0][13] = gene14[i]
            initial_group[id][0][14] = gene15[i]
            initial_group[id][0][15] = gene16[i]
            initial_group[id][0][16] = gene17[i]
            initial_group[id][0][17] = gene18[i]
            id+=1

    return initial_group

def normal_groups(filename, target_species, gene_list):
    
    data=normalize_all_columns(filename)

    
    gene1 = data['Action'].tolist()
    
    gene2 = data['Adventure'].tolist()
    
    gene3 = data['Animation'].tolist()
    
    gene4 = data['Children\'s'].tolist()

    gene5 = data['Comedy'].tolist()
    
    gene6 = data['Crime'].tolist()
    
    gene7 = data['Documentary'].tolist()
    
    gene8 = data['Drama'].tolist()
    
    gene9 = data['Fantasy'].tolist()
    
    gene10 = data['Film-Noir'].tolist()
    
    gene11 = data['Horror'].tolist()
    
    gene12 = data['Musical'].tolist()
    
    gene13 = data['Mystery'].tolist()
    
    gene14 = data['Romance'].tolist()
    
    gene15 = data['Sci-Fi'].tolist()
    
    gene16 = data['Thriller'].tolist()
    
    gene17 = data['War'].tolist()
    
    gene18 = data['Western'].tolist()
    
    tag_data=pd.read_csv(filename)
    target = tag_data['tag'].tolist()
    normal_group = []
    id = 0
    for i in range(len(target)):
        if target[i] == target_species:
            normal_group.append(np.zeros([1, len(gene_list)]))
            normal_group[id][0][0] = gene1[i]
            normal_group[id][0][1] = gene2[i]
            normal_group[id][0][2] = gene3[i]
            normal_group[id][0][3] = gene4[i]
            normal_group[id][0][4] = gene5[i]
            normal_group[id][0][5] = gene6[i]
            normal_group[id][0][6] = gene7[i]
            normal_group[id][0][7] = gene8[i]
            normal_group[id][0][8] = gene9[i]
            normal_group[id][0][9] = gene10[i]
            normal_group[id][0][10] = gene11[i]
            normal_group[id][0][11] = gene12[i]
            normal_group[id][0][12] = gene13[i]
            normal_group[id][0][13] = gene14[i]
            normal_group[id][0][14] = gene15[i]
            normal_group[id][0][15] = gene16[i]
            normal_group[id][0][16] = gene17[i]
            normal_group[id][0][17] = gene18[i]
            id += 1
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


def crossover(parent1, parent2, gene_list):
    child1=parent1.copy()
    child2=parent2.copy()
    point=random.randint(1, len(parent1[0])-1)
    child1[0][point:], child2[0][point:] = child2[0][point:], child1[0][point:]
    return child1, child2


def mutation(individual, gene_list, direct, rate):
    uniform=random.uniform
    for col in range(len(gene_list)):
        if random.random()<0.5:
                individual[0][col]+=uniform(-rate*direct, 0)
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
        child1, child2 = crossover(parent1, parent2, gene_list)
        child1 = mutation(child1, gene_list, -1, rate)
        child2 = mutation(child2, gene_list, -1, rate)
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
        child1, child2 = crossover(parent1, parent2, gene_list)
        child1 = mutation(child1, gene_list, 1, rate)
        child2 = mutation(child2, gene_list, 1, rate)
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



    


    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    with open(f'GAN_T_RBF_accuracy_records1.csv', 'a', newline='', encoding='utf-8') as csvfile:
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




movie = 'movies.csv'

rating = 'ratings.csv'
filename='ml-25m.csv'

data_split(filename)

data_collection(rating, movie, gene_list)



if os.path.exists('initial_group.npz') == 0:
    initial_group = initial_groups(filename, 1, gene_list)
    np.savez('initial_group.npz', *initial_group)
else:
    loaded_data=np.load('initial_group.npz')
    initial_group = [loaded_data[f'arr_{i}'] for i in range(len(loaded_data.files))]

if os.path.exists('normal_group.npz') == 0:
    normal_group = normal_groups( filename,0,  gene_list)
    np.savez('normal_group.npz', *normal_group)
else:
    loaded_data=np.load('normal_group.npz')
    normal_group = [loaded_data[f'arr_{i}'] for i in range(len(loaded_data.files))]


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

rate=[0.02]
generations=[600]

for i in range(len(rate)):
    for j in range(len(generations)):
        print("before：", len(initial_group)/len(normal_group))
        population, population1, fitness_record, fitness_record1=genetic_algorithm(initial_group, normal_group, gene_list, generations[j], rate[i])
        print("after：", len(population)/len(population1))


    
        with open('fitness_record.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for item in fitness_record:
                writer.writerow([item]) 


       
        with open('fitness_record1.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for item in fitness_record1:
                writer.writerow([item]) 

        os.makedirs('{rate[i]}/{generations[j]}', exist_ok=True)

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