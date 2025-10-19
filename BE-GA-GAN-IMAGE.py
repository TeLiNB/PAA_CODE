import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Dense, LeakyReLU, Input
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import csv
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)

def build_generator(input_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim), LeakyReLU(0.2),
        Dense(64), LeakyReLU(0.2),
        Dense(input_dim, activation='tanh')
    ])
    return model

def build_discriminator(input_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim), LeakyReLU(0.2),
        Dense(64), LeakyReLU(0.2),
        Dense(1, activation='sigmoid')
    ])
    return model

class GAN:
    def __init__(self, input_dim, dp1=1e-5, dp2=0.5, gp1=2e-5, gp2=0.999):
        self.generator = build_generator(input_dim)
        self.discriminator = build_discriminator(input_dim)

        
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=tf.keras.optimizers.Adam(dp1, dp2))

      
        self.discriminator.trainable = False
        z = Input(shape=(input_dim,))
        fake = self.generator(z)
        validity = self.discriminator(fake)
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy',
                              optimizer=tf.keras.optimizers.Adam(gp1, gp2))


        self.discriminator.trainable = True


def train_gan(gan, real_samples, noise_samples, epochs, batch_size=32):
    
    real_samples = (real_samples.astype(np.float32) - 127.5) / 127.5
    noise_samples = (noise_samples.astype(np.float32) - 127.5) / 127.5

    n_real = real_samples.shape[0]
    n_noise = noise_samples.shape[0]
    steps_per_epoch = max(1, n_real // batch_size)

    for epoch in range(epochs):
        d_losses = []
        g_losses = []
        for step in range(steps_per_epoch):
            # sample batches
            idx = np.random.randint(0, n_real, batch_size)
            real = real_samples[idx]
            noise_idx = np.random.randint(0, n_noise, batch_size)
            noise = noise_samples[noise_idx]

            # generator produces fake from noise
            fake = gan.generator.predict(noise, verbose=0)

            # label smoothing: real ~0.9, fake 0.0
            real_labels = np.ones((batch_size, 1)) * 0.9
            fake_labels = np.zeros((batch_size, 1))

            d_loss_real = gan.discriminator.train_on_batch(real, real_labels)
            d_loss_fake = gan.discriminator.train_on_batch(fake, fake_labels)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            # train generator via combined (discriminator frozen in combined because we compiled that way)
            g_loss = gan.combined.train_on_batch(noise, np.ones((batch_size, 1)))

            d_losses.append(d_loss)
            g_losses.append(g_loss)

        print(f"{epoch+1}/{epochs} [D loss: {np.mean(d_losses):.4f}] [G loss: {np.mean(g_losses):.4f}]")
    return gan.discriminator




def pre_evaluate(population, discriminator):
    feats, _ = extract_features(population)
    feats = (feats.astype(np.float32) - 127.5) / 127.5   
    probs = discriminator.predict(feats, verbose=0).flatten()
    return probs

def eliminate_population(population, discriminator, threshold):
    feats, _ = extract_features(population)
    feats = (feats.astype(np.float32) - 127.5) / 127.5   
    probs = discriminator.predict(feats, verbose=0).flatten()
    keep = np.where(probs > threshold)[0]
    return [population[i] for i in keep]


def load_images_from_dir(path, label):
    images = []
    if label == 'abnormal':
        
        for subdir in os.listdir(path):
            if subdir.lower() == 'good':
                continue
            subpath = os.path.join(path, subdir)
            if not os.path.isdir(subpath):
                continue
            for fname in os.listdir(subpath):
                fpath = os.path.join(subpath, fname)
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (64, 64))
                chrom = img.flatten().tolist()
                images.append({'chromosome': chrom, 'label': label})
    else:
        
        for fname in os.listdir(path):
            fpath = os.path.join(path, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (64, 64))
            chrom = img.flatten().tolist()
            images.append({'chromosome': chrom, 'label': label})
    return images


def mutate_more_diverse(chrom, intensity=20):
    data = np.array(chrom).reshape(64, 64)
    x, y = np.random.randint(0, 64, 2)
    size = np.random.randint(1, 4)  # 扰动 1~3 像素
    noise = np.random.randint(-intensity, intensity+1, (size, size))
    x_end, y_end = min(x+size, 64), min(y+size, 64)
    data[x:x_end, y:y_end] = np.clip(
        data[x:x_end, y:y_end] + noise[:x_end-x, :y_end-y],
        0, 255
    )
    return data.flatten().tolist()


def crossover(p1, p2):
    point = random.randint(1, 64*64 - 1)
    child1 = p1['chromosome'][:point] + p2['chromosome'][point:]
    child2 = p2['chromosome'][:point] + p1['chromosome'][point:]
    return [{'chromosome': child1, 'label': p1['label']},
            {'chromosome': child2, 'label': p2['label']}]


def save_sample_images(samples, generation, prefix, num_save=5, output_dir="ga_outputs"):
    os.makedirs(f"{output_dir}/samples", exist_ok=True)
    counts = {'normal': 0, 'abnormal': 0}
    for ind in samples:
        label = ind['label']
        if counts[label] < num_save:
            img = np.array(ind['chromosome'], dtype=np.uint8).reshape(64, 64)
            filename = f"{output_dir}/samples/{prefix}_gen{generation}_{label}_{counts[label]}.png"
            cv2.imwrite(filename, img)
            counts[label] += 1
        if all(c >= num_save for c in counts.values()):
            break


def extract_features(samples):
    feats = []
    labels = []
    for s in samples:
        arr = np.array(s['chromosome'], dtype=np.float32)
        feats.append(arr)
        labels.append(0 if s['label'] == 'normal' else 1)
    return np.array(feats), np.array(labels)


def evaluate_population_cv(population, n_splits=10, output_dir="roc_outputs", gen=None):
    X, y = extract_features(population)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', gamma='scale', probability=True))
        ])


        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        # 常规指标
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred))
        metrics['recall'].append(recall_score(y_test, y_pred))
        metrics['f1'].append(f1_score(y_test, y_pred))
        metrics['roc_auc'].append(roc_auc_score(y_test, y_prob))

        # ROC 曲线
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tprs.append(tpr_interp)

    # 平均 ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(metrics['roc_auc'])

    if gen is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.figure()
        plt.plot(mean_fpr, mean_tpr, color='b', label=f'ROC (AUC={mean_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='r', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (Generation {gen})')
        plt.legend(loc="lower right")
        plt.savefig(f"{output_dir}/roc_gen{gen}.png", dpi=300)
        plt.close()

    return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}


# ========== 主函数 ==========
def genetic(
    pop_size=200,
    generations=50,
    mutate_rate=0.0,
    crossover_rate=0,
    save_samples_n=6):


    

    folder_name=['bottle', 'cable', 'capsule', 'hazelnut',  'metal_nut', 'pill', 'screw', 

                 'toothbrush', 'transistor', 'zipper']

    for folder in folder_name:
        fitness=[]
        print(f'-------------------------{folder}---------------------------------')
        normal_dir="D:/Desktop1/MVTecAD/"+folder+"/train/good/"
        abnormal_dir="D:/Desktop1/MVTecAD/"+folder+"/test/"
        output_dir="GAN_0_"+folder
        normal_pop = load_images_from_dir(normal_dir, 'normal')
        abnormal_pop = load_images_from_dir(abnormal_dir, 'abnormal')
        population = normal_pop + abnormal_pop

        while len([p for p in population if p['label'] == 'normal']) < pop_size:
            population += normal_pop
        while len([p for p in population if p['label'] == 'abnormal']) < pop_size:
            population += abnormal_pop

        population = population[:pop_size] + population[-pop_size:]

        os.makedirs(output_dir, exist_ok=True)
        history = []

        
        X_normal, _ = extract_features(normal_pop)
        X_abnormal, _ = extract_features(abnormal_pop)
        gan = GAN(input_dim=X_normal.shape[1])
        disc = train_gan(gan, X_normal, X_abnormal, epochs=0)

        
        
        normal_prob = np.min(pre_evaluate(normal_pop, disc))
        abnormal_prob = np.min(pre_evaluate(abnormal_pop, disc))

        

        for gen in range(1, generations+1):
            print(f"Generation {gen}")
            
            new_pop = []
            for ind in population:
                new_ind = ind.copy()
                if random.random() < mutate_rate:
                    new_ind['chromosome'] = mutate_more_diverse(new_ind['chromosome'])
                new_pop.append(new_ind)

            
            if crossover_rate > 0:
                for _ in range(int(len(new_pop) * crossover_rate // 2)):
                    p1, p2 = random.sample(new_pop, 2)
                    new_pop.extend(crossover(p1, p2))

            
            new_pop = random.sample(new_pop, min(len(new_pop), pop_size*2))
            population = random.sample(new_pop, len(new_pop))[:len(new_pop)]
            
        
            population_normal = [p for p in population if p['label']=='normal']
            population_abnormal = [p for p in population if p['label']=='abnormal']
            

            population_normal = eliminate_population(population_normal, disc, normal_prob)
            population_abnormal = eliminate_population(population_abnormal, disc, abnormal_prob)

            print("after",len(population_normal)/len(population_abnormal))

            prob = pre_evaluate(population_normal, disc)
            fitness.append(np.mean(prob))
            
            with open(f'D:/Desktop1/MVTECAD/{folder}.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for item in fitness:
                    writer.writerow([item]) 


                population = population_normal + population_abnormal
                # ROC
                scores = evaluate_population_cv(population, gen=gen, output_dir=f"{output_dir}/roc_curves")
                history.append({'generation': gen, **{f"{k}_mean": v[0] for k, v in scores.items()},
                                                **{f"{k}_std": v[1] for k, v in scores.items()}})
                print(f"Gen {gen}: {scores}")

    
                save_sample_images(population, gen, f'{folder}', save_samples_n, output_dir)


        df = pd.DataFrame(history)
        df.to_csv(f"{output_dir}/cv_history.csv", index=False)


        plt.figure(figsize=(8,6))
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            plt.plot(df['generation'], df[f'{metric}_mean'], label=metric)
        plt.xlabel('Generation')
        plt.ylabel('Score')
        plt.title(f'10-Fold CV Metrics per Generation {folder}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/cv_metrics.png", dpi=300)
        plt.close()

    return 0


if __name__ == "__main__":
    genetic()
