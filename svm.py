# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 13:39:11 2026

@author: nepor
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,classification_report
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

df_real_data = pd.read_csv('D:/Codigos_VSCODE/IME/projectiles_articles/Dataset/RAMP-RT real v1.csv')
df_synthetic_data = pd.read_csv('D:/Codigos_VSCODE/IME/projectiles_articles/Dataset/RAMP-RT synthetic v1.csv')



#### ------ SIMPLE PIPELINE -------######

#classification target and data
X_synthetic = df_synthetic_data.drop('class', axis=1)
y_synthetic = df_synthetic_data['class']

X_real = df_real_data.drop('class', axis=1)
y_real = df_real_data['class']


#Normal Scale
scaler = StandardScaler() 
X_synthetic_scaled = scaler.fit_transform(X_synthetic)
X_real_scaled = scaler.transform(X_real)

## Cross validation in order to find the best C

c_values = [i/10 for i in range(1,11)]
c_values.extend([i for i in range (10, 110, 10)])
cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 
best_c = None
best_score = 0

for c in c_values:
    svm_model =  SVC(C=c)
    scores = cross_val_score(svm_model, X_synthetic_scaled, y_synthetic, cv=cross_validator, scoring='accuracy')
    mean_score = scores.mean()
    print(f"C={c}, mean accuracy={mean_score:.4f}")
    if(round(mean_score,4)==round(best_score,4)):break
    if mean_score > best_score:
        best_score = mean_score
        best_c = c
        
#Metrics of cross validation with synthetic data

svm_model = SVC(C=best_c)
svm_model.fit(X_synthetic_scaled, y_synthetic)

y_synthetic_pred = svm_model.predict(X_synthetic_scaled)

accuracy_syn = accuracy_score(y_synthetic, y_synthetic_pred)
precision_syn = precision_score(y_synthetic, y_synthetic_pred, average='weighted')
recall_syn = recall_score(y_synthetic, y_synthetic_pred, average='weighted')
f1_syn = f1_score(y_synthetic, y_synthetic_pred, average='weighted')
cm_syn = confusion_matrix(y_synthetic, y_synthetic_pred)

print("\nResults for the complete train of synthetic_data :")
print("Accuracy:", accuracy_syn)
print("Precision:", precision_syn)
print("Recall:", recall_syn)
print("F1-score:", f1_syn)
print("Confusion Matrix:\n", cm_syn)

#Blind Test
svm_model = SVC(C=best_c)
svm_model.fit(X_synthetic_scaled, y_synthetic)

# predict based on real data
y_real_pred = svm_model.predict(X_real_scaled)

# Metrics
accuracy = accuracy_score(y_real, y_real_pred)
precision = precision_score(y_real, y_real_pred, average='weighted')
recall = recall_score(y_real, y_real_pred, average='weighted')
f1 = f1_score(y_real, y_real_pred, average='weighted')
cm = confusion_matrix(y_real, y_real_pred)

print("Results for testing with real data")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:\n", cm)




#-------REFINEMENT 1: Training only with MM, MH, UN--------------#

# Filter synthetic dataset to keep only MM, MH, UN
df_synthetic_filtered = df_synthetic_data[df_synthetic_data['class'].isin(['MM', 'MH', 'UN'])]

X_synthetic_filt = df_synthetic_filtered.drop('class', axis=1)
y_synthetic_filt = df_synthetic_filtered['class']

# Scale again with filtered data
scaler = StandardScaler()
X_synthetic_filt_scaled = scaler.fit_transform(X_synthetic_filt)
X_real_scaled = scaler.transform(X_real)  
best_c = 10 #pattern for a small base


# Train final model with filtered synthetic data
svm_model = SVC(C=best_c)
svm_model.fit(X_synthetic_filt_scaled, y_synthetic_filt)

# Predict on real data
y_real_pred = svm_model.predict(X_real_scaled)

# Metrics
accuracy = accuracy_score(y_real, y_real_pred)
precision = precision_score(y_real, y_real_pred, average='weighted')
recall = recall_score(y_real, y_real_pred, average='weighted')
f1 = f1_score(y_real, y_real_pred, average='weighted')
cm = confusion_matrix(y_real, y_real_pred)

print("\n[Refinement 1] Results for testing with real data (MM, MH, UN only):")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:\n", cm)



### --------REFINEMENT 2: Give weights to the classes which exist on real data ------#


## same process of cross validation



cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 
best_c = None
best_score = 0

for c in c_values:
    svm_model =  SVC(C=c, class_weight={'MM':2.5, 'MH':2.5, 'UN':1})
    scores = cross_val_score(svm_model, X_synthetic_scaled, y_synthetic, cv=cross_validator, scoring='accuracy')
    mean_score = scores.mean()
    print(f"C={c}, mean accuracy={mean_score:.4f}")
    if(round(mean_score,4)==round(best_score,4)):break
    if mean_score > best_score:
        best_score = mean_score
        best_c = c
        
## not better


## trying oversample

# Oversample MM and MH by duplicating their rows
df_mm = df_synthetic_filtered[df_synthetic_filtered['class'] == 'MM']
df_mh = df_synthetic_filtered[df_synthetic_filtered['class'] == 'MH']
df_un = df_synthetic_filtered[df_synthetic_filtered['class'] == 'UN']

#duplicate MM and MH once (factor=2). 
df_synthetic_weighted = pd.concat([df_mm, df_mm, df_mh, df_mh, df_un])

# Features and target
X_synthetic_weighted = df_synthetic_weighted.drop('class', axis=1)
y_synthetic_weighted = df_synthetic_weighted['class']

# Scale again
scaler = StandardScaler()
X_synthetic_weighted_scaled = scaler.fit_transform(X_synthetic_weighted)
X_real_scaled = scaler.transform(X_real)


cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 
best_c = None
best_score = 0

for c in c_values:
    svm_model =  SVC(C=c, class_weight='balanced')
    scores = cross_val_score(svm_model, X_synthetic_weighted_scaled, y_synthetic_weighted, cv=cross_validator, scoring='accuracy')
    mean_score = scores.mean()
    print(f"C={c}, mean accuracy={mean_score:.4f}")
    if(round(mean_score,4)==round(best_score,4)):break
    if mean_score > best_score:
        best_score = mean_score
        best_c = c
        
## O best_c fica 100 aqui, mas temos que evitar overfitting, portanto,usar 30

best_c=30

# Train final model with oversampled synthetic data
svm_model = SVC(C=best_c, class_weight='balanced')
svm_model.fit(X_synthetic_weighted_scaled, y_synthetic_weighted)

# Metrics on synthetic data
y_synthetic_pred = svm_model.predict(X_synthetic_weighted_scaled)

accuracy_syn = accuracy_score(y_synthetic_weighted, y_synthetic_pred)
precision_syn = precision_score(y_synthetic_weighted, y_synthetic_pred, average='weighted')
recall_syn = recall_score(y_synthetic_weighted, y_synthetic_pred, average='weighted')
f1_syn = f1_score(y_synthetic_weighted, y_synthetic_pred, average='weighted')
cm_syn = confusion_matrix(y_synthetic_weighted, y_synthetic_pred)

print("\n[Refinement 2] Results for synthetic data (oversampled MM/MH):")
print("Accuracy:", accuracy_syn)
print("Precision:", precision_syn)
print("Recall:", recall_syn)
print("F1-score:", f1_syn)
print("Confusion Matrix:\n", cm_syn)

#  Metrics on real data (blind test) 
y_real_pred = svm_model.predict(X_real_scaled)

accuracy = accuracy_score(y_real, y_real_pred)
precision = precision_score(y_real, y_real_pred, average='weighted')
recall = recall_score(y_real, y_real_pred, average='weighted')
f1 = f1_score(y_real, y_real_pred, average='weighted')
cm = confusion_matrix(y_real, y_real_pred)

print("\n[Refinement 2] Results for testing with real data (oversampled MM/MH):")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:\n", cm)



## - Testando SPlit 75/25 ----- ###


## at first try to find the best c for real data

cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 
best_c = None
best_score = 0

for c in c_values:
    svm_model =  SVC(C=c, kernel='linear')
    scores = cross_val_score(svm_model, X_real_scaled, y_real, cv=cross_validator, scoring='accuracy')
    mean_score = scores.mean()
    print(f"C={c}, mean accuracy={mean_score:.4f}")
    if mean_score > best_score:
        best_score = mean_score
        best_c = c


#great values for all c(the same value)




# --- Sintético ---
X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(
    X_synthetic, y_synthetic, test_size=0.3, stratify=y_synthetic, random_state=3
)
sc_syn = StandardScaler()
X_train_syn = sc_syn.fit_transform(X_train_syn)
X_test_syn = sc_syn.transform(X_test_syn)

svm_syn = SVC(C=30)
svm_syn.fit(X_train_syn, y_train_syn)
y_pred_syn = svm_syn.predict(X_test_syn)

print("\n[SVM - Sintético com split e kernel RBF]")
print("Accuracy:", accuracy_score(y_test_syn, y_pred_syn))
print("Precision:", precision_score(y_test_syn, y_pred_syn, average='weighted'))
print("Recall:", recall_score(y_test_syn, y_pred_syn, average='weighted'))
print("F1-score:", f1_score(y_test_syn, y_pred_syn, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test_syn, y_pred_syn))


# --- Real ---
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X_real, y_real, test_size=0.3, stratify=y_real, random_state=3
)
sc_real = StandardScaler()
X_train_real = sc_real.fit_transform(X_train_real)
X_test_real = sc_real.transform(X_test_real)

svm_real = SVC(kernel='linear')
svm_real.fit(X_train_real, y_train_real)
y_pred_real = svm_real.predict(X_test_real)

print("\n[SVM - Real com linear]")
print("Accuracy:", accuracy_score(y_test_real, y_pred_real))
print("Precision:", precision_score(y_test_real, y_pred_real, average='weighted'))
print("Recall:", recall_score(y_test_real, y_pred_real, average='weighted'))
print("F1-score:", f1_score(y_test_real, y_pred_real, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test_real, y_pred_real))


# --- Superbase ---
df_superbase = pd.concat([df_synthetic_data, df_real_data], ignore_index=True)
X_super = df_superbase.drop('class', axis=1)
y_super = df_superbase['class']

X_train_super, X_test_super, y_train_super, y_test_super = train_test_split(
    X_super, y_super, test_size=0.3, stratify=y_super, random_state=3
)
sc_super = StandardScaler()
X_train_super = sc_super.fit_transform(X_train_super)
X_test_super = sc_super.transform(X_test_super)

svm_super = SVC(C=30)
svm_super.fit(X_train_super, y_train_super)
y_pred_super = svm_super.predict(X_test_super)

print("\n[SVM - Superbase com split e RBF]")
print("Accuracy:", accuracy_score(y_test_super, y_pred_super))
print("Precision:", precision_score(y_test_super, y_pred_super, average='weighted'))
print("Recall:", recall_score(y_test_super, y_pred_super, average='weighted'))
print("F1-score:", f1_score(y_test_super, y_pred_super, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test_super, y_pred_super))



## Test with the synthetic and the mixed with lienar kernel

# --- Sintético ---
X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(
    X_synthetic, y_synthetic, test_size=0.3, stratify=y_synthetic, random_state=3
)
sc_syn = StandardScaler()
X_train_syn = sc_syn.fit_transform(X_train_syn)
X_test_syn = sc_syn.transform(X_test_syn)

svm_syn = SVC(C=30, kernel='linear')
svm_syn.fit(X_train_syn, y_train_syn)
y_pred_syn = svm_syn.predict(X_test_syn)

print("\n[SVM - Sintético com kernel linear e split]")
print("Accuracy:", accuracy_score(y_test_syn, y_pred_syn))
print("Precision:", precision_score(y_test_syn, y_pred_syn, average='weighted'))
print("Recall:", recall_score(y_test_syn, y_pred_syn, average='weighted'))
print("F1-score:", f1_score(y_test_syn, y_pred_syn, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test_syn, y_pred_syn))


# --- Superbase ---
df_superbase = pd.concat([df_synthetic_data, df_real_data], ignore_index=True)
X_super = df_superbase.drop('class', axis=1)
y_super = df_superbase['class']

X_train_super, X_test_super, y_train_super, y_test_super = train_test_split(
    X_super, y_super, test_size=0.3, stratify=y_super, random_state=3
)
sc_super = StandardScaler()
X_train_super = sc_super.fit_transform(X_train_super)
X_test_super = sc_super.transform(X_test_super)

svm_super = SVC(C=30, kernel='linear')
svm_super.fit(X_train_super, y_train_super)
y_pred_super = svm_super.predict(X_test_super)

print("\n[SVM - Superbase com kernel linear e split]")
print("Accuracy:", accuracy_score(y_test_super, y_pred_super))
print("Precision:", precision_score(y_test_super, y_pred_super, average='weighted'))
print("Recall:", recall_score(y_test_super, y_pred_super, average='weighted'))
print("F1-score:", f1_score(y_test_super, y_pred_super, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test_super, y_pred_super))


## let's see what happens with the filtered classes:

# Synt filtered
X_train_filt_syn, X_test_filt_syn, y_train_filt_syn, y_test_filt_syn = train_test_split(
    X_synthetic_filt, y_synthetic_filt, test_size=0.3, stratify=y_synthetic_filt, random_state=3
)

sc_filt_syn = StandardScaler()
X_train_filt_syn = sc_filt_syn.fit_transform(X_train_filt_syn)
X_test_filt_syn = sc_filt_syn.transform(X_test_filt_syn)

svm_syn = SVC(C=30, kernel='linear')
svm_syn.fit(X_train_filt_syn, y_train_filt_syn)
y_pred_syn = svm_syn.predict(X_test_filt_syn)

print("\n[SVM - Sintético filtrado split]")
print("Accuracy:", accuracy_score(y_test_filt_syn, y_pred_syn))
print("Precision:", precision_score(y_test_filt_syn, y_pred_syn, average='weighted'))
print("Recall:", recall_score(y_test_filt_syn, y_pred_syn, average='weighted'))
print("F1-score:", f1_score(y_test_filt_syn, y_pred_syn, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test_filt_syn, y_pred_syn))


# --- Filtered ---
X_train_filt, X_test_filt, y_train_filt, y_test_filt = train_test_split(
    X_synthetic_filt, y_synthetic_filt, test_size=0.3, stratify=y_synthetic_filt, random_state=3
)

# Padronização apenas no treino
sc_filt = StandardScaler()
X_train_filt = sc_filt.fit_transform(X_train_filt)
X_test_filt = sc_filt.transform(X_test_filt)

# SVM com kernel linear
svm_filt = SVC(C=30,kernel='linear')
svm_filt.fit(X_train_filt, y_train_filt)
y_pred_filt = svm_filt.predict(X_test_filt)

print("\n[SVM - Filtered com linear]")
print("Accuracy:", accuracy_score(y_test_filt, y_pred_filt))
print("Precision:", precision_score(y_test_filt, y_pred_filt, average='weighted'))
print("Recall:", recall_score(y_test_filt, y_pred_filt, average='weighted'))
print("F1-score:", f1_score(y_test_filt, y_pred_filt, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test_filt, y_pred_filt))


# --- Oversampling ---
X_train_over, X_test_over, y_train_over, y_test_over = train_test_split(
    X_synthetic_weighted, y_synthetic_weighted, test_size=0.3, stratify=y_synthetic_weighted, random_state=3
)

# Padronização apenas no treino
sc_over = StandardScaler()
X_train_over = sc_over.fit_transform(X_train_over)
X_test_over = sc_over.transform(X_test_over)

# SVM com kernel RBF e C=30
svm_over = SVC(C=30)
svm_over.fit(X_train_over, y_train_over)
y_pred_over = svm_over.predict(X_test_over)

print("\n[SVM - Oversampling]")
print("Accuracy:", accuracy_score(y_test_over, y_pred_over))
print("Precision:", precision_score(y_test_over, y_pred_over, average='macro'))
print("Recall:", recall_score(y_test_over, y_pred_over, average='macro'))
print("F1-score:", f1_score(y_test_over, y_pred_over, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_test_over, y_pred_over))



# Synthetic only
print("\n[Refinement 4] Synthetic only - Class-wise metrics:")
print(classification_report(y_test_syn, svm_syn.predict(X_test_syn)))

# Synthetic + Real (Superbase)
print("\n[Refinement 4] Synthetic + Real - Class-wise metrics:")
print(classification_report(y_test_super, svm_super.predict(X_test_super)))

# Real only
print("\n[Refinement 4] Real only - Class-wise metrics:")
print(classification_report(y_test_real, svm_real.predict(X_test_real)))






### To see the boundary plot with PCA





# Split dos dados reais
X_train, X_test, y_train, y_test = train_test_split(
    X_real, y_real, test_size=0.3, stratify=y_real, random_state=3
)

# Padronização
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Redução de dimensionalidade para 2 componentes principais
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Treina SVM linear nos dados projetados
svm_real_pca = SVC(kernel='linear')
svm_real_pca.fit(X_train_pca, y_train)

# Avaliação
y_pred = svm_real_pca.predict(X_test_pca)
print("Accuracy com PCA (2D):", accuracy_score(y_test, y_pred))

# Função para plotar fronteira
def plot_decision_boundary(X, y, model, title):
    X_set, y_set = X, y
    X1, X2 = np.meshgrid(
        np.linspace(X_set[:, 0].min()-1, X_set[:, 0].max()+1, 200),
        np.linspace(X_set[:, 1].min()-1, X_set[:, 1].max()+1, 200)
    )
    grid_points = np.array([X1.ravel(), X2.ravel()]).T.astype(float)
    Z = model.predict(grid_points)
    # Converte rótulos para índices numéricos
    Z_num = np.array([np.where(model.classes_ == label)[0][0] for label in Z])
    Z_num = Z_num.reshape(X1.shape)

    plt.contourf(X1, X2, Z_num, alpha=0.75, cmap=ListedColormap(('lightblue', 'salmon','lightgreen')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(
            X_set[y_set == j, 0], X_set[y_set == j, 1],
            c=ListedColormap(('blue', 'red','yellow'))(i), label=j, edgecolor='k'
        )
    plt.title(title)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.show()



# Gráfico com dados de treino projetados
plot_decision_boundary(X_train_pca, y_train, svm_real_pca, 'SVM (Treino - Reais com PCA)')

# Gráfico com dados de teste projetados
plot_decision_boundary(X_test_pca, y_test, svm_real_pca, 'SVM (Teste - Reais com PCA)')

