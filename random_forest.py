# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 16:06:03 2026

@author: nepor
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score
import matplotlib.pyplot as plt 
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE

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



### First try: random values by using randomized search


# Modelo base
rf = RandomForestClassifier(random_state=3)

# Espaço de busca
param_dist = {
    'n_estimators': [100, 200, 300],  
    'max_depth': [None, 10, 20],       
    'max_leaf_nodes': [None, 10, 20],  
    'min_samples_split': [2, 4],      
    'min_samples_leaf': [1, 2]         
}


# Randomized Search com validação cruzada
search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    random_state=3,
    n_jobs=-1
)

search.fit(X_synthetic_scaled, y_synthetic)

print("Melhores parâmetros:", search.best_params_)
print("Melhor score (CV nos sintéticos):", search.best_score_)

# Modelo final com melhores parâmetros
best_rf = search.best_estimator_

# --- Teste nos próprios sintéticos (diagnóstico) ---
y_synthetic_pred = best_rf.predict(X_synthetic_scaled)

print("\nResultados no treino sintético (diagnóstico):")
print("Accuracy:", accuracy_score(y_synthetic, y_synthetic_pred))
print("Precision:", precision_score(y_synthetic, y_synthetic_pred, average='weighted'))
print("Recall:", recall_score(y_synthetic, y_synthetic_pred, average='weighted'))
print("F1-score:", f1_score(y_synthetic, y_synthetic_pred, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_synthetic, y_synthetic_pred))

# --- Blind test nos reais ---
y_real_pred = best_rf.predict(X_real_scaled)

print("\nResultados no teste real (blind test):")
print("Accuracy:", accuracy_score(y_real, y_real_pred))
print("Precision:", precision_score(y_real, y_real_pred, average='weighted'))
print("Recall:", recall_score(y_real, y_real_pred, average='weighted'))
print("F1-score:", f1_score(y_real, y_real_pred, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_real, y_real_pred))
print("\nRelatório de classificação:\n", classification_report(y_real, y_real_pred))



### ------ REFINEMENT 1: Oversampling com SMOTE nos sintéticos ------- ###

# Estratégia: aumentar MM, MH e UN
sampling_strategy = {
    'MM': 1276,   # 876 originais + 400 novos
    'MH': 1055,   # 655 originais + 400 novos
    'UN': 52      # 37 originais + 15 novos
}

smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)

X_syn_res, y_syn_res = smote.fit_resample(X_synthetic, y_synthetic)

print("Distribuição sintética original:\n", y_synthetic.value_counts())
print("Distribuição sintética após SMOTE:\n", pd.Series(y_syn_res).value_counts())

# Escalar novamente
scaler = StandardScaler()
X_syn_res_scaled = scaler.fit_transform(X_syn_res)
X_real_scaled = scaler.transform(X_real)  # importante: usar o mesmo scaler

# Espaço de busca
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'max_leaf_nodes': [None, 10, 20],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2]
}

# Randomized Search com validação cruzada nos sintéticos balanceados
search = RandomizedSearchCV(
    RandomForestClassifier(random_state=3),
    param_distributions=param_dist,
    n_iter=20,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    random_state=3,
    n_jobs=-1
)

search.fit(X_syn_res_scaled, y_syn_res)

print("Melhores parâmetros:", search.best_params_)
print("Melhor score (CV nos sintéticos balanceados):", search.best_score_)

# Modelo final
best_rf = search.best_estimator_

# --- Blind test nos reais ---
y_real_pred = best_rf.predict(X_real_scaled)

print("\nResultados no teste real (blind test):")
print("Accuracy:", accuracy_score(y_real, y_real_pred))
print("Precision:", precision_score(y_real, y_real_pred, average='weighted'))
print("Recall:", recall_score(y_real, y_real_pred, average='weighted'))
print("F1-score:", f1_score(y_real, y_real_pred, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_real, y_real_pred))
print("\nRelatório de classificação:\n", classification_report(y_real, y_real_pred))



### ------ TESTE COM SPLIT 70/30 ------- ###


# Espaço de busca
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'max_leaf_nodes': [None, 10, 20],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2]
}

def run_pipeline(X, y, label):
    # Split 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Randomized Search
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=3),
        param_distributions=param_dist,
        n_iter=20,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
        random_state=3,
        n_jobs=-1
    )
    
    search.fit(X_train_scaled, y_train)
    
    print(f"\n[{label}] Melhores parâmetros:", search.best_params_)
    print(f"[{label}] Melhor score (CV):", search.best_score_)
    
    # Modelo final
    best_rf = search.best_estimator_
    
    # Avaliar no teste
    y_pred = best_rf.predict(X_test_scaled)
    
    print(f"\nResultados no teste ({label}):")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1-score:", f1_score(y_test, y_pred, average='weighted'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nRelatório de classificação:\n", classification_report(y_test, y_pred))

# 1. Sintético
run_pipeline(X_synthetic, y_synthetic, "Sintético")

# 2. Sintético + Real
X_concat = pd.concat([X_synthetic, X_real])
y_concat = pd.concat([y_synthetic, y_real])
run_pipeline(X_concat, y_concat, "Sintético + Real")

# 3. Real
run_pipeline(X_real, y_real, "Real")

