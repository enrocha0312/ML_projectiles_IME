# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 15:48:37 2026

@author: nepor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier, plot_importance
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import time 
from memory_profiler import memory_usage

df_real_data = pd.read_csv('D:/Codigos_VSCODE/IME/projectiles_articles/Dataset/RAMP-RT real v1.csv')
df_synthetic_data = pd.read_csv('D:/Codigos_VSCODE/IME/projectiles_articles/Dataset/RAMP-RT synthetic v1.csv')

#### ------ SIMPLE PIPELINE ------- ######


X_synthetic = df_synthetic_data.drop('class', axis=1)
y_synthetic = df_synthetic_data['class']

X_real = df_real_data.drop('class', axis=1)
y_real = df_real_data['class']

scaler = StandardScaler()
X_synthetic_scaled = scaler.fit_transform(X_synthetic)
X_real_scaled = scaler.transform(X_real)


# Codificação das classes, exigência do XGBoost
le = LabelEncoder()
y_synthetic_enc = le.fit_transform(y_synthetic)
y_real_enc = le.transform(y_real)  # usa o mesmo encoder para manter consistência

# Espaço de busca de hiperparâmetros
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 1, 5],
    'reg_lambda': [1, 5, 10]
}



# Codificação das classes
le = LabelEncoder()
y_synthetic_enc = le.fit_transform(y_synthetic)
y_real_enc = le.transform(y_real)  # usa o mesmo encoder para manter consistência

# Espaço de busca de hiperparâmetros
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 1, 5],
    'reg_lambda': [1, 5, 10]
}

# Configuração do RandomizedSearchCV
search = RandomizedSearchCV(
    XGBClassifier(
        objective='multi:softmax',  # classificação multiclasse
        n_jobs=-1,                  # paralelismo em CPU
        random_state=42,
        eval_metric='mlogloss'      # métrica de avaliação
    ),
    param_distributions=param_dist,
    n_iter=20,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

##criada para obter métricas de desempenho
def train_model(): 
    search.fit(X_synthetic_scaled, y_synthetic_enc)
# Treino nos sintéticos
start = time.time()
mem_usage = memory_usage(train_model) 
end = time.time() 
print("Tempo de treino (s):", end - start) 
print("Uso de memória (MB):", max(mem_usage) - min(mem_usage))



print("Melhores parâmetros:", search.best_params_)
print("Melhor score (CV nos sintéticos):", search.best_score_)

# Modelo final
best_xgb = search.best_estimator_

# Teste cego nos reais
start = time.time()
y_pred_enc = best_xgb.predict(X_real_scaled) 
end = time.time() 

print("Tempo de predição (s):", end - start)

y_pred = le.inverse_transform(y_pred_enc)  # volta para os nomes originais

# Classes reais esperadas
real_classes = ['MM', 'MH', 'UN']

print("\nResultados no teste real (blind test):")
print("Accuracy:", accuracy_score(y_real, y_pred))
print("Precision:", precision_score(y_real, y_pred, average='weighted'))
print("Recall:", recall_score(y_real, y_pred, average='weighted'))
print("F1-score:", f1_score(y_real, y_pred, average='weighted'))

# Matriz de confusão e relatório restritos às 3 classes reais
print("Confusion Matrix:\n", confusion_matrix(y_real, y_pred, labels=real_classes))
print("\nRelatório de classificação:\n", classification_report(y_real, y_pred, labels=real_classes))


# Importância dos atributos com nomes originais
importances = best_xgb.feature_importances_
feature_names = X_synthetic.columns

plt.figure(figsize=(10,6))
plt.bar(range(len(importances)), importances)
plt.xticks(range(len(importances)), feature_names, rotation=90)
plt.title("Feature Importance (XGBoost)")
plt.show()


### ------ REFINIMENT 1 ---- #########


# Oversampling com SMOTE
sampling_strategy = {
    'MM': 1276,   # 876 originais + 400 novos
    'MH': 1055,   # 655 originais + 400 novos
    'UN': 52      # 37 originais + 15 novos
}

smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
X_syn_res, y_syn_res = smote.fit_resample(X_synthetic, y_synthetic)

print("Distribuição sintética original:\n", y_synthetic.value_counts())
print("Distribuição sintética após SMOTE:\n", pd.Series(y_syn_res).value_counts())

# Codificação das classes
le = LabelEncoder()
y_syn_res_enc = le.fit_transform(y_syn_res)
y_real_enc = le.transform(y_real)

# Escalar novamente
scaler = StandardScaler()
X_syn_res_scaled = scaler.fit_transform(X_syn_res)
X_real_scaled = scaler.transform(X_real)

# Espaço de busca para XGBoost
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 1, 5],
    'reg_lambda': [1, 5, 10]
}

search = RandomizedSearchCV(
    XGBClassifier(
        objective='multi:softmax',
        n_jobs=-1,
        random_state=42,
        eval_metric='mlogloss'
    ),
    param_distributions=param_dist,
    n_iter=20,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

# Ajuste da função de treino para usar os dados balanceados
def train_model_ref1():
    search.fit(X_syn_res_scaled, y_syn_res_enc)

# Medir tempo e memória de treino
start = time.time()
mem_usage = memory_usage(train_model_ref1)
end = time.time()

print("Tempo de treino (s):", end - start)
print("Uso de memória (MB):", max(mem_usage) - min(mem_usage))

print("Melhores parâmetros:", search.best_params_)
print("Melhor score (CV nos sintéticos balanceados):", search.best_score_)

# Modelo final
best_xgb = search.best_estimator_

# Medir tempo de predição
start = time.time()
y_pred_enc = best_xgb.predict(X_real_scaled)
end = time.time()
print("Tempo de predição (s):", end - start)

y_pred = le.inverse_transform(y_pred_enc)

# Classes reais esperadas
real_classes = ['MM', 'MH', 'UN']

print("\nResultados no teste real (blind test):")
print("Accuracy:", accuracy_score(y_real, y_pred))
print("Precision:", precision_score(y_real, y_pred, average='weighted'))
print("Recall:", recall_score(y_real, y_pred, average='weighted'))
print("F1-score:", f1_score(y_real, y_pred, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_real, y_pred, labels=real_classes))
print("\nRelatório de classificação:\n", classification_report(y_real, y_pred, labels=real_classes))

# Importância dos atributos com nomes originais
importances = best_xgb.feature_importances_
feature_names = X_synthetic.columns

plt.figure(figsize=(10,6))
plt.bar(range(len(importances)), importances)
plt.xticks(range(len(importances)), feature_names, rotation=90)
plt.title("Feature Importance (XGBoost com SMOTE)")
plt.show()



### -----------REFINEMENT 2 -------#

# Filtrar o sintético para manter só as classes presentes no real
classes_reais = ['MM', 'MH', 'UN']
mask = y_synthetic.isin(classes_reais)
X_syn_filtered = X_synthetic[mask]
y_syn_filtered = y_synthetic[mask]

print("Distribuição sintética filtrada:\n", y_syn_filtered.value_counts())

# Codificação das classes (apenas MM, MH, UN)
le = LabelEncoder()
y_syn_filtered_enc = le.fit_transform(y_syn_filtered)
y_real_enc = le.transform(y_real)

# Escalar novamente
scaler = StandardScaler()
X_syn_filtered_scaled = scaler.fit_transform(X_syn_filtered)
X_real_scaled = scaler.transform(X_real)

# Espaço de busca para XGBoost
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 1, 5],
    'reg_lambda': [1, 5, 10]
}

search = RandomizedSearchCV(
    XGBClassifier(
        objective='multi:softmax',
        n_jobs=-1,
        random_state=42,
        eval_metric='mlogloss'
    ),
    param_distributions=param_dist,
    n_iter=20,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

# Ajuste da função de treino para usar os dados filtrados
def train_model_ref2():
    search.fit(X_syn_filtered_scaled, y_syn_filtered_enc)

# Medir tempo e memória de treino
start = time.time()
mem_usage = memory_usage(train_model_ref2)
end = time.time()

print("Tempo de treino (s):", end - start)
print("Uso de memória (MB):", max(mem_usage) - min(mem_usage))

print("Melhores parâmetros:", search.best_params_)
print("Melhor score (CV nos sintéticos filtrados):", search.best_score_)

# Modelo final
best_xgb = search.best_estimator_

# Medir tempo de predição
start = time.time()
y_pred_enc = best_xgb.predict(X_real_scaled)
end = time.time()
print("Tempo de predição (s):", end - start)

y_pred = le.inverse_transform(y_pred_enc)

# Classes reais esperadas
real_classes = ['MM', 'MH', 'UN']

print("\nResultados no teste real (blind test):")
print("Accuracy:", accuracy_score(y_real, y_pred))
print("Precision:", precision_score(y_real, y_pred, average='weighted'))
print("Recall:", recall_score(y_real, y_pred, average='weighted'))
print("F1-score:", f1_score(y_real, y_pred, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_real, y_pred, labels=real_classes))
print("\nRelatório de classificação:\n", classification_report(y_real, y_pred, labels=real_classes))

# Importância dos atributos com nomes originais
importances = best_xgb.feature_importances_
feature_names = X_synthetic.columns

plt.figure(figsize=(10,6))
plt.bar(range(len(importances)), importances)
plt.xticks(range(len(importances)), feature_names, rotation=90)
plt.title("Feature Importance (XGBoost filtrado nas classes reais)")
plt.show()




#### -------- PIPELINE 70/30 ------------ ######
# Espaço de busca para XGBoost
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 1, 5],
    'reg_lambda': [1, 5, 10]
}

def run_pipeline(X, y, label):
    # Codificar labels em inteiros
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Split 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
    )
    
    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Randomized Search
    search = RandomizedSearchCV(
        XGBClassifier(
            objective='multi:softmax',
            n_jobs=-1,
            random_state=42,
            eval_metric='mlogloss'
        ),
        param_distributions=param_dist,
        n_iter=20,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    
    # Medir tempo e memória de treino
    def train_model_ref3():
        search.fit(X_train_scaled, y_train)
    
    start = time.time()
    mem_usage = memory_usage(train_model_ref3)
    end = time.time()
    
    print(f"\n[{label}] Tempo de treino (s):", end - start)
    print(f"[{label}] Uso de memória (MB):", max(mem_usage) - min(mem_usage))
    print(f"[{label}] Melhores parâmetros:", search.best_params_)
    print(f"[{label}] Melhor score (CV):", search.best_score_)
    
    # Modelo final
    best_xgb = search.best_estimator_
    
    # Medir tempo de predição
    start = time.time()
    y_pred_enc = best_xgb.predict(X_test_scaled)
    end = time.time()
    print(f"[{label}] Tempo de predição (s):", end - start)
    
    # Voltar para os nomes originais
    y_pred = le.inverse_transform(y_pred_enc)
    y_test_orig = le.inverse_transform(y_test)
    
    # Avaliar no teste
    print(f"\nResultados no teste ({label}):")
    print("Accuracy:", accuracy_score(y_test_orig, y_pred))
    print("Precision:", precision_score(y_test_orig, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test_orig, y_pred, average='weighted'))
    print("F1-score:", f1_score(y_test_orig, y_pred, average='weighted'))
    print("Confusion Matrix:\n", confusion_matrix(y_test_orig, y_pred))
    print("\nRelatório de classificação:\n", classification_report(y_test_orig, y_pred))

# 1. Sintético
run_pipeline(X_synthetic, y_synthetic, "Sintético")

# 2. Sintético + Real
X_concat = pd.concat([X_synthetic, X_real])
y_concat = pd.concat([y_synthetic, y_real])
run_pipeline(X_concat, y_concat, "Sintético + Real")

# 3. Real
run_pipeline(X_real, y_real, "Real")





