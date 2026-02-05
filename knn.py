# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 21:12:25 2026

@author: nepor
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report



df_real_data = pd.read_csv('D:/Codigos_VSCODE/IME/projectiles_articles/Dataset/RAMP-RT real v1.csv')
df_synthetic_data = pd.read_csv('D:/Codigos_VSCODE/IME/projectiles_articles/Dataset/RAMP-RT synthetic v1.csv')

print(df_real_data.columns)

class_synthetic_count = df_synthetic_data['class'].value_counts()
class_real_count = df_real_data['class'].value_counts()

#count per class
print(class_synthetic_count)
print(class_real_count)

#---------------PIPELINE--------------#



#classification target and data
X_synthetic = df_synthetic_data.drop('class', axis=1)
y_synthetic = df_synthetic_data['class']

X_real = df_real_data.drop('class', axis=1)
y_real = df_real_data['class']


#Normal Scale
scaler = StandardScaler() 
X_synthetic_scaled = scaler.fit_transform(X_synthetic)
X_real_scaled = scaler.transform(X_real)

#Cross Validation - some tests to take the best model
cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 
k_values = [3, 5, 7, 9, 11, 13, 15] 
best_k = None
best_score = 0


for k in k_values: 
    knn_model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_model, X_synthetic_scaled, y_synthetic, cv=cross_validator, scoring='accuracy')
    mean_score = scores.mean()
    print(f"k={k}, mean accuracy={mean_score:.4f}")
    
    if mean_score > best_score:
        best_score = mean_score
        best_k = k
        
        
print(f"\nBest k: {best_k} with mean_value {best_score:.4f}")

#Metrics of cross validation with synthetic data

knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_synthetic_scaled, y_synthetic)

y_synthetic_pred = knn_model.predict(X_synthetic_scaled)

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
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_synthetic_scaled, y_synthetic)

# Prever nos dados reais
y_real_pred = knn_model.predict(X_real_scaled)

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


#-------REFINEMENTS--------------#



#-------REFINEMENT 1: Training only with MM, MH, UN--------------#

# Filter synthetic dataset to keep only MM, MH, UN
df_synthetic_filtered = df_synthetic_data[df_synthetic_data['class'].isin(['MM', 'MH', 'UN'])]

X_synthetic_filt = df_synthetic_filtered.drop('class', axis=1)
y_synthetic_filt = df_synthetic_filtered['class']

# Scale again with filtered data
scaler = StandardScaler()
X_synthetic_filt_scaled = scaler.fit_transform(X_synthetic_filt)
X_real_scaled = scaler.transform(X_real)  

# Cross Validation with filtered synthetic data
cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_k = None
best_score = 0

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_model, X_synthetic_filt_scaled, y_synthetic_filt, cv=cross_validator, scoring='accuracy')
    mean_score = scores.mean()
    print(f"[Refinement 1] k={k}, mean accuracy={mean_score:.4f}")
    
    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print(f"\n[Refinement 1] Best k: {best_k} with mean_value {best_score:.4f}")

# Train final model with filtered synthetic data
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_synthetic_filt_scaled, y_synthetic_filt)

# Predict on real data
y_real_pred = knn_model.predict(X_real_scaled)

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


#-------REFINEMENT 2: Oversampling MM and MH--------------#

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

# Cross Validation
best_k = None
best_score = 0

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_model, X_synthetic_weighted_scaled, y_synthetic_weighted,
                             cv=cross_validator, scoring='accuracy')
    mean_score = scores.mean()
    print(f"[Refinement 2] k={k}, mean accuracy={mean_score:.4f}")
    
    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print(f"\n[Refinement 2] Best k: {best_k} with mean_value {best_score:.4f}")

# Train final model with oversampled synthetic data
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_synthetic_weighted_scaled, y_synthetic_weighted)

# Metrics on synthetic data
y_synthetic_pred = knn_model.predict(X_synthetic_weighted_scaled)

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
y_real_pred = knn_model.predict(X_real_scaled)

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


#-------REFINEMENT 3: Comparison of splits--------------#


# 1. Split 70/30 only synthetic
X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(
    X_synthetic_filt_scaled, y_synthetic_filt, test_size=0.3, stratify=y_synthetic_filt, random_state=42
)

# 2. Split 70/30 synthetic + real (superbase)
df_superbase = pd.concat([df_synthetic_data, df_real_data], ignore_index=True)
X_super = df_superbase.drop('class', axis=1)
y_super = df_superbase['class']
X_super_scaled = scaler.fit_transform(X_super)

X_train_super, X_test_super, y_train_super, y_test_super = train_test_split(
    X_super_scaled, y_super, test_size=0.3, stratify=y_super, random_state=42
)

# 3. Split 70/30 only real
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X_real_scaled, y_real, test_size=0.3, stratify=y_real, random_state=42
)

# --- evaluate cross validation and metrics ---
def evaluate_split(X_train, y_train, X_test, y_test, label):
    best_k = None
    best_score = 0
    for k in k_values:
        knn_model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn_model, X_train, y_train, cv=cross_validator, scoring='accuracy')
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_k = k
    print(f"\n[Refinement 3] {label} - Best k: {best_k}, CV mean acc={best_score:.4f}")
    
    knn_model = KNeighborsClassifier(n_neighbors=best_k)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    
    print(f"[Refinement 3] {label} - Test metrics:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1-score:", f1_score(y_test, y_pred, average='weighted'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Run evaluations
evaluate_split(X_train_syn, y_train_syn, X_test_syn, y_test_syn, "Synthetic only")
evaluate_split(X_train_super, y_train_super, X_test_super, y_test_super, "Synthetic + Real")
evaluate_split(X_train_real, y_train_real, X_test_real, y_test_real, "Real only")


#-----Refinamento 4: compare metrics------#
# Synthetic only
print("\n[Refinement 4] Synthetic only - Class-wise metrics:")
print(classification_report(y_test_syn, knn_model.predict(X_test_syn)))

# Synthetic + Real
print("\n[Refinement 4] Synthetic + Real - Class-wise metrics:")
print(classification_report(y_test_super, knn_model.predict(X_test_super)))

# Real only
print("\n[Refinement 4] Real only - Class-wise metrics:")
print(classification_report(y_test_real, knn_model.predict(X_test_real)))




