import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

def get_base_classifiers():
    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'SVM': SVC(kernel='linear', C=1.0, random_state=42, probability=True),
        'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, alpha=1e-4,
                             solver='adam', random_state=42, early_stopping=True),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    }
    return classifiers

def get_diverse_classifiers(n_estimators=20):
    
    diverse_classifiers = []
    for i in range(5):
        diverse_classifiers.append((f'RF_{i}', RandomForestClassifier(n_estimators=20 * (i + 1), max_depth=5 * (i + 1) + 5, random_state=42 + i)))
    for k in [1, 3, 5, 7, 9]:
        diverse_classifiers.append((f'KNN_{k}', KNeighborsClassifier(n_neighbors=k)))
    for i, kernel in enumerate(['linear', 'rbf', 'poly']):
        C = 0.1 * (i + 1)
        diverse_classifiers.append((f'SVM_{kernel}_{C}', SVC(kernel=kernel, C=C, random_state=42 + i, probability=True)))

    diverse_classifiers.append(('MLP_100', MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42, early_stopping=True)))
    diverse_classifiers.append(('MLP_50_50', MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=200, random_state=43, early_stopping=True)))
    
    base_names = list(get_base_classifiers().keys())
    for name, clf in get_base_classifiers().items():
        if name not in [n for n, c in diverse_classifiers]:
             diverse_classifiers.append((name, clf))

    if len(diverse_classifiers) < n_estimators:
        print(f"AVISO: Menos de {n_estimators} classificadores diversos gerados. Usando Random Forest extra.")
        for i in range(len(diverse_classifiers), n_estimators):
            diverse_classifiers.append((f'RF_Extra_{i}', RandomForestClassifier(n_estimators=50, random_state=100 + i)))

    return diverse_classifiers[:n_estimators]


def train_and_evaluate_classifier(clf, X_train, y_train, X_test, y_test, scaler=None):

    if scaler is None:
        scaler = StandardScaler().fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    try:
        clf.fit(X_train_scaled, y_train)
    except Exception as e:
        print(f"ERRO no treinamento do {clf.__class__.__name__}: {e}")
        return None, None, None, scaler

    y_pred = clf.predict(X_test_scaled)
    
    try:
        y_proba = clf.predict_proba(X_test_scaled)
    except:
        y_proba = None 

    acc = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average='macro') * 100
    
    return acc, f1, y_pred, scaler, y_proba


def create_static_ensemble(base_classifiers_list, voting='soft'):
    return VotingClassifier(estimators=base_classifiers_list, voting=voting)