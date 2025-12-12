import os
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import joblib

from tools.data_loader import load_simpsons_dataset, load_image_data, get_cross_validation_splits
from tools.feature_extractor import extract_feature_hog, extract_feature_color_histogram, extract_feature_vgg16
from tools.classifier_model import get_base_classifiers, get_diverse_classifiers, train_and_evaluate_classifier, create_static_ensemble
from tools.results_handler import generate_confusion_matrix, generate_results_report

# Configs básicas
DATA_DIR = 'data'
RESULTS_DIR = 'resultados'
N_FOLDS_CV = 10 # N° de folds p cross-validation 
N_ENSEMBLE = 20 # N° de classificadores para fusão

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def main():
    start_time = time.time()

    all_results = {}
    
    # Carregamento e Divisão do Dataset
    try:
        X_train_paths, y_train, X_test_paths, y_test, classes = load_simpsons_dataset(DATA_DIR)
        N_CLASSES = len(classes)
    except FileNotFoundError as e:
        print(f"ERRO CRÍTICO: {e}. Verifique se a pasta '{DATA_DIR}' e as subpastas de classes estão corretas.")
        return
    
    # Extração de Características
    feature_extractors = {
        'HOG': extract_feature_hog,
        'ColorHistogram': extract_feature_color_histogram,
        'VGG16_DeepFeatures': extract_feature_vgg16, 
    }
    
    features_cache = {}
    scaler_cache = {}
    
    for feature_name, extractor_func in feature_extractors.items():
        print(f"\n==================== FEATURE: {feature_name} ====================")
        X_train_features = extractor_func(X_train_paths)
        X_test_features = extractor_func(X_test_paths)
        
        if X_train_features is None:
            continue
        scaler = StandardScaler()
        scaler.fit(X_train_features)

        features_cache[feature_name] = {'train': X_train_features, 'test': X_test_features}
        scaler_cache[feature_name] = scaler
        
    if not features_cache:
        print("ERRO CRÍTICO: Nenhuma característica pôde ser extraída. O código será encerrado.")
        return
    
    base_classifiers = get_base_classifiers()
    
    for feature_name, features_data in features_cache.items():
        X_train = features_data['train']
        X_test = features_data['test']
        scaler = scaler_cache[feature_name]
        
        for clf_name, clf_model in base_classifiers.items():
            print(f"\n--- Treinando {clf_name} com {feature_name} ---")

            cv_scores_acc = []
            cv_scores_f1 = []
            
            cv_splits = get_cross_validation_splits(X_train, y_train, N_FOLDS_CV)
            
            for fold_idx, (train_index, val_index) in enumerate(cv_splits):
                X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
                
                acc, f1, _, _, _ = train_and_evaluate_classifier(
                    clf_model, X_train_fold, y_train_fold, X_val_fold, y_val_fold, 
                    scaler=StandardScaler().fit(X_train_fold)
                )

                if acc is not None:
                    cv_scores_acc.append(acc)
                    cv_scores_f1.append(f1)
                
            if cv_scores_acc:
                mean_acc = np.mean(cv_scores_acc)
                mean_f1 = np.mean(cv_scores_f1)
                print(f"CV Média (10 Folds): Acurácia: {mean_acc:.2f}%, F1-Score: {mean_f1:.2f}%")
                
                all_results[f'{clf_name}_with_{feature_name}_CV'] = {
                    'mean_acc': mean_acc, 
                    'mean_f1': mean_f1
                }

            # Avaliação Final no Conjunto de Teste Fixo
            clf_final = get_base_classifiers()[clf_name]
            acc, f1, y_pred, _, _ = train_and_evaluate_classifier(
                clf_final, X_train, y_train, X_test, y_test, scaler=scaler
            )
            
            if acc is not None:
                print(f"Teste Fixo: Acurácia: {acc:.2f}%, F1-Score: {f1:.2f}%")
                
                all_results[f'{clf_name}_with_{feature_name}_TEST'] = {
                    'acc': acc, 
                    'f1': f1,
                    'y_pred': y_pred
                }

                generate_confusion_matrix(y_test, y_pred, classes, 
                                          f'{clf_name}_with_{feature_name}_TEST', RESULTS_DIR)


    # Combinação Estática de Classificadores 
    
    best_feature_name = 'VGG16_DeepFeatures' if 'VGG16_DeepFeatures' in features_cache else 'HOG'
    print(f"\n==================== COMBINAÇÃO ESTÁTICA (ENSEMBLE) com {best_feature_name} ====================")
    
    X_train_ensemble = features_cache[best_feature_name]['train']
    X_test_ensemble = features_cache[best_feature_name]['test']
    scaler_ensemble = scaler_cache[best_feature_name]

    diverse_clfs = get_diverse_classifiers(N_ENSEMBLE)

    fitted_diverse_clfs = []
    
    X_train_scaled = scaler_ensemble.transform(X_train_ensemble)
    
    for clf_id, clf in diverse_clfs:
        try:
            clf.fit(X_train_scaled, y_train)
            fitted_diverse_clfs.append((clf_id, clf))
        except Exception as e:
            print(f"AVISO: Falha ao treinar classificador para o Ensemble ({clf_id}): {e}")

    ensemble_clf = create_static_ensemble(fitted_diverse_clfs, voting='soft')
 
    print("Treinando o Voting Classifier (Soft Voting)...")
    ensemble_clf.fit(X_train_scaled, y_train)

    X_test_scaled = scaler_ensemble.transform(X_test_ensemble)
    y_pred_ensemble = ensemble_clf.predict(X_test_scaled)

    acc_ensemble = accuracy_score(y_test, y_pred_ensemble) * 100
    f1_ensemble = f1_score(y_test, y_pred_ensemble, average='macro') * 100
    
    print(f"Resultado Final Ensemble: Acurácia: {acc_ensemble:.2f}%, F1-Score: {f1_ensemble:.2f}%")

    all_results[f'Ensemble_with_{best_feature_name}_TEST'] = {
        'acc': acc_ensemble, 
        'f1': f1_ensemble,
        'y_pred': y_pred_ensemble,
        'feature_name': best_feature_name
    }
    
    # Gerar matriz de confusão
    generate_confusion_matrix(y_test, y_pred_ensemble, classes, 
                              f'Ensemble_with_{best_feature_name}_TEST', RESULTS_DIR)

    # Gerar do relatório de desempenho
    generate_results_report(all_results, RESULTS_DIR)

    end_time = time.time()
    print(f"\n--- FIM do Processamento ---")
    print(f"Tempo total de execução: {(end_time - start_time):.2f} segundos.")
    print(f"Resultados e Gráficos salvos na pasta: '{RESULTS_DIR}'.")
    print("O artigo deverá detalhar todos estes resultados, matrizes e análise de pontos fracos/fortes.")

    # Salvar modelo
    print("\n--- Salvando o Modelo Treinado ---")
    model_filename = os.path.join(RESULTS_DIR, 'simpsons_ensemble_model.pkl')
    scaler_filename = os.path.join(RESULTS_DIR, 'simpsons_scaler.pkl')
    classes_filename = os.path.join(RESULTS_DIR, 'classes_names.pkl')
    
    # Salva o classificador, o scaler e os nomes das classes
    joblib.dump(ensemble_clf, model_filename)
    joblib.dump(scaler_ensemble, scaler_filename)
    joblib.dump(classes, classes_filename)
    
    print(f"Modelo salvo em: {model_filename}")
    print(f"Scaler salvo em: {scaler_filename}")

    end_time = time.time()

if __name__ == "__main__":
    main()