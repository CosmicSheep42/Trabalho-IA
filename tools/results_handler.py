import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

def generate_confusion_matrix(y_true, y_pred, classes, filename, results_dir):
    """
    Gera e salva a matriz de confusão normalizada
    """
    print(f"Gerando Matriz de Confusão para {filename}...")
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.round(cm_normalized * 100, 2)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, 
                annot=True, 
                fmt=".2f", 
                cmap="Blues",
                xticklabels=classes, 
                yticklabels=classes,
                cbar_kws={'label': 'Percentual (%)'})
    
    plt.title(f'Matriz de Confusão Normalizada - {filename}')
    plt.ylabel('Classe Verdadeira (True Label)')
    plt.xlabel('Classe Predita (Predicted Label)')
    
    save_path = os.path.join(results_dir, f'matriz_confusao_{filename}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Matriz de Confusão salva em: {save_path}")
    
    return cm_normalized 


def generate_results_report(results_dict, results_dir):
    
    report_path = os.path.join(results_dir, "relatorio_desempenho.md")
    
    with open(report_path, 'w', encoding='utf-8') as f: 
        f.write("# Relatório de Desempenho do Sistema\n\n")
        f.write("Este relatório resume as taxas de classificação (Acurácia e F1-Score) obtidas no Cross-Validation (Média) e no Teste Fixo (Final).\n\n")
        f.write("## 1. Desempenho Médio no Cross-Validation (10 Folds)\n")
        f.write("| Classificador | Feature Extractor | Acurácia Média (%) | F1-Score Médio (%) |\n")
        f.write("| :--- | :--- | :---: | :---: |\n")
        
        for key, value in results_dict.items():
            if 'CV' in key:
                clf_name, feature_name = key.replace('_CV', '').split('_with_')
                f.write(f"| {clf_name} | {feature_name} | {value['mean_acc']:.2f} | {value['mean_f1']:.2f} |\n")
        
        f.write("\n")
        
        f.write("## 2. Desempenho Final no Conjunto de Teste Fixo (Avaliação Final)\n")
        f.write("| Classificador | Feature Extractor | Acurácia (%) | F1-Score (%) |\n")
        f.write("| :--- | :--- | :---: | :---: |\n")

        ensemble_info = None
        for key, value in results_dict.items():
            if 'TEST' in key:
                if 'Ensemble' in key:
                    ensemble_info = value
                    continue
                    
                clf_name, feature_name = key.replace('_TEST', '').split('_with_')
                f.write(f"| {clf_name} | {feature_name} | {value['acc']:.2f} | {value['f1']:.2f} |\n")

        if ensemble_info:
            f.write("\n---\n")
            f.write("## 3. Impacto da Combinação Estática de Classificadores (20 CLFs)\n")
            f.write(f"**Método de Combinação:** Soft Voting\n")
            f.write(f"**Feature:** {ensemble_info['feature_name']}\n")
            f.write(f"**Acurácia no Teste Fixo:** **{ensemble_info['acc']:.2f}%**\n")
            f.write(f"**F1-Score no Teste Fixo:** **{ensemble_info['f1']:.2f}%**\n")
            f.write("*(O Voting Classifier tipicamente utiliza a melhor feature disponível para a fusão)*\n")

    print(f"Relatório de Desempenho salvo em: {report_path}")