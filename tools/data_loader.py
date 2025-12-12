import os
import cv2
import numpy as np
from sklearn.model_selection import StratifiedKFold
from PIL import Image

def load_simpsons_dataset(data_dir):
    """
    Carrega o dataset dos Simpsons esperando que dentro da pasta "data" existam subpastas
    nomeadas como classes individuais.
    Realiza uma divisão 80/20 train/test estratificada, além da separação por classes.
    """
    print("--- Carregando dados ---")
    
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not classes:
        raise FileNotFoundError(f"Nenhuma subpasta de classes encontrada em: {data_dir}")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    all_filepaths = []
    all_labels = []

    for class_name, class_idx in class_to_idx.items():
        class_path = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_path):
            if filename.endswith(('.bmp', '.png', '.jpg')):
                all_filepaths.append(os.path.join(class_path, filename))
                all_labels.append(class_idx)

    print(f"Total de imagens carregadas: {len(all_filepaths)}")
    print(f"Classes encontradas: {classes}")
    
    # Agrupar por classe
    class_files = {c: [] for c in classes}
    for filepath, label in zip(all_filepaths, all_labels):
        class_name = classes[label]
        class_files[class_name].append(filepath)

    X_train_paths, y_train, X_test_paths, y_test = [], [], [], []
    
    # Divisão 80/20
    for i, class_name in enumerate(classes):
        files = sorted(class_files[class_name])
        split_point = int(0.8 * len(files))
        
        # Treino
        X_train_paths.extend(files[:split_point])
        y_train.extend([i] * len(files[:split_point]))
        
        # Teste
        X_test_paths.extend(files[split_point:])
        y_test.extend([i] * len(files[split_point:]))

    print(f"Conjunto de Treino: {len(X_train_paths)} imagens")
    print(f"Conjunto de Teste/Validação Fixo: {len(X_test_paths)} imagens")
    
    return X_train_paths, np.array(y_train), X_test_paths, np.array(y_test), classes

def load_image_data(filepaths, target_size=(128, 128)):
    images = []
    for fp in filepaths:
        try:
            img = Image.open(fp).convert('RGB')
            img = np.array(img.resize(target_size))
            images.append(img)
        except Exception as e:
            print(f"ERRO ao carregar ou processar a imagem {fp}: {e}")

    return np.array(images, dtype='float32') / 255.0

def get_cross_validation_splits(X_train_paths, y_train, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return skf.split(X_train_paths, y_train)