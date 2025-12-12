import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf

try:
    
    base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    VGG16_FEATURE_MODEL = Model(inputs=base_model.input, outputs=base_model.output)
    print("VGG16 Deep Feature Model carregado com sucesso.")
except Exception as e:
    print(f"AVISO: Falha ao carregar VGG16 para Deep Features. O código continuará, mas essa feature não estará disponível. Erro: {e}")
    VGG16_FEATURE_MODEL = None
    
def load_and_preprocess_for_vgg(filepaths, target_size=(224, 224)):
    images = []
    for fp in filepaths:
        try:
            img = cv2.imread(fp)
            if img is None:
                raise Exception("Não foi possível ler a imagem com cv2.")
            img = cv2.resize(img, target_size)
            images.append(img)
        except Exception as e:
            print(f"ERRO ao carregar/redimensionar {fp} para VGG: {e}")
            
    images_array = np.array(images)
    return preprocess_input(images_array)


def extract_feature_vgg16(filepaths):
    if VGG16_FEATURE_MODEL is None:
        print("ERRO: VGG16 não está disponível. Pulando extração de Deep Features.")
        return None

    print(f"Extraindo VGG16 Deep Features para {len(filepaths)} imagens...")
    try:
        processed_images = load_and_preprocess_for_vgg(filepaths)
        features = VGG16_FEATURE_MODEL.predict(processed_images, batch_size=32)
        print(f"Formato das Deep Features: {features.shape}")
        return features
    except Exception as e:
        print(f"ERRO durante a extração de VGG16 features: {e}")
        return None


def extract_feature_hog(filepaths, target_size=(128, 128)):
    print(f"Extraindo HOG Features para {len(filepaths)} imagens...")
    hog = cv2.HOGDescriptor(_winSize=(target_size[0], target_size[1]),
                            _blockSize=(16, 16),
                            _blockStride=(8, 8),
                            _cellSize=(8, 8),
                            _nbins=9)
    features = []
    for fp in filepaths:
        try:
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if img is None:
                 raise Exception("Não foi possível ler a imagem.")
            img = cv2.resize(img, target_size)
            h = hog.compute(img)
            features.append(h.flatten())
        except Exception as e:
            print(f"ERRO ao extrair HOG de {fp}: {e}")
            features.append(np.zeros(hog.getDescriptorSize()))

    features_array = np.array(features)
    print(f"Formato das HOG Features: {features_array.shape}")
    return features_array


def extract_feature_color_histogram(filepaths, bins=(8, 8, 8), target_size=(128, 128)):
    print(f"Extraindo Color Histogram Features para {len(filepaths)} imagens...")
    features = []
    for fp in filepaths:
        try:
            img = cv2.imread(fp)
            if img is None:
                 raise Exception("Não foi possível ler a imagem.")
            img = cv2.resize(img, target_size)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            features.append(hist.flatten())
        except Exception as e:
            print(f"ERRO ao extrair Histograma de Cor de {fp}: {e}")
            features.append(np.zeros(np.prod(bins)))

    features_array = np.array(features)
    print(f"Formato das Color Histogram Features: {features_array.shape}")
    return features_array