import pandas as pd
import os


def read_cityscapes():
    """
    Crea i retorna un dataframe dels fitxers labels format per les columnes
    filename (nom del fitxer semse extensió), ciutat, objecte, x, y, w, h i confidence.
    :return: pandas dataset amb el format anterior
    """
    header = ['filename', 'city', 'object', 'x', 'y', 'w', 'h', 'confidence']
    data = pd.DataFrame(None, columns=header)
    for file in os.listdir('dataset_cities/labels'):
        filename = file.split('.')[0]
        city = file.split('_')[0]
        with open("dataset_cities/labels/"+file, "r") as f:
            # la última línia de cada fitxer és buida
            lines = f.read().split('\n')[:-1]
            lines = map(lambda x: x.split(' '), lines)
            lines = [[filename, city, *line] for line in lines]
            df = pd.DataFrame(lines, columns=header)
            data = pd.concat([data, df], ignore_index=True)
    return data


def read_cityscapes_sample(n=100, labels_path='dataset_cities/labels/'):
    """
    Crea i retorna un dataframe dels n primers fitxers labels format per les columnes
    filename (nom del fitxer semse extensió), ciutat, objecte, x, y, w, h i confidence.
    :return: pandas dataset amb el format anterior
    """
    header = ['filename', 'city', 'object', 'x', 'y', 'w', 'h', 'confidence']
    data = pd.DataFrame(None, columns=header)
    count = 0
    for file in os.listdir(labels_path):
        if count == n:
            break
        filename = file.split('.')[0]
        city = file.split('_')[0]
        with open(labels_path + file, "r") as f:
            # la última línia de cada fitxer és buida
            lines = f.read().split('\n')[:-1]
            lines = map(lambda x: x.split(' '), lines)
            lines = [[filename, city, *line] for line in lines]
            df = pd.DataFrame(lines, columns=header)
            data = pd.concat([data, df], ignore_index=True)
        count += 1
    return data


def read_clean_sample(n=100, labels_path='dataset_cities/labels/'):
    """
    Crea el dataframe de treball amb els n primers fitxers i el neteja
    tal com es demana a l'enunciat.
    :param labels_path: directori dels fitxers YOLO
    :param n: número de fitxers
    :return: pandas dataframe
    """
    import read_and_validate.validate_cityscapes as vc
    data = read_cityscapes_sample(n, labels_path)
    error_rows = vc.validate_YOLO(data.iloc[:, 2:])
    # eliminem files incorrectes
    vc.remove_filenames(data, error_rows)
    return data[data["confidence"] > 0.4]
