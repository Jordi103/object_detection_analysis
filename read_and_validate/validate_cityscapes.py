import pandas as pd
import re
import matplotlib.image as im
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use('TKAgg')


def validate_YOLO(data: pd.DataFrame):
    """
    Comprovem que les dades en el dataframe d'entrada segueixen el format YOLO:
    cada fila ha d'estar formada per sis elements on el primer és un enter entre 0
    i 80 i la resta han de ser nombres decimals entre 0 i 1. A més, cada rectangle
    determinat per la fila ha d'estar dins de la imatge.
    Notem que cal que les columnes del dataframe siguin exactament en format YOLO,
    per tant aquesta funció no accepta la sortida de read_cityscapes directament.
    Returna una llista dels números de fila de les files invàlides.
    :param data: dataframe en format YOLO
    :return: llistat d'enters
    """
    floats01 = ['x', 'y', 'w', 'h', 'confidence']

    def correct_types(row):
        """
        Comprova que els valors de la fila siguin del format correcte.
        :param row: fila
        :return: True si són tots correctes False sinó
        """
        # object
        if re.search(r'^\d+$', row["object"]) is None:
            return False
        # x, y, w, h, confindence
        xywhcomp_floats = lambda x: re.search(r'^-?\d+\.\d+$', x) is None
        if any(map(xywhcomp_floats,
                   row[floats01])):
            return False
        # modify types
        row["object"] = int(row["object"])
        for var in floats01:
            row[var] = float(row[var])

        return True

    def correct_ranges(row):
        """
        Comprova si tots els valors de la fila són coherents amb YOLO.
        :param row: fila a comprovar
        :return: True si són tots correctes, False sinó
        """
        # object
        if row["object"] < 0 or row["object"] > 80:
            return False
        # x, y, w, h, confidence between 0 and 1
        if any(map(lambda x: row[x] < 0 or row[x] > 1, floats01)):
            return False
        # rectangle inside borders
        right = row['x'] + row['w']/2 > 1
        left = row['x'] - row['w']/2 < 0
        top = row['y'] + row['h']/2 > 1
        bottom = row['y'] - row['h']/2 < 0
        if any([right, left, top, bottom]):
            return False
        return True

    error_rows = []
    for i, row_iter in data.iterrows():
        if not correct_types(row_iter):
            error_rows.append(i)
            continue
        if not correct_ranges(row_iter):
            error_rows.append(i)

    return error_rows


def remove_filenames(data, row_indices):
    """
    Donat el  DataFrame original i un llistat d'índexos de files, per cada
    fila obtenim el nom de fitxer associat i eliminem totes les files del
    dataframe que provenen d'aquest fitxer.
    :param data: dataframe en format de read_cityscapes
    :param row_indices: llista d'índexos
    :return: None
    """
    # find files to remove
    ftr = data.loc[row_indices, "filename"].drop_duplicates().values
    # find row indices for that file
    itr = data.loc[data["filename"].isin(ftr)].index.values
    # delete rows in itr
    data.drop(index=itr, inplace=True)
    # change type
    num_vars = ["object", "x", "y", "w", "h", "confidence"]
    data[num_vars] = data[num_vars].apply(pd.to_numeric)


def validate_images(data, image_directory="./dataset_cities/images/"):
    """
    Dibuixa rectangles sobre la primera foto de cada ciutat. Guarda les imatges
    en la carpeta output/validation_rectangle
    :param data: dataframe obtingut de read_cityscapes
    :param image_directory: directori on estan les imatges
    :return: None
    """

    def find_images():
        """
        Troba la primera foto de cada ciutat.
        :return: un diccionari amb les ciutats com a claus i les imatges
        com a valors.
        """
        image_by_city = {}
        cities = data["city"].unique()
        for city in cities:
            files = data.loc[data["city"] == city, "filename"].values
            image_by_city[city] = sorted(files)[0]
        return image_by_city

    def draw_and_show_wrong(image_by_city):
        """
        Donat el diccionari que associa a cada ciutat la primera foto (en
        ordre lexicogràfic), dibuixa els rectangles del fitxer YOLO i guarda
        la imatge.
        :param image_by_city: diccionari ciutat imatge
        :return:
        """
        for city, filename in image_by_city.items():
            img = im.imread("./dataset_cities/images/" + filename + ".png")
            # draw actual image
            fig, ax = plt.subplots(1)
            ax.imshow(img)
            # draw rectangles
            objects = data[data["filename"] == filename]
            W, H = 2048, 1024
            for _, row in objects.iterrows():
                rect = patches.Rectangle(((row['x'] - row['w'] / 2)*W, (row['y'] - row['h'] / 2)*H),
                                         row['w']*W, row['h']*H,
                                         edgecolor='r', facecolor="none")

                ax.add_patch(rect)
            # plt.savefig('output/validation_rectangle/' + city + '.png')
            plt.show()

    def draw_and_show(filename):
        """
        Donat el diccionari que associa a cada ciutat la primera foto (en
        ordre lexicogràfic), dibuixa els rectangles del fitxer YOLO i guarda
        la imatge.
        :param filename: nom de la imatge a dibuixar, sense extensió
        :return:
        """
        img = im.imread(image_directory + filename + ".png")
        # draw actual image
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        # draw rectangles
        objects = data[data["filename"] == filename]
        W, H = 2048, 1024
        for _, row in objects.iterrows():
            rect = patches.Rectangle(((row['x'] - row['w'] / 2)*W, (row['y'] - row['h'] / 2)*H),
                                     row['w']*W, row['h']*H,
                                     edgecolor='r', facecolor="none")

            ax.add_patch(rect)
        # plt.savefig('output/validation_rectangle/' + city + '.png')
        plt.show()

    for city, filename in find_images().items():
        draw_and_show(filename)


if __name__ == '__main__':
    from read_and_validate.read_cityscapes import read_clean_sample
    data = read_clean_sample(n=-1, labels_path='../dataset_cities/labels/')
    validate_YOLO(data.iloc[:, 2:])
