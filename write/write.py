

def write_working_data(data, wrong_filenames):
    """
    Escriu per cada fitxer del dataset el nom de la imatge, el número de cotxes,
    el número de semàfors, el número de persones, ciutat, any i si és o no una
    imatge detectada com a intrusa.
    :param data: pandas dataframe en el format retornat per read_cityscapes
    :param wrong_filenames: llista de noms de fitxers, sense extensió
    :return: el path i el nom del fitxer csv de sortida
    """
    with open('output/image_info.csv', "w") as f:
        print("Escrivint a output/image_info.csv...")
        header = "nom imatge,número de cotxes,número de semàfors,número de persones"
        header += "cutat,any,ciutat correcta"
        f.write(header + '\n')
        for filename in data["filename"].unique():
            sel_rows = data[data["filename"] == filename]
            num_cars = sel_rows[sel_rows["object"] == 2].shape[0]
            num_smph = sel_rows[sel_rows["object"] == 9].shape[0]
            num_ppl = sel_rows[sel_rows["object"] == 0].shape[0]
            city = sel_rows["city"].values[0]
            year = sel_rows["filename"].values[0].split('-')[-1]
            if filename in wrong_filenames:
                correct_city = 'n'
            else:
                correct_city = 'y'
            row = [filename, num_cars, num_smph, num_ppl, city, year, correct_city]
            f.write(','.join(map(lambda x: str(x), row)) + '\n')
        print("Fet")
