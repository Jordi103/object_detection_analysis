import unittest
import read_and_validate.validate_cityscapes as vc
import analysis.other as oth
import pandas as pd

def yolo_row_to_df(row):
    """
    Crea un pandas DataFrame amb les columnes del format YOLO que conté
    fila d'entrada coma  a única fila.
    :param row: llista python en format YOLO
    :return: pandas dataframe en format YOLo
    """
    header = ['object', 'x', 'y', 'w', 'h', 'confidence']
    row = row.split(' ')
    return pd.DataFrame({header[i]: row[i] for i in range(len(header))}, index=[0])


class ValidateYOLO(unittest.TestCase):

    def test_types(self):
        testrow = '56 0.779541 0.482422 0.0268555 0.0839844 0.203185'
        self.assertTrue(vc.validate_YOLO(yolo_row_to_df(testrow)) == [])

        testrow = '2 0.162842 0.474121 0.150879 0.108398 0.529135'
        self.assertTrue(vc.validate_YOLO(yolo_row_to_df(testrow)) == [])

        # objecte no és un número
        testrow = '2a 0.162842 0.474121 0.150879 0.108398 0.529135'
        self.assertFalse(vc.validate_YOLO(yolo_row_to_df(testrow)) == [])

        # coma enlloc de punt
        testrow = '2 0.162842 0,474121 0.150879 0.108398 0.529135'
        self.assertFalse(vc.validate_YOLO(yolo_row_to_df(testrow)) == [])

        # objecte no és enter
        testrow = '2.0 0.162842 0.474121 1.1 0.108398 0.529135'
        self.assertFalse(vc.validate_YOLO(yolo_row_to_df(testrow)) == [])

        # y, w, h, confidence no són decimals
        testrow = '7 0.168213 23 13 78 956'
        self.assertFalse(vc.validate_YOLO(yolo_row_to_df(testrow)) == [])

    def test_values(self):
        testrow = '0 0.779541 0.482422 0.0268555 0.0839844 0.203185'
        self.assertTrue(vc.validate_YOLO(yolo_row_to_df(testrow)) == [])

        testrow = '2 0.162842 0.474121 0.150879 0.108398 0.529135'
        self.assertTrue(vc.validate_YOLO(yolo_row_to_df(testrow)) == [])

        # número d'objecte > 80
        testrow = '93 0.162842 0.474121 0.150879 0.108398 0.529135'
        self.assertFalse(vc.validate_YOLO(yolo_row_to_df(testrow)) == [])

        # x > 1
        testrow = '3 1.2 0.474121 0.150879 0.108398 0.529135'
        self.assertFalse(vc.validate_YOLO(yolo_row_to_df(testrow)) == [])

        # x + w/2 > 1
        testrow = '2 0.162842 0.474121 15.0 0.108398 0.529135'
        self.assertFalse(vc.validate_YOLO(yolo_row_to_df(testrow)) == [])

        # confidence negatiu
        testrow = '2.0 0.162842 0.474121 1.1 0.108398 -0.5'
        self.assertFalse(vc.validate_YOLO(yolo_row_to_df(testrow)) == [])


def object_string_to_df(objects):
    """
    Pren per entrada una llista de codis d'objectes i retorna un dataframe
    en el format de treball amb els codis en la columna object i valors
    qualssevol en les altres columnes.
    :param ostring: cadena de codis d'objecte separats per espais
    :return:
    """
    header = ['filename', 'object', 'x', 'y', 'w', 'h', 'confidence']
    rows = [['file', obj] + 5*[0] for obj in objects]
    return pd.DataFrame(rows, columns=header)


class ValidateTop3Dict(unittest.TestCase):

    def test_top3(self):
        # menys de 3 objectes
        testlist, realtop3 = [5, 4], [5, 4]
        top3 = oth.extract_top3(object_string_to_df(testlist), 'file')
        self.assertTrue(all([o in realtop3 for o in top3]) and len(top3) == len(realtop3))

        # exactament 3 objectes
        testlist, realtop3 = [5, 4, 4], [5, 4]
        top3 = oth.extract_top3(object_string_to_df(testlist), 'file')
        self.assertTrue(all([o in realtop3 for o in top3]) and len(top3) == len(realtop3))

        # frequencia més alta en més de 3 objectes
        testlist, realtop3 = [5, 5, 4, 4, 2, 2, 1, 1], [5, 4, 2, 1]
        top3 = oth.extract_top3(object_string_to_df(testlist), 'file')
        self.assertTrue(all([o in realtop3 for o in top3]) and len(top3) == len(realtop3))

        # no empat entre tercer i quart
        testlist, realtop3 = [5, 5, 4, 4, 2, 2, 1, 0], [5, 4, 2]
        top3 = oth.extract_top3(object_string_to_df(testlist), 'file')
        self.assertTrue(all([o in realtop3 for o in top3]) and len(top3) == len(realtop3))

        # empat entre tercer i quart
        testlist, realtop3 = [5, 5, 4, 4, 2, 1, 0], [5, 4]
        top3 = oth.extract_top3(object_string_to_df(testlist), 'file')
        self.assertTrue(all([o in realtop3 for o in top3]) and len(top3) == len(realtop3))

        # empat entre segon i tercer
        testlist, realtop3 = [5, 5, 5, 4, 4, 4, 2, 2, 1, 1, 0], [5, 4]
        top3 = oth.extract_top3(object_string_to_df(testlist), 'file')
        self.assertTrue(all([o in realtop3 for o in top3]) and len(top3) == len(realtop3))

        # no empat entre tercer i quart
        testlist, realtop3 = [5, 5, 5, 4, 4, 2, 2, 1, 1, 0], [5]
        top3 = oth.extract_top3(object_string_to_df(testlist), 'file')
        self.assertTrue(all([o in realtop3 for o in top3]) and len(top3) == len(realtop3))

        # no empat
        testlist, realtop3 = [5, 5, 5, 5, 4, 4, 4, 2, 2, 1, 0], [5, 4, 2]
        top3 = oth.extract_top3(object_string_to_df(testlist), 'file')
        self.assertTrue(all([o in realtop3 for o in top3]) and len(top3) == len(realtop3))


if __name__ == '__main__':
    suite_YOLO = unittest.TestLoader().loadTestsFromTestCase(ValidateYOLO)
    suite_top3 = unittest.TestLoader().loadTestsFromTestCase(ValidateTop3Dict)

    print("Executant tests per validate_YOLO...")
    unittest.TextTestRunner(verbosity=2).run(suite_YOLO)
    print("Fet")

    print("Executant tests per extract_top3...")
    unittest.TextTestRunner(verbosity=2).run(suite_top3)
    print("Fet")
