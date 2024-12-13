import os
import pandas as pd
import time
import numpy as np
import datetime
import shutil

from utils import getPrecipitationsFromCsv
from config import Configuration
from model import Model

# Función para modificar el archivo CSV de entrada
def modify_csv(file_path, variation, output_file):
    df = pd.read_csv(file_path)
    df["Mean Temperature (C)"] = df["Mean Temperature (C)"] * (1 + variation)
    df.to_csv(output_file, index=False)
    print(f"Archivo temporal generado: {output_file}")

# Configuración del modelo
def myConf():
    configuration = Configuration('example.cfg')
    configuration.validate()
    configuration.save('myConf.cfg')
    return configuration

# Ejecutar la simulación para cada variación
if __name__ == '__main__':
    original_csv = "cordoba.csv"
    backup_csv = "cordoba_backup.csv"
    shutil.copy(original_csv, backup_csv)
    print(f"Archivo original respaldado en: {backup_csv}")

    # Variaciones porcentuales
    variations = [-0.20, -0.10, -0.05, -0.02, -0.01, 0.01, 0.02, 0.05, 0.10, 0.20]

    # Crear la carpeta de resultados si no existe
    results_dir = "sensitivity_ambientales"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Carpeta '{results_dir}' creada.")

    for variation in variations:
        # Modificar el archivo CSV con la variación actual
        modify_csv(backup_csv, variation, original_csv)

        # Configurar y ejecutar el modelo
        OUTPUT_FILENAME = os.path.join(results_dir, f"variation_T_{int(variation * 100)}.csv")
        configuration = myConf()

        # Inicializar y ejecutar el modelo
        model = Model(configuration)
        t1 = time.time()
        time_range, results = model.solveEquations()
        t2 = time.time()
        print(f'Variación {variation * 100}% completada en {t2-t1:.2f} segundos')

        # Procesar resultados
        indexOf = lambda t: (np.abs(time_range - t)).argmin()
        start_datetime = datetime.datetime.strptime(configuration.getString('simulation', 'start_date'), '%Y-%m-%d')
        dates = [(start_datetime + datetime.timedelta(days=t)) for t in time_range]

        parameters = model.parameters
        EGG = parameters.EGG
        LARVAE = parameters.LARVAE
        PUPAE = parameters.PUPAE
        ADULT1 = parameters.ADULT1
        ADULT2 = parameters.ADULT2
        WATER = parameters.WATER
        OVIPOSITION = parameters.OVIPOSITION
        BS_a = parameters.BS_a

        # Calcular las variables
        E = np.sum(results[:, EGG], axis=1) / BS_a
        L = np.sum(results[:, LARVAE], axis=1) / BS_a
        A = (results[:, ADULT1] + results[:, ADULT2]) / BS_a

        lwO = np.array([results[indexOf(t), OVIPOSITION] - results[indexOf(t - 7), OVIPOSITION] for t in time_range])
        lwO_mean = np.array([lwO[indexOf(t - 7):indexOf(t + 7)].mean(axis=0) for t in time_range])
        O = np.sum(lwO_mean, axis=1) / BS_a

        T = parameters.weather.T(time_range) - 273.15
        RH = parameters.weather.RH(time_range)
        P = parameters.weather.p(time_range)

        # Crear DataFrame con todas las columnas necesarias
        df = pd.DataFrame({
            'date': dates,
            'E': E,
            'L': L,
            'A': A,
            'O': O,
            'p': P,
            'T': T,
            'RH': RH
        })

        df.set_index('date', inplace=True)

        # Guardar resultados en la carpeta de resultados
        df.to_csv(OUTPUT_FILENAME)
        print(f"Resultados guardados en {OUTPUT_FILENAME}")

    # Restaurar el archivo original
    shutil.copy(backup_csv, original_csv)
    print(f"Archivo original restaurado: {original_csv}")

