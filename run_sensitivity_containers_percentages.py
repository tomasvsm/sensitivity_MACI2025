import time
import numpy as np
import pandas as pd
import datetime
import os

from utils import getPrecipitationsFromCsv
from config import Configuration
from model import Model

def myConf(dictionary=None):
    configuration = Configuration('example.cfg')

    if dictionary:
        for section in dictionary:
            for key in dictionary[section]:
                value = dictionary[section][key]
                if isinstance(value, list):
                    configuration.config_parser.set(section, key, ','.join([str(x) for x in value]))
                else:
                    configuration.config_parser.set(section, key, str(value))

    configuration.validate()
    configuration.save('myConf.cfg')
    return configuration


def run_and_save_model(configuration, output_path):
    # Ejecuta el modelo
    model = Model(configuration)
    t1 = time.time()
    time_range, results = model.solveEquations()
    t2 = time.time()
    print('Elapsed time: ', t2 - t1)

    # Calcular fechas
    start_datetime = datetime.datetime.strptime(
        configuration.getString('simulation', 'start_date'), '%Y-%m-%d')
    dates = [(start_datetime + datetime.timedelta(days=t)) for t in time_range]

    # Extraer parámetros del modelo
    parameters = model.parameters
    EGG = parameters.EGG
    LARVAE = parameters.LARVAE
    ADULT1 = parameters.ADULT1
    ADULT2 = parameters.ADULT2
    OVIPOSITION = parameters.OVIPOSITION
    BS_a = parameters.BS_a

    E = np.sum(results[:, EGG], axis=1) / BS_a
    L = np.sum(results[:, LARVAE], axis=1) / BS_a
    A = (results[:, ADULT1] + results[:, ADULT2]) / BS_a

    T = parameters.weather.T(time_range) - 273.15  # Convertir a Celsius
    P = parameters.weather.p(time_range)
    RH = parameters.weather.RH(time_range)

    # Crear el DataFrame con todos los resultados
    df = pd.DataFrame({
        'date': dates, 
        'E': E, 
        'L': L, 
        'A': A, 
        'p': P, 
        'T': T, 
        'RH': RH
    })

    # Guardar el CSV con todos los resultados
    df.set_index('date', inplace=True)
    df.to_csv(output_path, index=True)


if __name__ == '__main__':
    # Directorio de salida
    output_dir = "sensitivity_results_contenedores_porcent"
    os.makedirs(output_dir, exist_ok=True)

    # Variaciones generales
    variations = [-0.5, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, -0.01, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
    

    # Parámetros fijos
    fixed_parameters = {'level_height': 0.1, 'surface': 50, 'initial_water': 0, 'manually_filled': 0.0, 'bare': 1, 'evaporation_factor': 1}


    # Base model (sin ninguna variación)
    print("Running base model (no variations)...")
    base_configuration = Configuration('example.cfg')
    base_output_path = os.path.join(output_dir, 'base_model.csv')
    run_and_save_model(base_configuration, base_output_path)


    # Itera sobre parámetros fijos y sus variaciones
    for param, base_value in fixed_parameters.items():
        for variation in variations:
            param_value = base_value * (1 + variation)
            update = {
                'breeding_site': {
                    param: param_value
                }
            }

            # Actualiza la configuración y genera un archivo temporal
            configuration = myConf(update)

            # Define el nombre de salida basado en el parámetro y la variación
            output_filename = f"{param}_var{int(variation * 100)}.csv"
            output_path = os.path.join(output_dir, output_filename)

            # Ejecuta el modelo y guarda los resultados
            print(f"Running model for {param} with variation {variation * 100}%...")
            run_and_save_model(configuration, output_path)

    print("Sensitivity analysis completed!")

