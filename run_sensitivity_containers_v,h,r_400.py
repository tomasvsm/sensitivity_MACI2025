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
    PUPAE = parameters.PUPAE
    ADULT1 = parameters.ADULT1
    ADULT2 = parameters.ADULT2
    OVIPOSITION = parameters.OVIPOSITION
    BS_a = parameters.BS_a

    E = np.sum(results[:, EGG], axis=1) / BS_a
    L = np.sum(results[:, LARVAE], axis=1) / BS_a
    PU = np.sum(results[:, PUPAE], axis=1) / BS_a
    A = (results[:, ADULT1] + results[:, ADULT2]) / BS_a

    T = parameters.weather.T(time_range) - 273.15  # Convertir a Celsius
    P = parameters.weather.p(time_range)
    RH = parameters.weather.RH(time_range)

    # Crear el DataFrame con todos los resultados
    df = pd.DataFrame({
        'date': dates, 
        'E': E, 
        'L': L,
        'PU': PU, 
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
    output_dir = "sensitivity_results_contenedores"
    os.makedirs(output_dir, exist_ok=True)

    # Definir los volúmenes, radios y alturas específicos para los diferentes contenedores
    container_specs = [
        {"name": "Tapa_Gaseosa", "height_range": np.linspace(1, 3, 10), "radius_range": np.linspace(3, 5, 10)},
        {"name": "Neumatico", "height_range": np.linspace(10, 20, 10), "radius_range": np.linspace(25, 40, 10)},
        {"name": "Florero", "height_range": np.linspace(20, 50, 10), "radius_range": np.linspace(5, 15, 10)},
        {"name": "Bebedero_Animales", "height_range": np.linspace(10, 30, 10), "radius_range": np.linspace(10, 20, 10)}
    ]

    # Generar combinaciones de volumen, radio y altura lógicas
    combinations = []
    for spec in container_specs:
        for height in spec["height_range"]:
            for radius in spec["radius_range"]:
                volume = np.pi * radius ** 2 * height  # Calcular el volumen
                volume = round(volume, 2)  # Redondear el volumen a dos decimales
                height = round(height, 2)
                radius = round(radius, 2)
                combinations.append((volume, height, radius, spec["name"]))

    # Verificar cuántas combinaciones tenemos
    num_combinations = len(combinations)
    print(f"Total possible combinations: {num_combinations}")

    # Ejecutar simulaciones para todas las combinaciones disponibles
    for volume, height, radius, container_name in combinations:
        update = {
            'breeding_site': {
                'height': height,
                'radius': radius,
            }
        }

        # Actualiza la configuración y genera un archivo temporal
        configuration = myConf(update)

        # Define el nombre de salida basado en el volumen y las variaciones
        output_filename = f"{container_name}_volume_{int(volume)}_h{height:.2f}_r{radius:.2f}.csv"
        output_path = os.path.join(output_dir, output_filename)

        # Ejecuta el modelo y guarda los resultados
        print(f"Running model for {container_name} with volume={volume} cm³, height={height:.2f} cm, radius={radius:.2f} cm...")
        run_and_save_model(configuration, output_path)

    print("Sensitivity analysis completed!")

