# Los datos de AEMET son diarios y por estación meteorológica. Necesitamos crear datos a nivel nacional para poder
# relacionarlos con los datos de la REE. Esta ponderación hay que hacerla en base a su relación con los datos de la REE.
# La temperatura predice la demanda. Debemos ponderar por población.
# El viento predice la energía eólica. Debemos ponderar por capacidad eólica instalada.
# Los datos de AEMET no nos dicen nada sobre el sol que hace en un día, así que usaremos la hora, el día y las
# precipitaciones totales (ponderadas en base a capacidad solar instalada)
# Estas ponderaciones las he recopilado de internet y las almaceno en data/3_external/ponderacion_provincias.csv
# datos población: https://www.ine.es/jaxiT3/Tabla.htm?t=2852
# datos energía:
#         solar: https://www.sistemaelectrico-ree.es/es/informe-de-energias-renovables/sol/potencia-instalada/solar-fotovoltaica-solpotencia
#        eólica: https://www.sistemaelectrico-ree.es/es/informe-de-energias-renovables/viento/potencia-instalada-viento

import pandas as pd
import numpy as np

try:
    df_aemet = pd.read_parquet('data/1_intermediate/aemet_diario_limpio.parquet')
    df_pesos = pd.read_csv('data/3_external/ponderacion_provincias.csv', header=0)
except FileNotFoundError:
    print("Error: No se encontraron los archivos de AEMET o de pesos.")
    print("Asegúrate de tener 'aemet_..._limpio.parquet' en 'data/1_intermediate/'")
    print("Y 'ponderacion_provincias.csv' en 'data/3_external/'")
    exit()

df_aemet['provincia'] = df_aemet['provincia'].str.upper() 
df_pesos['provincia'] = df_pesos['provincia'].str.upper() 
df_merged = pd.merge(df_aemet, df_pesos, on='provincia', how='left')
df_merged = df_merged.fillna(0)

df_merged['temp_min_ponderada_demanda'] = df_merged['temp_min'] * df_merged['peso_poblacion'] / df_merged['n_estaciones_provincia']
df_merged['temp_max_ponderada_demanda'] = df_merged['temp_max'] * df_merged['peso_poblacion'] / df_merged['n_estaciones_provincia']
df_merged['viento_ponderado_eolica'] = df_merged['viento_max'] * df_merged['peso_eolica'] / df_merged['n_estaciones_provincia']
df_merged['precipitacion_ponderada_solar'] = df_merged['precipitacion_total'] * df_merged['peso_solar'] / df_merged['n_estaciones_provincia']

df_agregado_diario = df_merged.groupby('fecha').agg(
    temp_max_nacional_ponderada=('temp_max_ponderada_demanda', 'sum'),
    temp_min_nacional_ponderada=('temp_min_ponderada_demanda', 'sum'),
    viento_nacional_ponderado=('viento_ponderado_eolica', 'sum'),
    precipitacion_nacional_ponderada=('precipitacion_ponderada_solar', 'sum')
)

archivo_salida = 'data/2_processed/aemet_agregado_diario.parquet'
df_agregado_diario.to_parquet(archivo_salida)

print(f"Features nacionales de AEMET guardadas en: {archivo_salida}")
print(df_agregado_diario.head())