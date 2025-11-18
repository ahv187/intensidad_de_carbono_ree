# Procesar los archivos obtenidos de https://datosclima.es/Aemet2013/DescargaDatos.html

import pandas as pd
import glob
import os
import re

def limpiar_valor(valor):
    if pd.isna(valor) or valor == "--":
        return None
    
    valor_str = str(valor)
    
    # buscar el primer número en el string
    match = re.search(r'(\d+\.?\d*)', valor_str)
    
    if match:
        return float(match.group(1))
    else:
        return None

print(f'Buscando archivos Aemet%Y-%m-%d.xls en \'data/0_raw/aemet_xls/\'')

archivos = glob.glob('data/0_raw/aemet_xls/**/Aemet*.xls', recursive=True)
if not archivos:
    print(f'No se encontró ningún archivo.')
    exit()

print(f'Se encontraron {len(archivos)} archivos.')

dfs = []
for archivo in archivos:
    nombre_archivo = os.path.basename(archivo)
    print(f'Leyendo archivo: {nombre_archivo}...')
    try:
        fecha_str = nombre_archivo.split('Aemet')[1].split('.')[0]
        df_dia = pd.read_excel(archivo, header=4)
        df_dia['fecha'] = pd.to_datetime(fecha_str)
        
        df_dia = df_dia[df_dia['Provincia'] != 'Ceuta']
        df_dia = df_dia[df_dia['Provincia'] != 'Melilla']

        dfs.append(df_dia)
    except Exception as e:
        print(f'Error al leer el archivo {nombre_archivo}: {e}')
if not dfs:
    print('No se pudo leer ningún archivo.')
    exit()


df_aemet = pd.concat(dfs, ignore_index=True)

columnas_a_limpiar = [
    'Temperatura máxima (ºC)', 
    'Temperatura mínima (ºC)', 
    'Racha (km/h)', 
    'Velocidad máxima (km/h)',
    'Precipitación 00-06h (mm)',
    'Precipitación 06-12h (mm)',
    'Precipitación 12-18h (mm)',
    'Precipitación 18-24h (mm)'
]

for col in columnas_a_limpiar:
    if col in df_aemet.columns:
        df_aemet[col] = df_aemet[col].apply(limpiar_valor)
    else:
        print(f"Advertencia: Columna '{col}' no encontrada.")


df_aemet = df_aemet.drop(columns=['Estación', 'Precipitación 00-06h (mm)', 'Precipitación 06-12h (mm)', 'Precipitación 12-18h (mm)', 'Precipitación 18-24h (mm)', 'Temperatura media (ºC)', 'Racha (km/h)'])
df_aemet = df_aemet.rename(columns={
    'Provincia': 'provincia',
    'Temperatura máxima (ºC)': 'temp_max',
    'Temperatura mínima (ºC)': 'temp_min',
    'Velocidad máxima (km/h)': 'viento_max',
    'Precipitación 00-24h (mm)': 'precipitacion_total'
})

df_aemet['provincia'] = df_aemet['provincia'].str.upper().str.strip()
conteo_provincia = df_aemet.groupby(['fecha', 'provincia'])['provincia'].transform('size')
df_aemet['n_estaciones_provincia'] = conteo_provincia
df_aemet.to_parquet('data/1_intermediate/aemet_diario_limpio.parquet')
print(f'Datos guardados.')