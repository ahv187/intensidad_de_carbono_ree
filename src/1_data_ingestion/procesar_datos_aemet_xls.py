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

print(f'Buscando archivos Aemet%Y-%m-%d.xls en \'data/aemet\'')

archivos = glob.glob('data/0_raw/aemet_xls/**/Aemet*.xls', recursive=True)
if not archivos:
    print(f'No se encontró ningún archivo.')
else:
    print(f'Se encontraron {len(archivos)} archivos.')
    dfs = []
    for archivo in archivos:
        nombre_archivo = os.path.basename(archivo)
        print(f'Leyendo archivo: {nombre_archivo}...')
        try:
            fecha_str = nombre_archivo.split('Aemet')[1].split('.')[0]
            df_dia = pd.read_excel(archivo, header=4)
            df_dia['fecha'] = pd.to_datetime(fecha_str)

            dfs.append(df_dia)
        except Exception as e:
            print(f'Error al leer el archivo {nombre_archivo}: {e}')
    if not dfs:
        print('No se pudo leer ningún archivo.')
    else:
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

        df_aemet['Precipitacion_Total_Dia'] = df_aemet[
            ['Precipitación 00-06h (mm)', 'Precipitación 06-12h (mm)', 
            'Precipitación 12-18h (mm)', 'Precipitación 18-24h (mm)']
        ].sum(axis=1)

        df_aemet.to_parquet('data/1_intermediate/aemet_bruto.parquet')
        print(f'Datos guardados.')