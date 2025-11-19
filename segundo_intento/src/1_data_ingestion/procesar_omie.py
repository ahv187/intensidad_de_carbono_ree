import pandas as pd
import os
import glob
import sys

DIR_OMIE = 'data/0_raw/omie_horas/' 
RESULTADO = 'data/2_processed/omie_spot_price_1h.parquet'

csvs = glob.glob(os.path.join(DIR_OMIE, "*.csv"))

if not csvs:
    print("No se encontró ningún archivo.")
    sys.exit()

print(f"Encontrados {len(csvs)} archivos...")

df_list = []

for file in csvs:
    try:
        df = pd.read_csv(file, sep=';', engine='python')
        
        if 'datetime' in df.columns and 'value' in df.columns:
            temp_df = df[['datetime', 'value']].copy()
            
            temp_df = temp_df.rename(columns={'value': 'precio_omie'})
            df_list.append(temp_df)
        else:
            print(f"El archivo {os.path.basename(file)} no tiene columnas 'datetime' o 'value'")
            
    except Exception as e:
        print(f"Error leyendo {os.path.basename(file)}: {e}")

if not df_list:
    print("No se pudieron cargar datos válidos.")
    sys.exit()

df_final = pd.concat(df_list, ignore_index=True)
df_final['datetime'] = pd.to_datetime(df_final['datetime'], utc=True)
df_final = df_final.set_index('datetime')
df_final = df_final.sort_index()

df_final = df_final[~df_final.index.duplicated(keep='first')]

os.makedirs(os.path.dirname(RESULTADO), exist_ok=True)

df_final.to_parquet(RESULTADO)

print(f"Archivo guardado en: {RESULTADO}")
print("Primeras filas:")
print(df_final.head())