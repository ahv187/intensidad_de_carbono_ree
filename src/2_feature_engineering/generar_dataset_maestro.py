import pandas as pd

try:
    df_ree_5min = pd.read_parquet('data/1_intermediate/ree_5min_limpio.parquet')
    df_aemet_diario = pd.read_parquet('data/2_processed/aemet_agregado_diario.parquet')
except FileNotFoundError:
    print("Error: No se encontraron los archivos Parquet procesados.")
    print("Asegúrate de haber ejecutado 'aggregate_aemet.py' primero.")
    exit()

df_aemet_diario['fecha_dia'] = df_aemet_diario.index.date
df_ree_5min['fecha_dia'] = df_ree_5min.index.date

df_maestro = pd.merge(
    df_ree_5min,
    df_aemet_diario,
    on='fecha_dia',
    how='left'
)

df_maestro = df_maestro.set_index(df_ree_5min.index)
df_maestro['hora'] = df_maestro.index.hour
df_maestro['minuto'] = df_maestro.index.minute
df_maestro['dia_semana'] = df_maestro.index.dayofweek
df_maestro['dia_año'] = df_maestro.index.dayofyear
df_maestro['mes'] = df_maestro.index.month


lag_1h = 12
lag_24h = 288 # 24 h / 5 min
lag_1s = lag_24h * 7

for col in ['dem', 'eol', 'solFot', 'solTer', 'cc', 'conb', 'turb']:
    print(f"Creando lags para: {col}")
    df_maestro[f'{col}_lag_1'] = df_maestro[col].shift(1)
    df_maestro[f'{col}_lag_3'] = df_maestro[col].shift(3)
    df_maestro[f'{col}_lag_1h'] = df_maestro[col].shift(lag_1h)
    df_maestro[f'{col}_lag_1d'] = df_maestro[col].shift(lag_24h)
    df_maestro[f'{col}_lag_1s'] = df_maestro[col].shift(lag_1s)
    df_maestro[f'{col}_media_1h'] = df_maestro[col].rolling(window=lag_1h).mean()
    df_maestro[f'{col}_std_1h'] = df_maestro[col].rolling(window=lag_1h).std()
    df_maestro[f'{col}_media_6h'] = df_maestro[col].rolling(window=lag_1h*6).mean()
    df_maestro[f'{col}_media_1d'] = df_maestro[col].rolling(window=lag_24h).mean()

    

for col in ['temp_max_nacional_ponderada', 'temp_min_nacional_ponderada', 'viento_nacional_ponderado', 'precipitacion_nacional_ponderada']:
    print(f"Creando lags para: {col}")
    df_maestro[f'{col}_lag_1d'] = df_maestro[col].shift(lag_24h)


df_maestro = df_maestro.drop(columns=['fecha_dia'])
df_maestro = df_maestro.dropna()

archivo_salida = 'data/2_processed/dataset_maestro_5min.parquet'
df_maestro.to_parquet(archivo_salida)