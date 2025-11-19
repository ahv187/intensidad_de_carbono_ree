import xarray as xr
import numpy as np
import pandas as pd
import os
import sys
import gc
import cfgrib

GRIBS = [
    'data/0_raw/cds_grib/2022.grib',
    'data/0_raw/cds_grib/2023.grib',
    'data/0_raw/cds_grib/2024.grib',
    'data/0_raw/cds_grib/2025.grib',
]
PONDERACIONES = 'data/3_external/ponderacion_provincias_lat_lon.csv'
RESULTADO = 'data/2_processed/features_cds.parquet'

if not all(os.path.exists(f) for f in GRIBS):
    print("Alguno de los archivos .grib no encontrado.")
    sys.exit()


datasets_ref = cfgrib.open_datasets(GRIBS[0])
ds_ref = datasets_ref[0] 
lats_grid = ds_ref.latitude.values
lons_grid = ds_ref.longitude.values
ds_ref.close()

df_pesos = pd.read_csv(PONDERACIONES)
df_pesos = df_pesos[df_pesos['provincia'].str.upper() != 'TOTAL']

lat_idxs = []
lon_idxs = []

for _, row in df_pesos.iterrows():
    lat_idx = np.abs(lats_grid - row['lat']).argmin()
    lon_idx = np.abs(lons_grid - row['lon']).argmin()
    lat_idxs.append(lat_idx)
    lon_idxs.append(lon_idx)

x_lat = xr.DataArray(lat_idxs, dims="provincia")
x_lon = xr.DataArray(lon_idxs, dims="provincia")

w_pob = df_pesos['peso_poblacion'].values
w_sol = df_pesos['peso_solar'].values
w_eol = df_pesos['peso_eolica'].values  

def deaccumulate_variable(series):
    diff = series.diff()
    diff.iloc[0] = series.iloc[0]
    
    mask_reset = diff < 0
    diff[mask_reset] = series[mask_reset]
    
    diff[diff < 0] = 0
    return diff

dfs_anuales = []

for archivo in GRIBS:
    print(f'Procesando {archivo}...')
    nombre_base = os.path.basename(archivo)

    try:
        datasets_list = cfgrib.open_datasets(archivo)
    except Exception:
        print(f"No se pudo abrir GRIB {nombre_base}")
        continue

    dfs_parts = []

    for i, ds_part in enumerate(datasets_list):
        vars_interest = ['t2m', 'u10', 'v10', 'tcc', 'tp', 'ssrd', 'cp']
        vars_found = [v for v in vars_interest if v in ds_part.data_vars]
        if not vars_found:
            continue

        try:
            ds_small = ds_part.isel(latitude=x_lat, longitude=x_lon)
            df_part = ds_small.to_dataframe().reset_index()
            col_time_real = None
            if 'valid_time' in df_part.columns:
                col_time_real = 'valid_time'
            elif 'time' in df_part.columns and 'step' in df_part.columns:
                # Si no hay valid_time explícito, lo calculamos
                df_part['calculated_time'] = df_part['time'] + df_part['step']
                col_time_real = 'calculated_time'
            elif 'time' in df_part.columns:
                col_time_real = 'time'

            if col_time_real and col_time_real != 'time':
                df_part = df_part.rename(columns={col_time_real: 'time_final'})
                cols_drop = ['time', 'step', 'valid_time']
                df_part = df_part.drop(columns=[c for c in cols_drop if c in df_part.columns], errors='ignore')
                df_part = df_part.rename(columns={'time_final': 'time'})
            
            cols_keep = ['time', 'provincia'] + vars_found
            cols_final = [c for c in cols_keep if c in df_part.columns]
            df_part = df_part[cols_final].copy()

            df_part = df_part[cols_final].copy()
            df_part = df_part.set_index(['time', 'provincia'])
            df_part = df_part.sort_index()
            
            df_part = df_part[~df_part.index.duplicated(keep='first')]
            dfs_parts.append(df_part)
            
        except Exception as e:
            print(f"Error procesando grupo {i}: {e}")
    
    if not dfs_parts:
        print(f"No se obtuvieron datos de {nombre_base}.")
        continue

    df_merged = pd.concat(dfs_parts, axis=1)
    df_merged.index.names = ['time', 'provincia']
    df_merged = df_merged.reset_index()

    year_file = int(''.join(filter(str.isdigit, nombre_base)))
    start_date = pd.Timestamp(f"{year_file}-01-01")
    df_merged = df_merged[df_merged['time'] >= start_date]

    accum_vars = ['ssrd', 'tp', 'cp']

    df_merged = df_merged.sort_values(by=['provincia', 'time'])

    for col in accum_vars:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0)
            df_merged[col] = df_merged.groupby('provincia')[col].transform(deaccumulate_variable)

    if 't2m' in df_merged: df_merged['t2m'] = df_merged['t2m'] - 273.15 # k -> c
    if 'u10' in df_merged and 'v10' in df_merged: 
        df_merged['viento_ms'] = np.sqrt(df_merged['u10']**2 + df_merged['v10']**2) # vector -> magnitud
    if 'ssrd' in df_merged: df_merged['ssrd'] = df_merged['ssrd'] / 3600.0 # j/m2 -> w / m2
    if 'tp' in df_merged: df_merged['tp'] = df_merged['tp'] * 1000.0 # m -> mm
    if 'tcc' in df_merged: df_merged['tcc'] = df_merged['tcc'] * 100.0 # 0-1 -> 0-100
    if 'cp' in df_merged: df_merged['cp'] = df_merged['cp'] * 1000.0  # m -> mm

    df_merged['provincia'] = df_merged['provincia'].astype(int)
    
    df_merged['w_pob'] = df_merged['provincia'].map(lambda x: w_pob[x])
    df_merged['w_sol'] = df_merged['provincia'].map(lambda x: w_sol[x])
    df_merged['w_eol'] = df_merged['provincia'].map(lambda x: w_eol[x])

    cols_agg = {}
    
    if 't2m' in df_merged: 
        df_merged['w_t2m'] = df_merged['t2m'] * df_merged['w_pob']
        cols_agg['w_t2m'] = 'sum'
    if 'tp' in df_merged: 
        df_merged['w_tp'] = df_merged['tp'] * df_merged['w_pob']
        cols_agg['w_tp'] = 'sum'
    if 'cp' in df_merged: 
        df_merged['w_cp'] = df_merged['cp'] * df_merged['w_pob']
        cols_agg['w_cp'] = 'sum'
    if 'tcc' in df_merged: 
        df_merged['w_tcc'] = df_merged['tcc'] * df_merged['w_pob']
        cols_agg['w_tcc'] = 'sum'
    if 'viento_ms' in df_merged: 
        df_merged['w_viento'] = df_merged['viento_ms'] * df_merged['w_eol']
        cols_agg['w_viento'] = 'sum'
    if 'ssrd' in df_merged: 
        df_merged['w_solar'] = df_merged['ssrd'] * df_merged['w_sol']
        cols_agg['w_solar'] = 'sum'
        

    df_national = df_merged.groupby('time').agg(cols_agg)
    
    rename_dict = {
        'w_t2m': 'temp_celsius', 'w_tp': 'precipitacion_mm', 'w_tcc': 'nubes_total_percent',
        'w_viento': 'viento_magnitud_ms', 'w_solar': 'radiacion_solar_W_m2', 'w_cp': 'precipitacion_convectiva_mm'
    }
    df_national = df_national.rename(columns=rename_dict)
    
    dfs_anuales.append(df_national)
    
    del datasets_list, dfs_parts, df_merged
    gc.collect()

if not dfs_anuales:
    print("No se encontraron datos.")
    sys.exit()

df_final = pd.concat(dfs_anuales).sort_index()
df_final = df_final[~df_final.index.duplicated(keep='first')]

cols_pos = ['precipitacion_mm', 'precipitacion_convectiva_mm', 'radiacion_solar_W_m2']
for col in cols_pos:
    if col in df_final.columns:
        df_final[df_final[col] < 0] = 0

print(f"Escribiendo resultado en: {RESULTADO}")
df_final.to_parquet(RESULTADO)
    