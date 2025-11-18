import pandas as pd

try:
    df_ree = pd.read_parquet('data/0_raw/ree_5min_bruto.parquet')
except FileNotFoundError:
    print("Error: No se encontraron los archivos Parquet procesados.")
    print("Asegúrate de haber ejecutado 'descargar_datos_ree.py' primero.")
    exit()

df_ree = df_ree.drop(columns=['sol', 'dif', 'expAnd', 'expMar', 'expPor', 'expFra', 'inter', 'gnhd', 'impFra', 'impPor', 'impMar', 'impAnd', 'factorEmisionCO2_eol', 'factorEmisionCO2_nuc', 'factorEmisionCO2_sol', 'factorEmisionCO2_hid', 'factorEmisionCO2_solTer', 'factorEmisionCO2_solFot', 'factorEmisionCO2_termRenov'])
df_ree.to_parquet('data/1_intermediate/ree_5min_limpio.parquet')
print(f'Datos guardados.')