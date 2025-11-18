import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

MODEL_DIR = 'models'
modelos = {}

try:
    modelos['dem'] = xgb.XGBRegressor()
    modelos['dem'].load_model(os.path.join(MODEL_DIR, 'modelo_demanda.json'))
    modelos['eol'] = xgb.XGBRegressor()
    modelos['eol'].load_model(os.path.join(MODEL_DIR, 'modelo_eol.json'))
    modelos['sol'] = xgb.XGBRegressor()
    modelos['sol'].load_model(os.path.join(MODEL_DIR, 'modelo_sol.json'))
    modelos['cc'] = xgb.XGBRegressor()
    modelos['cc'].load_model(os.path.join(MODEL_DIR, 'modelo_cc.json'))
    modelos['hid'] = xgb.XGBRegressor()
    modelos['hid'].load_model(os.path.join(MODEL_DIR, 'modelo_hidraulica.json'))
    modelos['nuc'] = xgb.XGBRegressor()
    modelos['nuc'].load_model(os.path.join(MODEL_DIR, 'modelo_nuc.json'))
    modelos['otras'] = xgb.XGBRegressor()
    modelos['otras'].load_model(os.path.join(MODEL_DIR, 'modelo_otras.json'))
    
except Exception as e:
    print(f"No se pudo cargar un modelo. {e}")
    sys.exit()

DATA_PATH = 'data/2_processed/dataset_maestro_5min.parquet'
df_5min = pd.read_parquet(DATA_PATH)
df_5min = df_5min.sort_index()

df_5min['sol_total'] = df_5min['solFot'] + df_5min['solTer']
df_5min['otras_fosiles_total'] = df_5min['cogenResto'] + df_5min['die'] + \
                                 df_5min['gas'] + df_5min['vap'] + \
                                 df_5min['genAux'] + df_5min['termRenov']

agg_dict = {
    'dem': 'sum', 
    'eol': 'sum', 
    'sol_total': 'sum', 
    'cc': 'sum',
    'car': 'sum', 
    'hid': 'sum', 
    'nuc': 'sum', 
    'otras_fosiles_total': 'sum',
    'temp_max_nacional_ponderada': 'first', 
    'temp_min_nacional_ponderada': 'first',
    'viento_nacional_ponderado': 'first',
    'precipitacion_nacional_ponderada': 'first',
    'dia_año_sin': 'first', 
    'dia_año_cos': 'first',
    'dia_semana_sin': 'first', 
    'dia_semana_cos': 'first',
    'factorEmisionCO2_cc': 'mean', 
    'factorEmisionCO2_car': 'mean',
    'factorEmisionCO2_cogenResto': 'mean',
    'factorEmisionCO2_die': 'mean',
    'factorEmisionCO2_gas': 'mean',
    'factorEmisionCO2_vap': 'mean'
}
df_1d = df_5min.resample('D').agg(agg_dict).rename(columns={
    'dem': 'dem_total_dia', 'eol': 'eol_total_dia', 'sol_total': 'sol_total_dia',
    'cc': 'cc_total_dia', 'car': 'car_total_dia', 'hid': 'hid_total_dia',
    'nuc': 'nuc_total_dia', 'otras_fosiles_total': 'otras_fosiles_total_dia'
})

for col in ['dem', 'eol', 'sol', 'cc', 'car', 'hid', 'nuc']:
    df_1d[f'{col}_total_lag_7d'] = df_1d[f'{col}_total_dia'].shift(7)
    df_1d[f'{col}_total_lag_14d'] = df_1d[f'{col}_total_dia'].shift(14)
    df_1d[f'{col}_total_mean_28d'] = df_1d[f'{col}_total_dia'].shift(7).rolling(window=28).mean()
    df_1d[f'{col}_total_std_28d'] = df_1d[f'{col}_total_dia'].shift(7).rolling(window=28).std()

df_1d['otras_fosiles_lag_7d'] = df_1d['otras_fosiles_total_dia'].shift(7)
df_1d['otras_fosiles_lag_14d'] = df_1d['otras_fosiles_total_dia'].shift(14)
df_1d['otras_fosiles_mean_28d'] = df_1d['otras_fosiles_total_dia'].shift(7).rolling(window=28).mean()
df_1d['otras_fosiles_std_28d'] = df_1d['otras_fosiles_total_dia'].shift(7).rolling(window=28).std()
df_1d['sol_total_lag_364d'] = df_1d['sol_total_dia'].shift(364)

df_1d = df_1d.dropna()

FEATURES_DEM = ['temp_max_nacional_ponderada', 'temp_min_nacional_ponderada', 'dia_año_sin', 'dia_año_cos', 'dia_semana_sin', 'dia_semana_cos', 'dem_total_lag_7d', 'dem_total_lag_14d', 'dem_total_mean_28d']
FEATURES_EOL = ['viento_nacional_ponderado', 'dia_año_sin', 'dia_año_cos', 'eol_total_lag_7d', 'eol_total_lag_14d', 'eol_total_mean_28d']
FEATURES_SOL = ['dia_año_sin', 'dia_año_cos', 'precipitacion_nacional_ponderada', 'sol_total_lag_7d', 'sol_total_lag_14d', 'sol_total_mean_28d', 'sol_total_std_28d', 'sol_total_lag_364d']
FEATURES_CC = ['temp_max_nacional_ponderada','temp_min_nacional_ponderada','viento_nacional_ponderado','precipitacion_nacional_ponderada', 'dia_año_sin', 'dia_año_cos', 'dia_semana_sin', 'dia_semana_cos', 'cc_total_lag_7d', 'cc_total_lag_14d', 'cc_total_mean_28d']
FEATURES_HID = ['dia_año_sin', 'dia_año_cos', 'dia_semana_sin', 'dia_semana_cos', 'temp_max_nacional_ponderada','temp_min_nacional_ponderada','viento_nacional_ponderado','precipitacion_nacional_ponderada', 'hid_total_lag_7d', 'hid_total_lag_14d', 'hid_total_mean_28d', 'hid_total_std_28d']
FEATURES_NUC = ['nuc_total_lag_7d', 'nuc_total_lag_14d', 'nuc_total_mean_28d', 'temp_max_nacional_ponderada','temp_min_nacional_ponderada','viento_nacional_ponderado','precipitacion_nacional_ponderada', 'dia_año_sin', 'dia_año_cos', 'dia_semana_sin', 'dia_semana_cos']
FEATURES_OTRAS = ['dia_año_sin', 'dia_año_cos', 'dia_semana_sin', 'dia_semana_cos', 'temp_max_nacional_ponderada','temp_min_nacional_ponderada','viento_nacional_ponderado','precipitacion_nacional_ponderada', 'otras_fosiles_lag_7d', 'otras_fosiles_lag_14d', 'otras_fosiles_mean_28d', 'otras_fosiles_std_28d']

TEST_START_DATE = '2024-10-01'
test_data = df_1d.loc[df_1d.index >= TEST_START_DATE].copy()
results_list = []

for fecha in test_data.index:
    input_data = test_data.loc[fecha:fecha]
    predicciones = {}
    predicciones['dem'] = modelos['dem'].predict(input_data[FEATURES_DEM])[0]
    predicciones['eol'] = modelos['eol'].predict(input_data[FEATURES_EOL])[0]
    predicciones['sol'] = modelos['sol'].predict(input_data[FEATURES_SOL])[0]
    predicciones['cc'] = modelos['cc'].predict(input_data[FEATURES_CC])[0]
    predicciones['car'] = 0 # despreciamos el carbón ya que es imposible/muy difícil de modelar ... es impredecible
    predicciones['hid'] = modelos['hid'].predict(input_data[FEATURES_HID])[0]
    predicciones['nuc'] = modelos['nuc'].predict(input_data[FEATURES_NUC])[0]
    predicciones['otras'] = modelos['otras'].predict(input_data[FEATURES_OTRAS])[0]
    
    # "estrujamos" los resultados ya que los modelos no saben que deben "competir" entre sí
    pred_dem_residual = predicciones['dem'] - \
                        predicciones['eol'] - \
                        predicciones['sol'] - \
                        predicciones['nuc']
    
    suma_preds_gestionables = predicciones['cc'] + \
                              predicciones['car'] + \
                              predicciones['hid'] + \
                              predicciones['otras']

    if suma_preds_gestionables > 0:
        factor_correccion = abs(pred_dem_residual) / suma_preds_gestionables
    else:
        factor_correccion = 1.0

    predicciones['cc'] = predicciones['cc'] * factor_correccion
    predicciones['hid'] = predicciones['hid'] * factor_correccion
    predicciones['otras'] = predicciones['otras'] * factor_correccion
    
    emisiones_tCO2 = (
        predicciones['cc'] * input_data['factorEmisionCO2_cc'] +
        predicciones['otras'] * input_data['factorEmisionCO2_cogenResto']
    )
    intensidad_g_kWh = (emisiones_tCO2 / predicciones['dem']) * 1000

    real_dem = input_data['dem_total_dia'].iloc[0]
    real_cc = input_data['cc_total_dia'].iloc[0]
    real_car = input_data['car_total_dia'].iloc[0]
    real_otras = input_data['otras_fosiles_total_dia'].iloc[0]

    emisiones_reales_tCO2 = (
        real_cc * input_data['factorEmisionCO2_cc'] +
        real_car * input_data['factorEmisionCO2_car'] + 
        real_otras * input_data['factorEmisionCO2_cogenResto']
    )
    intensidad_real_g_kWh = (emisiones_reales_tCO2 / real_dem) * 1000
    
    results_list.append({
        'fecha': fecha,
        'pred_intensidad': intensidad_g_kWh,
        'real_intensidad': intensidad_real_g_kWh
    })

df_results = pd.DataFrame(results_list).set_index('fecha')

r2_global = r2_score(df_results['real_intensidad'], df_results['pred_intensidad'])
mae_global = mean_absolute_error(df_results['real_intensidad'], df_results['pred_intensidad'])

print(f"R²: {r2_global:.4f}")
print(f"MAE: {mae_global:.2f} gCO2/kWh")

plt.figure(figsize=(15, 7))
plt.plot(df_results.index, df_results['real_intensidad'], label='Intensidad real', 
         marker='.', linestyle='None', alpha=0.7)
plt.plot(df_results.index, df_results['pred_intensidad'], 
         label=f'Predicción (R²: {r2_global:.2f})', 
         linestyle='--', color='red')
plt.title('Intensidad de CO2')
plt.ylabel('Intensidad (gCO2/kWh)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()