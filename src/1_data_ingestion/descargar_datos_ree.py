import pandas as pd
import requests
from datetime import timedelta, date
import time
import json


def descargar_datos_ree(fecha):
    fecha_str = fecha.strftime('%Y-%m-%d')
    url_api_demanda_generacion = "https://demanda.ree.es/WSvisionaMovilesPeninsulaRest/resources/demandaGeneracionPeninsula"
    url_api_co2 = "https://demanda.ree.es/WSvisionaMovilesPeninsulaRest/resources/coeficientesCO2"
    querystr = {'curva': 'NACIONALAU', 'fecha': fecha_str}
    try:
        respuesta_demanda = requests.get(url_api_demanda_generacion, params=querystr)
        respuesta_demanda.raise_for_status()
        demanda_json = json.loads(respuesta_demanda.text[5:-2])

        if 'valoresHorariosGeneracion' not in demanda_json:
            print(f'La estructura del JSON ha cambiado')
            return None

        respuesta_co2 = requests.get(url_api_co2, params=querystr)
        respuesta_co2.raise_for_status()
        co2_json = json.loads(respuesta_co2.text[5:-2])
  
        df_demanda = pd.DataFrame(demanda_json['valoresHorariosGeneracion'])
        df_demanda['ts'] = pd.to_datetime(df_demanda['ts'])
        df_demanda = df_demanda.set_index('ts').sort_index()

        if df_demanda.empty:
            print(f'No hay datos para {fecha_str}')

        df_co2 = pd.DataFrame(co2_json)
        df_co2['ts'] = pd.to_datetime(df_co2['ts'])
        df_co2 = df_co2.set_index('ts').sort_index()

        return df_demanda.join(df_co2)
    
    except Exception as e:
        print(f'Error descargando datos para {fecha}: {e}')
        return None
    
fecha_ini = date(2020, 1, 1)
fecha_fin = date(2025, 10, 31)
delta = timedelta(days=1)

datos_ree = []

fecha = fecha_ini
while fecha <= fecha_fin:
    print(f'Procesando: {fecha}')
    df_dia = descargar_datos_ree(fecha)
    if df_dia is not None:
        datos_ree.append(df_dia)

    time.sleep(0.2)
    fecha += delta

df_total = pd.concat(datos_ree)
df_total.to_parquet('data/1_intermediate/ree.parquet')
print(f'Datos guardados.')