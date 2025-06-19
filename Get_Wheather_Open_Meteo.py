####################################################
# Get_Wheather_Open_Meteo.py
# Duomenų gavimas iš Open-Meteo API
# Duomenų išsaugojimas į CSV failą - kad nereiktu kiekvieną kartą siųsti užklausos
####################################################


import requests
import pandas as pd
import os 

def get_open_meteo_historical_data(latitude, longitude, start_date, end_date, hourly_params=None):
    """
    Gauna istorinius orų duomenis iš Open-Meteo API.
    start_date ir end_date turi būti YYYY-MM-DD formato.
    hourly_params: norimų valandinių parametrų sąrašas arba kableliais atskirta eilutė.
    """
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    if hourly_params is None:
        hourly_params = "temperature_2m,relativehumidity_2m,snow_depth,precipitation"
    elif isinstance(hourly_params, list):
        hourly_params = ",".join(hourly_params)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly_params,
        "timezone": "auto" # "auto" nustatys laiko juostą pagal koordinates
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status() # Išmeta klaidą, jei HTTP statusas yra 4xx ar 5xx
        data = response.json()
        
        if 'hourly' in data and data['hourly']: # Patikriname ar 'hourly' egzistuoja ir nėra tuščias
            df = pd.DataFrame(data['hourly'])
            # Konvertuojame 'time' stulpelį į datetime objektus
            df['time'] = pd.to_datetime(df['time'])
            return df
        else:
            print("API negrąžino 'hourly' duomenų arba 'hourly' duomenys tušti.")
            if 'reason' in data: # Open-Meteo gali pateikti klaidos priežastį
                print(f"Priežastis: {data['reason']}")
            return None
            
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP klaida: {http_err}")
        # response objektas bus pasiekiamas, nes raise_for_status() jį turėjo
        print(f"Atsakymo turinys: {response.text}")
    except requests.exceptions.RequestException as req_err: # Kitos tinklo klaidos
        print(f"Užklausos klaida: {req_err}")
    except ValueError as json_err: # Klaida dekoduojant JSON
        print(f"Klaida apdorojant JSON: {json_err}")
        # Jei response objektas egzistuoja ir įvyko JSON klaida, galime bandyti spausdinti tekstą
        if 'response' in locals() and response:
             print(f"Gautas tekstas: {response.text}")
    return None

if __name__ == "__main__":
    # Canazei apytikslės koordinatės
    canazei_lat = 46.47
    canazei_lon = 11.77
    
    # Norimi gauti valandiniai parametrai
    # Daugiau parametrų rasite Open-Meteo dokumentacijoje:
    # https://open-meteo.com/en/docs/historical-weather-api
    desired_hourly_params = [
        "temperature_2m",
        "relativehumidity_2m",
        "dewpoint_2m",
        "apparent_temperature",
        "precipitation",
        "rain",
        "snowfall", # Naujai iškritęs sniegas per valandą
        "snow_depth", # Bendras sniego gylis ant žemės
        "weathercode",
        "pressure_msl",
        "surface_pressure",
        "cloudcover",
        "windspeed_10m",
        "winddirection_10m",
        "windgusts_10m"
    ]
    
    start_date_str = "2011-01-01" # Pakeistas pradžios datos formatas į YYYY-MM-DD
    end_date_str = "2020-12-31"   # Laikotarpis testavimui

    # Išvedame informaciją apie užklausą
    print(f"Gaunami duomenys vietovei su koordinatėmis ({canazei_lat}, {canazei_lon}) nuo {start_date_str} iki {end_date_str}...")
    
    weather_df = get_open_meteo_historical_data(
        canazei_lat,
        canazei_lon,
        start_date_str,
        end_date_str,
        hourly_params=desired_hourly_params
    )
    
    if weather_df is not None:
        print("\nGauti duomenys (pirmos 5 eilutės):")
        print(weather_df.head())
        print(f"\nIš viso gauta eilučių: {len(weather_df)}")
        
        # Pavyzdys, kaip išsaugoti į CSV
        output_dir = "open_meteo_data"
        # Sukuriame katalogą, jei jo nėra (exist_ok=True nekelia klaidos, jei katalogas jau yra)
        os.makedirs(output_dir, exist_ok=True)
        
        # Formuojame failo pavadinimą
        csv_filename = f"canazei_{start_date_str}_to_{end_date_str}.csv"
        output_path = os.path.join(output_dir, csv_filename) # Naudojame os.path.join keliui formuoti
        
        try:
            weather_df.to_csv(output_path, index=False)
            print(f"\nDuomenys sėkmingai išsaugoti į: {output_path}")
        except Exception as e:
            print(f"\nKlaida saugant duomenis į CSV: {e}")
    else:
        print("\nNepavyko gauti duomenų.")
