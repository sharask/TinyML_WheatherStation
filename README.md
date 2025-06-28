# TinyML Orų Stotelė Sniego Prognozei

Šis projektas yra TinyML (mašininio mokymosi įterptinėse sistemose) pagrindu sukurta orų stotelė, kuri prognozuoja sniego tikimybę remdamasi temperatūros ir drėgmės duomenimis. Sistema apima visą procesą: nuo istorinių orų duomenų surinkimo, modelio apmokymo ir kvantavimo iki jo paleidimo ant skirtingų platformų: asmeninio kompiuterio (PC), Raspberry Pi ir Arduino Nano 33 BLE Sense.

## Projekto Struktūra ir Veikimas

Projektas susideda iš kelių pagrindinių etapų ir scenarijų:

### 1. Duomenų Surinkimas

*   **`Get_Wheather_Open_Meteo.py`**: scenarijus, kuris atsisiunčia istorinius orų duomenis (temperatūra, drėgmė, sniegas) iš Open-Meteo API ir išsaugo juos `.csv` faile. Tai leidžia išvengti pasikartojančių API užklausų.

### 2. Modelio Paruošimas

*   **`prepare_model.py`**: nuskaito duomenis iš `.csv` failo, juos apdoroja, subalansuoja ir apmoko TensorFlow/Keras modelį. Paruoštas modelis yra kvantuojamas (optimizuojamas dydžiui) ir konvertuojamas į du formatus:
    *   `.tflite`: skirtas naudoti su TensorFlow Lite interpretatoriumi (pvz., ant PC ar Raspberry Pi).
    *   `.h` (antraštės failas): skirtas tiesioginiam naudojimui mikrovaldiklių projektuose (pvz., Arduino) su C/C++ kalba.

### 3. Modelio Paleidimas ir Testavimas

*   **Ant PC:**
    *   `test_snow_model_PC.py`: paprastas scenarijus, leidžiantis testuoti `.tflite` modelį su rankiniu būdu įvestais duomenimis tiesiog kompiuteryje.
*   **Ant Raspberry Pi (Autonominės stotelės simuliacija):**
    *   `autonomous_station.py`: imituoja autonominės stotelės veikimą. Kas nustatytą laiko intervalą (pvz., 30 sekundžių) "nuskaito" jutiklių duomenis, įvykdo sniego prognozę su `.tflite` modeliu ir išsiunčia rezultatą į serverį.
    *   `server.py`: paprastas Flask serveris, kuris veikia ant PC ir priima duomenis, siunčiamus iš Raspberry Pi.
    *   `test_snow_model.py`: scenarijus, skirtas greitam modelio testavimui ant Raspberry Pi naudojant `tflite_runtime` biblioteką.
*   **Ant Arduino:**
    *   `Arduino/TinyML_WST`: projektas, paruoštas naudojimui su **VSCode ir PlatformIO**. Jis naudoja modernią `Chirale_TensorFlowLite` biblioteką ir `.h` modelio failą sniego prognozei atlikti ant **Arduino Nano 33 BLE Sense Rev2** plokštės.
    *   `Arduino/Inference_test`: senesnė projekto versija, paruošta su **Arduino IDE**. Naudoja pasenusią `Arduino_TensorFlowLite` biblioteką.

## Paleidimo Instrukcijos

### Būtina programinė įranga

*   Python 3.8+
*   VSCode su PlatformIO plėtiniu (Arduino daliai)
*   Arduino IDE (senesnei Arduino versijai)

### 1. Projekto paruošimas

```bash
# 1. Klonuokite repozitoriją
git clone https://github.com/tavovartotojas/tavorepozitorija.git
cd tavorepozitorija

# 2. Sukurkite virtualią aplinką ir ją aktyvuokite
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 3. Įdiekite reikalingas Python bibliotekas
pip install -r requirements.txt
```

### 2. Duomenų surinkimas

```bash
# Paleiskite scenarijų, kad gautumėte duomenis
python Get_Wheather_Open_Meteo.py
```
Duomenys bus išsaugoti `open_meteo_data/` aplanke.

### 3. Modelio apmokymas

```bash
# Paleiskite modelio paruošimo scenarijų
python prepare_model.py
```
Šis scenarijus sukurs `snow_forecast_model.tflite` ir `snow_forecast_model.h` failus.

### 4. Modelio vykdymas

*   **Ant PC:**
    ```bash
    python test_snow_model_PC.py
    ```
*   **Ant Raspberry Pi:**
    1.  Nukopijuokite `autonomous_station.py` ir `snow_forecast_model.tflite` į savo Raspberry Pi.
    2.  Savo PC paleiskite serverį:
        ```bash
        python server.py
        ```
    3.  Raspberry Pi faile `autonomous_station.py` pakeiskite `SERVER_URL` į savo PC IP adresą.
    4.  Raspberry Pi paleiskite stotelės scenarijų:
        ```bash
        python autonomous_station.py
        ```
*   **Ant Arduino (su PlatformIO):**
    1.  Atidarykite `Arduino/TinyML_WST` aplanką su VSCode.
    2.  Įsitikinkite, kad `src/snow_forecast_model.h` yra naujausia versija, sugeneruota `prepare_model.py`.
    3.  Įkelkite programą į Arduino Nano 33 BLE plokštę naudodami PlatformIO "Upload" mygtuką.
    4.  Stebėkite rezultatus per "Serial Monitor".

## Autonominės stotelės paleidimas kaip serviso (Raspberry Pi)

Kad `autonomous_station.py` veiktų nuolat ir pasileistų automatiškai po perkrovimo, galite sukurti `systemd` servisą.

1.  **Sukurkite serviso failą:**
    ```bash
    sudo nano /etc/systemd/system/weather_station.service
    ```
2.  **Įklijuokite šį turinį** (pakeiskite `User` ir kelius, jei reikia):
    ```ini
    [Unit]
    Description=Weather Station Service
    After=network.target

    [Service]
    ExecStart=/bin/bash -c "source /home/sarunas/TinyML_WheatherStation/venv/bin/activate && python3 /home/sarunas/TinyML_WheatherStation/autonomous_station.py"
    WorkingDirectory=/home/sarunas/TinyML_WheatherStation
    StandardOutput=journal
    StandardError=journal
    Restart=always
    User=sarunas

    [Install]
    WantedBy=multi-user.target
    ```
3.  **Serviso valdymas:**
    ```bash
    # Įjungti, kad pasileistų automatiškai
    sudo systemctl enable weather_station.service
    # Paleisti servisą
    sudo systemctl start weather_station.service
    # Patikrinti būseną
    sudo systemctl status weather_station.service
    # Sustabdyti
    sudo systemctl stop weather_station.service
    ```

## Technologijos

*   **Programavimas:** Python, C++ (Arduino)
*   **Mašininis mokymasis:** TensorFlow, Keras, Scikit-learn
*   **Duomenų apdorojimas:** Pandas, NumPy
*   **Vizualizacija:** Matplotlib, Seaborn
*   **Web Serveris:** Flask
*   **Įterptinės sistemos:** Arduino, PlatformIO