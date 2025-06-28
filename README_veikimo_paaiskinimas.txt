Veikimo aprašymas:

1. Get_Wheather_Open_Meteo.py - nuskaitome duomenis is Open Meteo ir issaugome scv, kad nereiktu daug kartu kreiptis dėl duomenu

2. prepare_model.py - nuoskaito duomenis iš scv ir paruošia Tensorflow modelį. Modelis konvertuojamas į .h (header) failą.
	- tensorflow modelį galima leisti ant PC, RPi (geriau naudoti interpretatorių)
	- .h skirtas Arduino ar kitiems MV su C kalba

3. Modelio paleidimas:

*test_snow_model.py - modelio testavimas ant PC

*test_snow_model.py - skirtas testavimui ant RPI (paleidžia modelį su nustatytais parametrais)

*autonomous_station.py - imituoja autonominės stotelės veikimą. Leidžiamas ant RPI, kas 30s "nuskaito" duomenis ir siunčia į serverį (PC)

*server.py - leidžiamas ant PC, "klauso" prognozės iš RPI.

Kaip paleisti automatiškai ant RPI:

- sukuriamas servisas
sudo nano /etc/systemd/system/weather_station.service

>>>>
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
>>>>

- "Enable" servisas
sudo systemctl enable weather_station.service
- "Start" servisas
sudo systemctl start weather_station.service
- Patikrinti ar veikia
sudo systemctl status weather_station.service
- Restart
sudo systemctl restart weather_station.service
- Stop
sudo systemctl stop weather_station.service
- Disable (jei nenorite, kad veiktų automatiškai po reboot)
sudo systemctl disable weather_station.service

/Arduino direktorija:
	/Inference_test - modelio paleidimas ant Arduino Nano 33 BLE, projektas parengtas su Arduino IDE, tačiau naudoja seną TensorFlowLite biblioteką
	/TinyML_WST - projektas parengtas su VSCode+Platformio, Arduino Nano 33 BLE, naudojamas atnaujinta Chirale TensorFlowLite biblioteka
