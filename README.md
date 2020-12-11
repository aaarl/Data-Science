# Data-Science

Link zur Google Doc: https://docs.google.com/document/d/1VHNHLRZtnDUGsXpSQcH1D46QQ5o75YK827jchYUHxlE/edit

## TODO
- Business Understandin schreiben
- Einleitung/CRISP-DM schreiben
- Spalte Beschreibung in der Datei /data/Preprocessing_beschreibung_Dataset.xlsx befüllen

## Dokumentation
Einleitung: 

 
Crisp-DM: 

 

Umsetzung: 

Business Understanding: 

 

Data Understanding:  

Das Dataset “Predict Successful Offers – Multivariant" wurde von der Quelle “squarkai.com/download-free-machine-learning-sample-data-sets/#toggle-id-2" (Stand 11.12.2020). Dieses Datenset enthält Datensätze für das überwachte Lernen, wobei die Spalte “Contracts” die Zielvariable darstellt. Wie der Name des Datensets andeutet handelt es sich um eine Klassifizierungsaufgabe. Diese Klassen sind “month to month”, “one year” und “two years”.  Da das Attribut “CustomerID” als Primärschlüssel dient und immer eindeutig ist, ist dieses Attribut für die weitere Datenanalyse irrelevant.  Alle anderen Attribute sind aktuell relevant, ob diese alle das Ergebnis beeinflussen kann noch nicht gesagt werden und wir unter anderem in den weiteren Phasen untersucht.  

In der Datei “Preprocessing_beschreibung_Dataset.xlsx” können die aktuellen Datentypen, Kategorisierungen, Anzahl der Ausprägungen sowie auch der Datentyp und Kategorisierung nach der Umwandlung der Merkmale eingesehen werden. Zudem wird angegeben, dass alle relevanten Attribute die mehr als zwei Ausprägungen besitzen nominalisiert werden, damit eine größere absolute Abweichung eines Wertes das Ergebnis nicht mehr beeinflusst als eine kleinere absolute Abweichung eines anderen Wertes, obwohl Prozentual betrachtet die Abweichung gleich ist. 

Probleme beim Data Understanding:  

Zu Beginn wurden die Datentypen der Spalten “Monthly Charges” und “Total Charges”  durch die Datei “NOCH EINFÜGEN” als String identifiziert. Da es eindeutig ist, dass diese Information falsch war, wurde hierbei auf Fehlersuche gegangen. Zudem kam, dass Werte des Dateityp String sich nur bedingt zur Visualisierung eignen. Der Fehler wurde schlussendlich in der Datei “data.csv” gefunden. Durch das Öffnen der Datei mit einer Drittsoftware wurde die Datei unbeabsichtigt konvertiert. Dadurch wurde zum Beispiel der Wert “19.8” in “19 Aug” konvertiert und somit wurde die Spalte korrekt als String interpretiert. Nach dem Ersetzen der Datei “data.csv” durch das unkonvertierte Dataset konnte die Visualisierung etc. korrekt erstellt werden.  

 

Data Visualisation: 

Wieso werden folgende Visualisierungen gewählt und wieso andere nicht. Z.B. Boxplots zu Attributen mit zwei Ausprägungen macht wenig SInn --> Torten-/Balkendiagramm 

Verteilungscharakteristiken ebenso nur bei Quantitativen Attributen mit vielen Ausprägungen  
