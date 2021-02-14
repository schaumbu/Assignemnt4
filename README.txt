Assignment 4 Projekt
3D Convolution

Arne Schaumburg (arne.schaumburg@ovgu.de)
Bennet Meier (bennet.meier@st.ovgu.de)


Wichtige Infos:

Im Elearning sind die Bilder aufgrund der Größe NICHT mit hochgeladen!
Das gesamte Projekt befindet sich in folgendem Github:

Der grundlegende Aufbau der Methoden und Klassen basiert auf Tutorial 8 und wurde für das Assignment erweitert und abgeändert.

Das Projekt läuft gerade nur mit Grafikkarten mit einer Architektur >= sm_53, da sonst bestimmte Funktionen für halfs nicht funktionieren.
Falls der PC nur eine Graka < sm_53 hat, müssen alle half-Funktionen auskommentiert werden und die Architektur in der cmakelist in Zeile 24 angepasst werden.

Wir konnten das Projekt am Ende nicht mehr auf den Servern der FIN laufen lassen, da die Bilder aus irgendeinem Grund nicht mehr richtig hochgeladen wurden.
Um es testen zu können, mussten wir alle anderen Ordner aus dem Projekt entfernen und nur einen Ordner mit Bildern (bspw. head) hochladen.

Bei Fragen oder Fehlern bitte per Mail kontaktieren.

Funktionsweise:

In der Main können die jeweiligen Kernel auskommentiert werden, wobei die kernel_supp angepasst werden muss:
5x5x5 Kernel -> kernel_supp = 5
3x3x3 Kernel -> kernel_supp = 3

Außerdem kann der Kernel auf unterschiedliche Bilder angewendet werden:
Dazu einfach eines der Bilder auskommentieren.

Zusätzlich kann unten angegeben werden, ob das Bild mit halfs oder floats berechnet werden soll.
Bei halfs erhält man erst bei sehr vielen Elementen einen Performanceunterschied.

In den Methoden floatConv und halfConv kann auskommentiert werden, ob der Kernel mit shared oder ohne shared memory laufen soll.

Außerdem empfehlen wir bei sehr großen Bildern, bspw. abdomen mit 361 Slides die Convolution auf der CPU auszukommentieren, da diese sehr lange dauert.

Die Ergebnisse werden dann in den Ordnern output_gpu und output_cpu gespeichert.


Benchmarks wurden mit 1000 Slides aus dem benchmark Ordner durchgeführt.
Die Ausführung hat durch das Einlesen der Bilder und das Laufen auf der CPU mehrere Stunden gedauert,
weshalb wir empfehlen, dies nicht zu wiederholen.
