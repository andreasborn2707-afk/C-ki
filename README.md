🚀 Wichtige Voraussetzungen

• ⚠️ AMD-Grafikkarte: Stellen Sie unbedingt sicher, dass Sie eine AMD-Grafikkarte besitzen, bevor Sie den Code ausführen.
• 📦 Bibliotheken & OpenCL: Vergewissern Sie sich, dass die korrekten Bibliotheken sowie OpenCL auf Ihrem System installiert und eingerichtet sind, damit die GPU-Beschleunigung einwandfrei funktioniert.

⚙️ Konfiguration (Modellgröße)

• Modellgröße (modelsize): Können Sie nach Belieben anpassen.
• Input- & Output-Größe: Es wird dringend empfohlen, die Werte für die input size und output size nicht zu verändern.

🛠️ Schritt-für-Schritt-Anleitung


1. Modell initialisieren (Optional)

Wenn Sie nicht die von mir trainierte Version nutzen möchten, müssen Sie zuerst die Gewichte und Biases initialisieren:
• Führen Sie die Datei CreateModel.cpp aus.
• Dadurch wird die benötigte bias.json-Datei generiert.

2. Modell trainieren

Führen Sie anschließend die Datei TrainModel.cpp aus. Sie können die Lernrate (Learning Rate) und die Anzahl der Epochen beliebig anpassen.
💡 Empfohlene Trainings-Strategie (Fine-Tuning):
Damit die KI optimal trainiert wird und die Daten nicht einfach auswendig lernt (Overfitting), wurde folgende dreistufige Strategie erfolgreich getestet:
1. Phase 1: 5 Epochen mit einer Lernrate von 0.01f
2. Phase 2: 3 Epochen mit einer Lernrate von 0.0001f
3. Phase 3: 1 Epoche mit einer Lernrate von 0.00005f

3. Modell ausführen

Sobald Sie das trainierte Modell besitzen oder wenn Sie direkt die vortrainierte Version nutzen möchten:
• Führen Sie die Datei RunModel.cpp aus und staunen Sie über das Ergebnis!

