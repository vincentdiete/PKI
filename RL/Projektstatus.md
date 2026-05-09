Kurze Übersicht Projektstatus bzgl. RL

Aktuelle Unterteilung:
Generell: Hauptunterschied: Umgebungen selbst Aktionsraum, Verhalten der Umgebung etc.
          Evaluate, Trainigs und Testklassen immer sehr ähnlich
1. first_env_ql:
    Enthält den ersten Versuch q learning zu implementieren: Kleines Feld. Agent kann sich nur bewegen. 2 Gegner sind stationär
2. grid_enemy ql:
    Agent kann jetzt laufen und schießen. Spiel vorbei wenn beide Ziele eliminiert.
3. enemy_survival_ql:
    Gegner bewegen sich auf den Agenten zu, Der Agent kann entweder schießen oder sich bewegen. Reward Design nach Kills und Überlebenszeit ausgerichtet. Gegner Respawnen wenn sie getötet wurden. Spiel vorbei, entweder wenn Agent alle Steps überlebt hat oder gestorben ist (Stirbt bei Kollision mit Gegnern)
    Lernen 50000 Episoden (Tabellarisches Q-Learning):
        Episode 1000, Avg reward last 1000: -1.78, Avg kills: 0.39, Avg steps alive: 3.44, Death rate: 100.00%, Epsilon: 0.606, Q-table states: 2546
        Episode 2000, Avg reward last 1000: -1.62, Avg kills: 0.56, Avg steps alive: 3.75, Death rate: 100.00%, Epsilon: 0.368, Q-table states: 4316
        Episode 3000, Avg reward last 1000: -1.36, Avg kills: 0.84, Avg steps alive: 4.25, Death rate: 100.00%, Epsilon: 0.223, Q-table states: 5807
        Episode 4000, Avg reward last 1000: -0.88, Avg kills: 1.33, Avg steps alive: 4.97, Death rate: 100.00%, Epsilon: 0.135, Q-table states: 7057
        Episode 5000, Avg reward last 1000: -0.04, Avg kills: 2.24, Avg steps alive: 6.82, Death rate: 100.00%, Epsilon: 0.082, Q-table states: 8355
        Episode 6000, Avg reward last 1000: 1.06, Avg kills: 3.41, Avg steps alive: 8.88, Death rate: 100.00%, Epsilon: 0.050, Q-table states: 9679
        Episode 7000, Avg reward last 1000: 1.92, Avg kills: 4.34, Avg steps alive: 10.64, Death rate: 100.00%, Epsilon: 0.050, Q-table states: 10947
        Episode 8000, Avg reward last 1000: 2.72, Avg kills: 5.18, Avg steps alive: 12.29, Death rate: 100.00%, Epsilon: 0.050, Q-table states: 12186
        Episode 9000, Avg reward last 1000: 3.07, Avg kills: 5.57, Avg steps alive: 13.24, Death rate: 100.00%, Epsilon: 0.050, Q-table states: 13262
        Episode 10000, Avg reward last 1000: 3.46, Avg kills: 5.99, Avg steps alive: 14.22, Death rate: 100.00%, Epsilon: 0.050, Q-table states: 14344
        Episode 11000, Avg reward last 1000: 3.86, Avg kills: 6.42, Avg steps alive: 15.31, Death rate: 100.00%, Epsilon: 0.050, Q-table states: 15453
        Episode 12000, Avg reward last 1000: 4.29, Avg kills: 6.88, Avg steps alive: 16.50, Death rate: 100.00%, Epsilon: 0.050, Q-table states: 16576
        Episode 13000, Avg reward last 1000: 4.52, Avg kills: 7.12, Avg steps alive: 17.34, Death rate: 100.00%, Epsilon: 0.050, Q-table states: 17499
        Episode 14000, Avg reward last 1000: 5.11, Avg kills: 7.76, Avg steps alive: 19.54, Death rate: 99.90%, Epsilon: 0.050, Q-table states: 18426
        Episode 15000, Avg reward last 1000: 5.41, Avg kills: 8.09, Avg steps alive: 20.66, Death rate: 99.90%, Epsilon: 0.050, Q-table states: 19335
        Episode 16000, Avg reward last 1000: 5.49, Avg kills: 8.19, Avg steps alive: 21.58, Death rate: 99.90%, Epsilon: 0.050, Q-table states: 20283
        Episode 17000, Avg reward last 1000: 5.80, Avg kills: 8.52, Avg steps alive: 23.10, Death rate: 99.50%, Epsilon: 0.050, Q-table states: 21109
        Episode 18000, Avg reward last 1000: 5.82, Avg kills: 8.54, Avg steps alive: 24.06, Death rate: 99.80%, Epsilon: 0.050, Q-table states: 21994
        Episode 19000, Avg reward last 1000: 6.28, Avg kills: 9.03, Avg steps alive: 24.51, Death rate: 99.60%, Epsilon: 0.050, Q-table states: 22812
        Episode 20000, Avg reward last 1000: 5.88, Avg kills: 8.63, Avg steps alive: 25.25, Death rate: 99.30%, Epsilon: 0.050, Q-table states: 23598
        Episode 21000, Avg reward last 1000: 6.24, Avg kills: 9.00, Avg steps alive: 25.51, Death rate: 99.70%, Epsilon: 0.050, Q-table states: 24411
        Episode 22000, Avg reward last 1000: 6.60, Avg kills: 9.37, Avg steps alive: 26.37, Death rate: 99.10%, Epsilon: 0.050, Q-table states: 25128
        Episode 23000, Avg reward last 1000: 7.27, Avg kills: 10.09, Avg steps alive: 27.71, Death rate: 98.90%, Epsilon: 0.050, Q-table states: 25904
        Episode 24000, Avg reward last 1000: 7.08, Avg kills: 9.89, Avg steps alive: 26.82, Death rate: 99.20%, Epsilon: 0.050, Q-table states: 26653
        Episode 25000, Avg reward last 1000: 6.58, Avg kills: 9.39, Avg steps alive: 26.84, Death rate: 99.40%, Epsilon: 0.050, Q-table states: 27426
        Episode 26000, Avg reward last 1000: 7.58, Avg kills: 10.42, Avg steps alive: 28.59, Death rate: 99.20%, Epsilon: 0.050, Q-table states: 28208
        Episode 27000, Avg reward last 1000: 7.45, Avg kills: 10.25, Avg steps alive: 27.00, Death rate: 98.80%, Epsilon: 0.050, Q-table states: 28901
        Episode 28000, Avg reward last 1000: 7.42, Avg kills: 10.23, Avg steps alive: 27.22, Death rate: 98.70%, Epsilon: 0.050, Q-table states: 29558
        Episode 29000, Avg reward last 1000: 7.53, Avg kills: 10.34, Avg steps alive: 28.09, Death rate: 98.30%, Epsilon: 0.050, Q-table states: 30191
        Episode 30000, Avg reward last 1000: 7.88, Avg kills: 10.74, Avg steps alive: 29.45, Death rate: 98.60%, Epsilon: 0.050, Q-table states: 30859
        Episode 31000, Avg reward last 1000: 8.27, Avg kills: 11.17, Avg steps alive: 30.05, Death rate: 98.70%, Epsilon: 0.050, Q-table states: 31437
        Episode 32000, Avg reward last 1000: 7.69, Avg kills: 10.54, Avg steps alive: 29.90, Death rate: 98.30%, Epsilon: 0.050, Q-table states: 32009
        Episode 33000, Avg reward last 1000: 7.90, Avg kills: 10.77, Avg steps alive: 30.20, Death rate: 98.40%, Epsilon: 0.050, Q-table states: 32566
        Episode 34000, Avg reward last 1000: 7.91, Avg kills: 10.79, Avg steps alive: 31.07, Death rate: 98.50%, Epsilon: 0.050, Q-table states: 33135
        Episode 35000, Avg reward last 1000: 8.33, Avg kills: 11.22, Avg steps alive: 30.52, Death rate: 97.80%, Epsilon: 0.050, Q-table states: 33706
        Episode 36000, Avg reward last 1000: 8.36, Avg kills: 11.25, Avg steps alive: 30.66, Death rate: 98.20%, Epsilon: 0.050, Q-table states: 34226
        Episode 37000, Avg reward last 1000: 8.20, Avg kills: 11.08, Avg steps alive: 31.25, Death rate: 98.10%, Epsilon: 0.050, Q-table states: 34808
        Episode 38000, Avg reward last 1000: 7.96, Avg kills: 10.83, Avg steps alive: 30.47, Death rate: 98.00%, Epsilon: 0.050, Q-table states: 35395
        Episode 39000, Avg reward last 1000: 8.61, Avg kills: 11.51, Avg steps alive: 31.52, Death rate: 97.80%, Epsilon: 0.050, Q-table states: 35898
        Episode 40000, Avg reward last 1000: 8.34, Avg kills: 11.22, Avg steps alive: 31.68, Death rate: 97.60%, Epsilon: 0.050, Q-table states: 36405
        Episode 41000, Avg reward last 1000: 8.63, Avg kills: 11.52, Avg steps alive: 31.61, Death rate: 97.40%, Epsilon: 0.050, Q-table states: 36908
        Episode 42000, Avg reward last 1000: 8.60, Avg kills: 11.50, Avg steps alive: 32.38, Death rate: 97.60%, Epsilon: 0.050, Q-table states: 37384
        Episode 43000, Avg reward last 1000: 8.37, Avg kills: 11.24, Avg steps alive: 31.72, Death rate: 97.00%, Epsilon: 0.050, Q-table states: 37867
        Episode 44000, Avg reward last 1000: 8.94, Avg kills: 11.86, Avg steps alive: 32.56, Death rate: 97.30%, Epsilon: 0.050, Q-table states: 38417
        Episode 45000, Avg reward last 1000: 8.76, Avg kills: 11.67, Avg steps alive: 32.35, Death rate: 97.70%, Epsilon: 0.050, Q-table states: 38836
        Episode 46000, Avg reward last 1000: 9.33, Avg kills: 12.26, Avg steps alive: 32.88, Death rate: 97.50%, Epsilon: 0.050, Q-table states: 39264
        Episode 47000, Avg reward last 1000: 9.11, Avg kills: 12.05, Avg steps alive: 33.74, Death rate: 97.30%, Epsilon: 0.050, Q-table states: 39737
        Episode 48000, Avg reward last 1000: 8.95, Avg kills: 11.88, Avg steps alive: 32.99, Death rate: 97.50%, Epsilon: 0.050, Q-table states: 40180
        Episode 49000, Avg reward last 1000: 9.38, Avg kills: 12.32, Avg steps alive: 33.97, Death rate: 96.60%, Epsilon: 0.050, Q-table states: 40580
        Episode 50000, Avg reward last 1000: 9.35, Avg kills: 12.28, Avg steps alive: 33.87, Death rate: 96.20%, Epsilon: 0.050, Q-table states: 40961

        Training abgeschlossen.
        Gelernte Zustände: 40961

    Weiteres Vorgehen: 
        Aktuelles Modell verbessern (Rewardstruktur, Gegnerverhalten)

Folgeschritte:
    4. DQN implementieren
    5. Zustandsraum kontinuierlich machen
    6. Umgebung und Aktionsraum erweitern