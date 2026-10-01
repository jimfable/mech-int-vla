# Adversarial Review des Experiments (2026-10-01)

Drei unabhängige Subagenten haben das Projekt geprüft: (1) Methodik des Gemachten, (2) fehlende Experimente mit
Kostenschätzung, (3) Einordnung in die Literatur aus Sicht einer skeptischen Mentorin. Die folgenden Punkte sind
zusammengeführt. Wo ich eine Behauptung selbst nachgeprüft habe, steht **[geprüft]**; sonst **[Prüfer, nicht
nachgeprüft]**.

Hinweis: Für die Diagnose des Patching-Tests haben Prüfer und ich die gespeicherten Evidenzdateien des Locked Test
gelesen. Das ist eine Diagnose der Methode, keine neue konfirmatorische Auswertung; die vorab festgelegten Ergebnisse
bleiben, wie sie sind.

## Budget-Rahmen

- Bisher ausgegeben: **54,15 $** laut Vast-Abrechnung **[geprüft]**: Instanz 46677323 (Discovery und Calibration,
  August, inkl. Speicher im gestoppten Zustand) 37,96 $; vier Hosts, die nicht starteten, 0,21 $; Instanz 51243106
  (Locked Test, September, inkl. Aufbau und Speicher) 15,97 $, davon 14,81 $ reine GPU-Zeit laut Cost-Receipt.
  Eingezahlt seit August: 54 $ (10 + 10 + 7 + 5 + 15 + 7).
- Geplant war „niedrige hunderte Euro“. Der größte Teil ist also unverbraucht.
- Die Vast-Instanz existiert nicht mehr, Guthaben 1,54 $ **[geprüft]**. Neue GPU-Arbeit braucht eine neue Instanz und
  einen Neuaufbau (≈ 2,5 GPU-h ≈ 1,7 $).
- Einheitskosten (RTX 5090, 0,68 $/h): Rollout ≈ 0,02 $, volles Scoring ≈ 0,06 $ pro Episode, 1.000
  Patching-Durchläufe ≈ 0,03 $.

## Teil 1: Bewertung des Gemachten

### Schwerwiegend

**1. Der Patching-Test (Claim 3) war vermutlich nicht empfindlich genug, um überhaupt etwas zu finden.**
- Das Netz rechnet an der Eingriffsstelle in bf16 **[geprüft]**: 100 % der gespeicherten Aktivierungen haben null in
  den unteren 16 Bit (Kontrolle: Simulatordaten 0 %). Die Verschiebung wird vor dem Addieren in bf16 umgewandelt
  (`src/mech_int_vla/instrumentation.py:731`) **[geprüft]**.
- Die Verschiebung ist winzig: Norm ≈ 0,10 bei einer Aktivierungsnorm von ≈ 39, also ≈ 0,004 pro Element. Das liegt
  in der Größenordnung eines halben bf16-Rundungsschritts. [Prüfer; Aktivierungsnorm 39 geprüft]
- Probe-Eingriff und Zufallsrichtungen bewegen die Aktion gleich stark (Median 5,4·10⁻⁴ gegen 5,2·10⁻⁴) und
  korrelieren pro Paar mit 0,72 **[geprüft]**. Das spricht für eine generische Reaktion auf jede kleine Störung,
  nicht für etwas Spezifisches.
- Keine Dosis-Wirkung: Bei α = 0,5 und 1,0 werden die Effekte nicht größer **[geprüft im Sensitivity-Receipt]**.
- Ein Off-Target-Verhältnis von 4,0 ist ungefähr das, was isotropes 7-D-Rauschen liefert (≈ 3,4). [Prüfer]
- Es gab **keine Positivkontrolle** (z. B. die ganze Donor-Aktivierung einsetzen), die zeigt, dass ein Eingriff an
  dieser Stelle die Aktion überhaupt bewegen kann.
- Eingegriffen wurde nur im ersten von 10 Verfeinerungsschritten. Die neun späteren lesen den unveränderten
  VLM-Kontext erneut und können den Eingriff ausgleichen **[geprüft: Code verlangt genau einen Schritt]**.
- **Folge:** Claim 3 sollte von „eine Ablesung, kein Hebel“ auf „**nicht schlüssig getestet**“ herabgestuft werden.
  Das betrifft Bericht und Blogpost.

**2. Claim 1 war fast eingebaut.** M1 enthält bereits genau den Winkel, den der Probe abliest
(`m1_symmetry_eef_object_yaw_sin/cos`) **[geprüft]**. M2 konnte M1 nur schlagen, wenn das Netz den Winkel *falsch*
wahrnimmt, und genau das hat M2 nicht gemessen. In einem deterministischen Simulator ist die Aktivierung außerdem
eine Funktion des Zustands. Der Satz „Internals bringen nichts über den Zustand hinaus“ wurde zudem nie mit
**M1 + rohe Aktivierungen** getestet.

**3. Die Fehler sind Timeouts, keine Brüche.** Auf dem Locked Test sind 47 von 60 Fehlern Timeouts und 13 Austritte
aus dem Arbeitsraum; keine Griff- oder Fallfehler. [Prüfer] Damit ist die Lead-Time-Frage (Claim 4) nicht negativ,
sondern **nicht testbar**: Es gibt keinen Zeitpunkt, an dem etwas „bricht“. Ein großer Teil der Vorhersage folgt aus
der Störungsbedingung am Start (AUROC innerhalb einer Bedingung nur 0,67–0,73). [Prüfer]

### Mittel

**4. Der Output-Monitor M0 ist schwach und nicht realistisch.** Seine Log Loss (0,637) ist nur ≈ 4 % besser als ein
konstanter Vorhersager (0,664) **[geprüft, Rechnung]**. Er hat keine Gelenkstellungen und keine Zeit; beides steckt
in M1, obwohl ein echter Roboter beides hat **[geprüft in `features.py`]**. Seine Kamera- und Objekt-Kontrafaktuale
brauchen den Simulator. Das schwächt auch Ergebnis 2: „Internals ersetzen Simulator-Zustand teilweise“ könnte zum
Teil heißen „Internals enthalten Gelenkstellung und Fortschritt“.

**5. Power-Aussage falsch formuliert.** Bei einem Standardfehler von ≈ 1 % hätte ein echter 3-%-Effekt die Hürde nur
etwa zur Hälfte geschafft; ≈ 80 % Power gibt es erst bei ≈ 4 %. Der Satz „a real improvement of 3% would most likely
have shown up“ (Bericht und Blog) ist so nicht richtig. [Prüfer, Rechnung plausibel]

**6. Off-Manifold-Zahlen sind ein Artefakt.** Schon 31 von 52 *ungepatchten* Empfängern liegen über der Schwelle;
das Patching ändert bei 0 von 52 etwas. Die 0 % auf Calibration kamen durch eine Referenz, die die eigene Episode
enthielt. Der Satz im Bericht „not explained by an overly cautious dose“ ist damit nicht gedeckt. (Im Blog ist er
schon gestrichen.) [Prüfer, nicht nachgeprüft]

**7. Kill-Switch knapp.** M1 lag auf Calibration bei 0,937 bzw. 0,949 (pro Lauf); 34 % der Bootstrap-Ziehungen
erreichen 0,95. Die Aufgabe war nach eigener Regel fast „zu leicht“. [Prüfer]

### Klein

- Die „Vorhersage“ vom 25. August (M2 ≈ M1 ≫ M0) wurde nach den Calibration-Ergebnissen geschrieben, ist also eine
  Postdiktion. Der Blog-Satz „I also wrote down what I expected“ sollte das sagen. [Prüfer]
- Mehrere kleinere Fehler im Bericht (AUROC-Definitionen vermischt, „first command is read“ statt Mittel der ersten
  10 Aktionen, Lead-Time-CI mit 19 statt 20 Clustern u. a.). [Prüfer]

### Was gut war (relativ zum Budget)

- Echt zurückgehaltener Test mit neuen Winkeln und neuer Kamera-Familie, einmal ausgewertet, ohne Label-Leck.
- Saubere Statistik: Cluster-Bootstrap über Startszenen, gepaarter Estimand, Episoden gleich gewichtet.
- Starke Baselines von Anfang an eingebaut, Negativergebnis ehrlich berichtet.
- Echte Selbstkorrekturen (CI-Lesefehler, Überclaim zur Substitution) und Amendments, die eher *gegen* M2 wirkten.
- Billig: ≈ 15 $ für den Locked Test.
- Schwäche: viel Aufwand in Infrastruktur, aber kein Fünf-Minuten-Sanity-Check des Eingriffs.

## Teil 2: Was gemacht werden sollte

Bewertung 1–5. Kosten inkl. Anteil am Neuaufbau, falls GPU. „CPU“ = auf den vorhandenen Calibration-Daten, kein
GPU nötig.

| Rang | Was | Beantwortet | Impact | Wichtigkeit | Aufwand | Kosten |
|---|---|---|---|---|---|---|
| 0 | **Bericht und Blog korrigieren** (Claim 3 herabstufen, Power-Satz, Postdiktion) | Ehrlichkeit des Gemachten | 5 | 5 | 1–2 h | 0 $ |
| 1 | **Positivkontrolle für Patching:** ganze Donor-Aktivierung, alle 10 Schritte, in fp32, plus Gradienten-Richtung | Kann der Test überhaupt etwas finden? Steht Claim 3? | 5 | 5 | ≈ 5 h | ≈ 0,5 $ GPU (+ Neuaufbau) |
| 2 | **Nutzt die Policy den Winkel überhaupt?** Greifer-Drehung beim Griff gegen Buchwinkel | Gibt es einen Hebel zu finden? | 4 | 5 | ≈ 3 h | CPU |
| 3 | **M0⁺ (realistisch: + Gelenke, Zeit, Konsistenz) und M0⁺ + Aktivierungen, M1 + Aktivierungen** | Ist Ergebnis 2 nur Gelenkstellung/Fortschritt? Bringen rohe Internals etwas über M1? | 4 | 5 | ≈ 4 h | CPU |
| 4 | **Probes für das, was scheitert** (Kontakt, Griff, Phase, Fortschritt) + „Wahrnehmungsfehler“-Features | Der einzige prinzipielle Weg, wie Internals M1 schlagen können | 4 | 5 | ≈ 6 h | CPU |
| 5 | **Fehler-Anatomie und Bedingungs-Confound** (AUROC nur aus Schritt 0, AUROC innerhalb einer Bedingung) | Misst der Monitor nur die Störung am Start? | 3 | 4 | ≈ 5 h | CPU |
| 6 | **Lokalisierung:** ganze Aktivierung Layer für Layer tauschen, auch VLM-Bild-Tokens | Wo kommt der Winkel in die Aktion? | 5 | 5 | ≈ 8 h | ≈ 0,3 $ GPU |
| 7 | **Aufgabe mit plötzlichen Fehlern suchen** (10 Rollouts je LIBERO-10-Aufgabe) | Gibt es eine Aufgabe, in der „Brechen“ testbar ist? | 3 | 4 | ≈ 4 h | ≈ 2 $ |
| 8 | **Ablation im geschlossenen Regelkreis** (Probe-Unterraum vs. zufällig vs. Positivkontrolle) | Ändert sich die Erfolgsrate? | 4 | 4 | ≈ 6 h | ≈ 3 $ |
| 9 | **DAS / gelernter Unterraum** an den lokalisierten Stellen | Echter Mechanismus-Test | 4 | 5 | ≈ 12 h | ≈ 3 $ |
| 10 | **Neuer konfirmatorischer Test** von Outputs + Internals auf frischen Daten (neue Prereg) | Ergebnis 2 ohne Post-hoc-Makel | 4 | 4 | ≈ 14 h | 6–13 $ |
| 11 | **Zweite Aufgabe mit plötzlichen Fehlern** (komplette Studie, neue Prereg), nur wenn 7 eine findet | Die ursprüngliche Frage, diesmal testbar | 5 | 5 | ≈ 28 h | ≈ 30 $ |

### Plan für ≈ 50 $

1. Rang 0 sofort.
2. CPU-Block (Rang 2–5), ≈ 18 h, 0 $. Entscheidet, welche GPU-Arbeit sich lohnt.
3. Ein GPU-Tag: Rang 1, dann 6 und 8 nur wenn die Positivkontrolle zeigt, dass der Test funktioniert; dazu Rang 7.
   ≈ 11 GPU-h ≈ 8 $.
4. Rang 10 mit billigem Scoring (≈ 8 $), nur wenn Rang 5 zeigt, dass der Monitor nicht bloß die Bedingung erkennt.

### Plan für ≈ 150 $

Alles aus dem 50-$-Plan, plus Rang 9, Rang 10 mit vollem Scoring und Rang 11, falls Rang 7 eine geeignete Aufgabe
findet. Sonst das Geld in einen größeren Rang 10.

### Was ich nicht machen würde

- Den Locked Test neu auswerten oder als konfirmatorisch wiederverwenden. Als Trainingsdaten darf er dienen.
- Eine größere M2-vs-M1-Wiederholung: Sie würde einen Effekt unter der eigenen Nützlichkeitsschwelle festnageln.
- Jetzt schon eine zweite Policy (π0, OpenVLA-OFT): 40–80 h neue Hooks, erst sinnvoll nach einem lokalisierten
  Mechanismus.
- Mehr α-Werte oder Probe-Varianten an der alten Stelle, bevor die Positivkontrolle zeigt, dass der Test funktioniert.

## Teil 3: Einordnung (Literatur)

- **Neu** ist vor allem: privilegierter Simulator-Zustand als explizite Obergrenze, Prä-Registrierung mit einer
  Auswertung, Patching gegen normgleiche Zufallsrichtungen. SAFE (2506.09937) hat keine Zustands-Baseline.
- **Repliziert** wird: Internals helfen gegenüber Outputs (Richtung wie SAFE). „Dekodierbar, aber nicht steuerbar“
  zeigt DiMaS (2607.14280) schon für SmolVLA-Action-Experten; 2608.13474 Ähnliches für π0.5.
- **Unbedingt zitieren:** 2609.34684 (hohe Dekodiergenauigkeit kann schwache Reaktion auf den echten Objektzustand
  verbergen) und DiMaS.
- **Beste Rahmung** laut Prüfer: „Interpretierbarkeits-Monitore gegen privilegierten Zustand *und* starke
  einsetzbare Baselines messen; Probes nur mit positiv kontrolliertem Patching prüfen“, als Protokoll plus
  Fallstudie. Veröffentlichbar als LessWrong/AF-Post; für einen Workshop nach Rang 1 und 3.

Quellen und Einzelheiten: Berichte der drei Prüfer in dieser Sitzung.
