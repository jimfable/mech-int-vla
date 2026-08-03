# Does Internal Geometry Break Before Robot Behavior Does?
**Projektdokument v2, Stand 2. August 2026 — final. Änderungen ab jetzt ausschließlich über AMENDMENTS.md mit Datum, Commit-Hash und Begründung.**
 
---
 
## 1. Primäre Behauptung (fixiert)
 
**Primärfrage:** Verbessern vorab definierte interne Geometrie- und Controllability-Signale die Held-out-Vorhersage von Closed-Loop-Fehlschlägen gegenüber dem stärksten nicht-internen Basismodell?
 
**Primärer Estimand:** Gepaarte Differenz der Out-of-Sample-Log-Loss zwischen M2 und M1 (Definition in §3) auf dem Locked Test Set.
 
**Wichtigste Sekundärfrage (Lead Time):** Erzeugen interne Signale bei festgelegter episode-level False-Positive-Rate einen früheren Alarm als behaviorale Signale?
 
Brier Score, AUROC, Condition-Level-Ranking und Patching-Spezifität sind Sekundär- bzw. Stützmetriken. Nur der Log-Loss-Lift entscheidet die primäre Vorhersagebehauptung.
 
Nicht Teil von Monat 1: Data-Acquisition-Race, SAEs, Hardware, Zertifizierungsnarrativ.
 
## 2. Kontext in einem Absatz
 
Das Projekt testet, ob Mechanistic Interpretability auf Vision-Language-Action-Modellen inkrementellen, messbaren Wert über nicht-interne Diagnosen hinaus liefert. Es ist zugleich ein persönlicher Feldtest: ein Monat, öffentliche Artefakte (Repo + 3 Blogposts), danach eine Entscheidung anhand vordefinierter Kriterien.
 
## 3. Vergleichsmodelle und Budgetdefinition
 
Drei verschachtelte Prädiktionsmodelle:
 
- **M0** = Counterfactual Action Drift + Output-Uncertainty (rein outputbasiert, deployment-realistisch)
- **M1** = M0 + Simulator-State-Coverage (nicht-intern, aber simulatorprivilegiert)
- **M2** = M1 + interne Geometrie-/Controllability-Signale
**Primärer Vergleich:** M2 vs. M1 (liefert White-Box-Zugriff Wert über alle nicht-internen Informationen hinaus?). **Sekundärer Vergleich:** M2 vs. M0 (deployment-relevante Lesart). Ein Ergebnis "schlägt M0, nicht M1" ist kein Nullresultat, sondern eine präzise Grenze des Nutzens.
 
**Budgetdefinition (gleiches Informationsbudget + transparente Kostenbilanz):** Alle Methoden erhalten dieselben Candidate States, dieselben Counterfactual-Transformationen und dieselben Train/Calibration/Test-Splits. Locked-Test-Erfolgslabels dürfen nicht zur Konstruktion oder Auswahl von Scores verwendet werden. Pro Methode werden bilanziert: Policy-Forward-Passes, Interventions-Passes, GPU-Sekunden, Aktivierungsspeicher, benötigte Kalibrierungsepisoden. Ergebnisse werden einmal inklusive einmaliger Kalibrierungskosten und einmal unter expliziter Amortisationsannahme berichtet.
 
## 4. Eingefrorene Scope-Entscheidungen
 
| Bestandteil | Entscheidung |
|---|---|
| Modell | Öffentlicher SmolVLA-LIBERO-Checkpoint (lerobot/smolvla_libero) |
| Umgebung | Bestehendes LIBERO, eine Aufgabe aus präregistrierter Shortlist (§5) |
| Eigener Task / Scripted Expert / Policy-Training | Nein |
| Konstruierter Shortcut im Datensatz | Nein (höchstens Unit-Test, nie Hauptergebnis) |
| Aktivierungsorte | Max. 3, vorab gewählt (finaler VLM-Kontext, früher + später Action-Expert-Block); bei Flow Matching 1–2 fixe Flow-Timesteps |
| SAEs | Nein |
| Primäre Variable | Relative planare Orientierung (symmetry-aware); Fallback-Reihenfolge in §5 |
| Interne Methoden | Symmetry-aware Circular Probes; DAS / isoliertes Subspace-Patching mit Kontrollen nach §8 |
| Ground Truth | Simulatorpose; Kontakt-Flags (Rolle: Fehlermodus 3 in §5 und Definition von "missed grasp"); Closed-Loop-Erfolg |
| Fallback bei Repro-Problemen | OpenVLA-OFT-LIBERO-Stack; keine Parallel-Baustellen |
 
## 5. Präregistrierte Task- und Fehlermodus-Auswahl
 
**Task-Auswahl:** Vor dem ersten Rollout wird in PREREG.md eine geordnete Shortlist von max. 3 LIBERO-Aufgaben mit orientierungsrelevanten Manipulanda fixiert. Verwendet wird die **erste** Aufgabe der Liste, die auf Discovery-Seeds das Dynamic-Range-Gate besteht:
 
> Baseline-IID-Erfolgsrate ≥ 60% **und** Fehlschlagrate unter den präregistrierten Perturbationen im Bereich 20–80% (nichtdegeneriert). Schwellen fixiert vor dem ersten Rollout.
 
**Fehlermodus-/Variablen-Reihenfolge (fixiert):** 1. relative Orientierung → 2. relative planare Position → 3. Kontakt / missed grasp. **Höchstens ein** methodischer Wechsel nach dem Reality Gate. Alle versuchten Aufgaben werden veröffentlicht, auch ungeeignete.
 
## 6. Datensplits (strikt getrennt)
 
- **Discovery Set:** technische Reproduktion, Task-Auswahl per Gate, Identifikation unbrauchbarer Perturbationen, Debugging der Instrumentierung.
- **Calibration Set:** Probe-Training; Auswahl von genau einem der max. 3 Aktivierungsorte; Auswahl genau einer Probe; Fit des Failure-Predictors; Alarm-Schwellen; Interventionsstärke. Erfolgslabels sind hier erlaubt.
- **Locked Test Set:** erst nach Lock-Commit; primäre Log-Loss-Differenz, Brier, Condition-Ranking, Lead Time, finale Patching-Evaluation.
Alle Splits gruppiert nach **Episode, Reset-Seed, Perturbationsbedingung und ggf. Objektinstanz**. Kein Frame gelangt über Trajektoriennähe indirekt in mehrere Splits. n = unabhängige Episoden/Bedingungen; Unsicherheit per Cluster-Bootstrap darüber, nie über Frames.
 
## 7. Begriffsdisziplin: Equivarianz vs. Action Drift
 
"Output-Equivarianzfehler" (E_out(x,g) = ‖π(gx) − T_g·π(x)‖) wird **nur** verwendet, wenn T_g analytisch definiert und als Symmetrie von Simulatorzustand, Aufgabe und Action-Koordinatensystem validiert ist:
 
- Task-erhaltende Render-Perturbationen (Farbe, Textur, Beleuchtung, kleine Kamera): T_g = I → echtes Equivarianzmaß.
- Exakt validierte räumliche Symmetrien: echtes Equivarianzmaß.
- Sonstige Poseveränderungen: neutral **"counterfactual action drift"** als prädiktiver Score, ohne normativen Equivarianzclaim (Kinematik, Hindernisse, multimodale Griffe und Workspace-Grenzen können die einfache Symmetrie brechen).
## 8. Kausaltest-Protokoll (Woche 3)
 
Donor- und Recipient-Zustände werden innerhalb definierter Toleranzen nach Taskphase, Propriozeption und allen nicht untersuchten Simulatorvariablen **gematcht**; unzureichend gematchte Paare fließen nicht in den konfirmatorischen Test. Erforderliche Kontrollen:
 
1. **Sign-correctness:** Zielaktionsdimension bewegt sich in die vorhergesagte Richtung.
2. **Specificity:** begrenzte Veränderung in nicht adressierten Aktionsdimensionen.
3. **Norm-gleiche Random Controls.**
4. **Off-Manifold-Check:** Distanz der gepatchten Aktivierung zur natürlichen Aktivierungsverteilung.
5. **Matched-Donor-Control:** Patch aus ähnlichem Zustand ohne die relevante geometrische Differenz.
## 9. Operationalisierung (Vorschlagswerte — endgültig zu fixieren in PREREG.md vor dem ersten Rollout)
 
| Begriff | Operationalisierung (Vorschlag) |
|---|---|
| Predictive Lift (primär) | Relative OOS-Log-Loss-Verbesserung M2 vs. M1 ≥ 3%, Cluster-Bootstrap-90%-Intervall schließt 0 aus |
| "Nahezu perfekt" (Kill-Switch 1) | M0 oder M1 erreicht AUROC ≥ 0,95 auf Calibration → Fehlermodus-Wechsel nach §5-Reihenfolge |
| Lead Time | Medianer Vorsprung ≥ 5 Kontrollschritte bei episode-level FPR = 10%, Alarm erst nach k = 3 aufeinanderfolgenden Schwellenüberschreitungen |
| Patching-Spezifität | Zielwirkung > 95. Perzentil der Random-Control-Verteilung bei Off-Target-Änderung ≤ 25% der Zielwirkung |
| Stabilität | Gleiche Effektrichtung über ≥ 2 von 3 präregistrierten Seeds und an mind. 2 Aktivierungsorten |
| "Instabil" | Effekt existiert nur bei einem einzigen Seed, Layer oder Interventionswert → als instabil berichten, nicht umdeuten |
 
Die konkreten Grenzwerte sind weniger wichtig als ihre Festlegung vor dem Locked Test; Anpassung nur via AMENDMENTS.md **vor** Öffnung des Locked Test Sets.
 
## 10. Monatsplan
 
**3.–5. Aug — Reality Gate.** Checkpoint reproduzierbar ausführen; Shortlist-Gate anwenden; kontrollierte Pose-/Kamera-Variation; erste erfolgreiche und fehlgeschlagene Rollouts (Discovery). Scheitert die Repro: LeRobot-Version pinnen oder Fallback-Stack, nichts parallel bauen. Danach: **Lock-Commit / Git-Tag `prereg-locked-v1`.**
 
**Rest Woche 1 — Black-Box Ceiling.** Counterfactual-Grid (Perturbationsfamilien nach §7); M0- und M1-Signale auf Calibration berechnen; Failure-Labels nur auf Discovery+Calibration. **Kill-Switch 1** nach §9.
 
**Woche 2 — Geometrie.** 2–3 Aktivierungsorte instrumentieren; zirkuläre und symmetry-aware Probes; Splits nach §6; auf Calibration genau einen Layer und eine Probe wählen, dann einfrieren.
 
**Woche 3 — Kausalität.** Gematchtes kontrafaktisches Patching mit allen Kontrollen aus §8; Lead-Time-Vorbereitung (Schwellen auf Calibration fixieren).
 
**Woche 4 — Locked Evaluation + Veröffentlichung.** Locked Test öffnen (zurückgehalten: Winkelintervalle, Reset-Seeds, mind. eine Perturbationsfamilie, möglichst zweite Aufgabe); Primär- und Sekundärmetriken nach §1/§9; Kostenbilanz nach §3; Repo bereinigen; drei Posts.
 
## 11. Repo-Struktur und Prä-Registrierung
 
```
START_DOCUMENT.md      (dieses Dokument)
PREREG.md              (erster Commit vor jedem Code: Estimand, Splits, Task-Shortlist,
                        Schwellen aus §9, Entscheidungstabelle)
AMENDMENTS.md          (jede spätere Änderung: Datum, Commit-Hash, technischer Grund,
                        betroffene Hypothesen, Zeitpunkt relativ zur Ergebniseinsicht;
                        nach Öffnung des Locked Test Sets nur noch explorativ)
EXPERIMENT_LOG.md
configs/task_order.yaml  split_protocol.yaml  perturbations.yaml
environment.lock
```
 
## 12. Entscheidungstabelle nach Monat 1
 
| Ergebnis (gemäß §9-Schwellen) | Konsequenz |
|---|---|
| Predictive Lift (M2>M1) **und** spezifisches Patching | Monat 2: Acquisition Race (Random vs. stärkste behaviorale Methode vs. behavioral+CRG; ein Budget, gepaarte Checkpoints, mehr Seeds) |
| Kein durchschnittlicher Lift, aber klarer Vorteil bei delayed failures / missed grasps / taskrelevanten Transformationen | Enger, wertvoller Spezialbefund; publizieren |
| Lift ohne kausale Spezifität | Guter Failure Detector, schwacher Mech-Interp-Claim; ehrlich so berichten |
| Kausale Spezifität ohne Lift | Wissenschaftlich interessant, ökonomisch schwach; publizieren, Acquisition-These pausieren |
| Weder Lift noch Spezifität | Projekt beenden, Negativbefund publizieren |
| Lift nur M2>M0, nicht M2>M1 | Präzise Nutzengrenze berichten: Internals ersetzen privilegierten Zustand, übertreffen ihn nicht |
 
## 13. Was wir wissen / nicht wissen
 
**In den Vorarbeiten berichtete Ergebnisse (nicht selbst repliziert):** Aktivations-Steering (2509.00328); Observability/Controllability auf π0.5/OpenVLA (2603.05487); SAE-Memorierungs-Befund (2603.19183); Failure Detection aus Internals (SAFE, 2506.09937); event-grounded SAEs (2605.17204); Sechs-Modell-Studie (2603.19233); rolloutfreie behaviorale Counterfactual-Diagnose (CFNBC, 2607.27261); Tri-Info (2606.19998); LIBERO-PRO-Memorisation (2510.03827); Grenzen linearer Steering-Methoden bei Flow Matching (DiMaS, 2607.14280); SAE-Manifold-Fragmentierung (2604.28119).
 
**Konzeptuelle Sicherheiten:** Dekodierbarkeit ≠ Mechanismus; Simulation ist Falsifikationsdomäne, kein Mechanismus-Orakel; Splits/Bootstrap/Vorab-Festlegung nicht verhandelbar.
 
**Offene Unsicherheiten:** (1) Größtes Monatsrisiko: ob der öffentliche Checkpoint saubere geometrische Fehlermoden hat oder diffuse Mittelmäßigkeit — abgefedert durch §5-Reihenfolge und Kill-Switch 1. (2) Die Primärfrage selbst; das Negativergebnis ist ein zulässiger, publizierbarer Ausgang. (3) Tragfähigkeit von Probes/Patching auf dem Flow-Matching-Action-Expert. (4) Repro des Stacks (Tag 1–3, Fallback definiert). (5) Statistische Power auf Episodenebene.
 
## 14. Deliverables, Distribution, Erfolgskriterien
 
**Deliverables:** schlankes reproduzierbares Repo (§11); 3 Posts: (1) Framing + Prä-Registrierung + Black-Box-Ceiling, (2) "The Difference Between Not Knowing and Not Using", (3) "Does Internal Geometry Break Before Robot Behavior Does?" (Audit-Vision höchstens im Schlussabsatz von Post 3).
 
**Distribution:** eigener Blog + X; Crosspost LessWrong/Alignment Forum; Direktmails an Autoren der Vorarbeiten (Stanford Pavone/Schwager/Kennedy, Berkeley Tomlin, TRI, CFNBC-/Tri-Info-Autoren).
 
**Interesse-Test (zählt):** Reproduktion durch Dritte; fachliche Issues/PRs aktiver Gruppen; mind. eine "wendet das auf unsere Policy an"-Anfrage; konkretes Kollaborations- oder Compute-/Datenangebot. Vanity-Nebenmetrik: Stars, Aufrufe.
 
## 15. Ressourcen, Abbruch, Meta-Regel
 
**Budget:** 1 Miet-GPU (4090/A100), niedrige hunderte Euro; 1 Monat committed, keine Parallelprojekte im kritischen Pfad.
 
**Abbruch/Umschwenken:** Reality Gate scheitert auch mit Fallback-Stack → Stack-Wechsel, Scope kürzen. Kein Task besteht das Dynamic-Range-Gate → Shortcut-/Robustheitsbefund als Post 1, Projektentscheidung neu. Instabilität nach §9 → als instabil berichten.
 
**Meta-Regel:** Dies ist v2 und die letzte Planversion. Es gibt keine v3. Jede Änderung läuft über AMENDMENTS.md; jede "Ideenverbesserung" ohne technischen Blocker ist per Definition Prokrastination. Ab 3. August zählt, was auf der GPU passiert.
 
---
 
*Commercial Notes (außerhalb des Experiments, kann veralten): Regulierungs-Timing für High-Risk-Anforderungen an maschinenintegrierte KI ist unsicher und für Monat 1 irrelevant; die kommerzielle Anschlussthese (Representation-guided Data Acquisition) wird erst nach Ausgang 1 der Entscheidungstabelle wieder angefasst.*