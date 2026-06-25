# Todolist implementazione — Veo follow-cam, 9v9

Pipeline attuale: `run_pipeline.py` → `unified.csv` → `tactical_roles.py` → `unified_roles.csv`.

Obiettivo: **identità stabile (canonical)**, **presenza per-frame**, **ruoli tattici stabili** anche con giocatori fuori inquadratura, senza panoramico Veo.

---

## Fase 0 — Baseline e criteri di successo

- [ ] **0.1** Definire clip di test fisse (es. `test_10s`, `test_2min`) e annotare manualmente su 3–5 frame:
  - quanti giocatori visibili per squadra
  - 1–2 uscite/rientri evidenti
  - ruolo atteso di 2–3 giocatori (se conosciuto)
- [ ] **0.2** Salvare output baseline attuale (`unified.csv`, `unified_roles.csv`, debug JPG) come riferimento.
- [ ] **0.3** Definire metriche minime:
  - `n_canonical_per_team` ∈ [8, 12] (target ~10)
  - `% frame` con ruolo assegnato ai visibili
  - merge corretti su uscite/rientri noti (verifica manuale)
- [ ] **0.4** Scrivere in README (o nota interna) assunzioni:
  - formazione fissa **3-2-3** (o parametrizzabile)
  - nessun panoramico Veo
  - ruolo = proprietà lenta; presenza = proprietà per-frame

**Done when:** hai numeri baseline e casi manuali da confrontare.

---

## Fase 1 — Schema dati e presenza (`in_frame`)

### 1.1 Estendere output CSV

- [ ] **1.1.1** Aggiungere colonne a `unified_roles.csv` (o nuovo `unified_state.csv`):
  - `in_frame` : `0|1`
  - `presence_conf` : opzionale, es. `1.0` se detection diretta
  - (fase 2+) `reid_emb` non in CSV principale — file separato se serve
- [ ] **1.1.2** Mantenere retrocompatibilità: `canonical_id`, `role` restano; nuove colonne in coda.

### 1.2 Modulo presenza (nuovo o dentro `tactical_roles.py`)

- [ ] **1.2.1** Funzione `build_frame_roster(rows, frame_idx) -> dict[team, set[canonical_id]]`
  - input: righe player del frame con `canonical_id` già mappato
- [ ] **1.2.2** Funzione `update_presence(prev_roster, curr_roster) -> events`
  - eventi: `entered`, `exited` per `(team, canonical_id)`
- [ ] **1.2.3** Per ogni riga player: `in_frame=1`; per canonical assenti nel frame: opzionale riga sintetica `in_frame=0` **oppure** file separato `presence.csv` (consigliato per non gonfiare il CSV):

  ```text
  frame, timestamp_sec, team_id, canonical_id, in_frame
  ```

- [ ] **1.2.4** Regola: **uscita** = canonical non compare nel frame corrente ma era presente nel precedente (stessa squadra).
- [ ] **1.2.5** Regola: **rientro** = canonical assente da ≥N frame ricompare (o nuovo `player_id` mergeato → stesso canonical).

### 1.3 Debug presenza

- [ ] **1.3.1** Log per frame: `team0 visible=6 exited=[3] entered=[9]`
- [ ] **1.3.2** Overlay debug: colore pieno se `in_frame=1`, bbox tratteggiata / etichetta `OUT` se canonical noto ma assente (solo se esporti righe sintetiche o layer separato).

**Done when:** su `test_2min` vedi uscite/rientri coerenti con l'occhio su almeno 3 episodi.

**File toccati:** `tactical_roles.py` (`_write_csv`, `run`), eventualmente nuovo `presence.py`.

---

## Fase 2 — Merge tracklet robusto (uscita/rientro follow-cam)

Sostituire/affiancare `_merge_tracklets` attuale in `tactical_roles.py`.

### 2.1 Fattibilità fisica

- [ ] **2.1.1** Parametri CLI:
  - `--v-max-px-per-sec` (default da tarare, es. 400–800 px/s sul follow-cam)
  - deprecare o rendere fallback `--max-dist`
- [ ] **2.1.2** Nel merge, per coppia tracklet `(A end → B start)` con gap `Δt` frame e fps noto:
  - `max_dist = v_max_px_per_sec * (Δt / fps)`
  - merge solo se `hypot(dx, dy) <= max_dist`
- [ ] **2.1.3** Se il rientro è dal bordo frame: soglia più permissiva sul bordo (opzionale: `near_edge` se `x` o `y` entro 5% dai margini).

### 2.2 Vincolo cardinalità roster

- [ ] **2.2.1** Parametro `--roster-size` default `10` per squadra outfield (+ GK gestito a parte).
- [ ] **2.2.2** Dopo merge greedy:
  - se `len(canonical_team) > roster_size + 1` (tolleranza GK): fondi coppie con costo minimo
  - costo merge = `gap_frames + dist_norm - track_length_bonus`
- [ ] **2.2.3** Se `len(canonical_team) < roster_size - 2`: log warning (troppa visibilità persa nel clip).

### 2.3 ReID BoT-SORT (tie-breaker)

- [ ] **2.3.1** Investigare export embedding da Ultralytics BoT-SORT durante `run_pipeline.py` / `player_tracking.py` (feature per detection/track).
- [ ] **2.3.2** Se disponibile: salvare `output/track_embeddings.npz` o parquet:
  - chiavi: `player_id`, `frame`, `emb[dim]`
- [ ] **2.3.3** Per ogni tracklet: `emb_mean = mean(emb)` (solo frame con conf alta).
- [ ] **2.3.4** Nel merge ambiguo (2 candidati entro soglia spaziale): scegli min distanza coseno tra `emb_mean`.
- [ ] **2.3.5** Se ReID non esportabile senza fork pesante: **rimandare** e usare solo 2.1 + 2.2 nella v1.

### 2.4 Merge incrementale al rientro (opzionale v1.1)

- [ ] **2.4.1** Oltre al pass offline su tutto il CSV, funzione `try_link_returning_track(new_track, active_canonicals, last_known_pos)` per simulazione frame-by-frame (utile per debug, non obbligatorio in v1).

**Done when:** su clip con 2+ uscite/rientri, stesso giocatore → stesso `canonical_id` nella maggior parte dei casi; `n_canonical` per team ~10.

**File toccati:** `tactical_roles.py` (`_merge_tracklets`), opz. `run_pipeline.py`.

---

## Fase 3 — Direzione attacco intra-frame (per ranking a coppie)

Il follow-cam invalida `mean(x)` globale; serve asse **avanti/indietro** per frame.

### 3.1 Asse di profondità per frame

- [ ] **3.1.1** Funzione `attack_axis_frame(team_players_xy) -> unit_vector` o segno scalare:
  - **v1 semplice:** usa differenza `mean(x_team0) vs mean(x_team1)` come oggi `_attack_signs`, ma applicata **solo dentro il frame**
  - **v1.1:** asse = verso del movimento medio della palla negli ultimi K frame (se palla tracciata)
- [ ] **3.1.2** Per ogni giocatore nel frame, calcola `depth_i = dot(pos_i - centroid, attack_axis)` (solo co-visibili stessa squadra).

### 3.2 Larghezza per frame

- [ ] **3.2.1** `width_i = y_center` oppure componente ortogonale all'asse attacco.
- [ ] **3.2.2** Normalizzare per frame (z-score o rank) per ridurre effetto zoom del crop.

**Done when:** in un frame con 5 giocatori, l'ordinamento depth è plausibile (portiere/difensori più "indietro").

**File:** nuovo `pair_ranking.py` o sezione in `tactical_roles.py`.

---

## Fase 4 — Ranking globale da coppie (core ruoli follow-cam)

### 4.1 Accumulo relazioni

- [ ] **4.1.1** Per ogni frame `t`, per ogni `team in {0,1}`:
  - prendi canonical visibili `C_t` con `|C_t| >= 2`
  - per ogni coppia `(i, j)` in `C_t`:
    - se `depth_i < depth_j` → voto `i behind j` (+1)
    - else → voto `i ahead j` (+1)
    - idem per `width` (sinistra/destra)
- [ ] **4.1.2** Parametro `--min-coappearances` (es. 5): ignora coppie con pochi voti totali.

### 4.2 Aggregazione in ranking

- [ ] **4.2.1** Costruire matrice pairwise `M_behind[i,j]` = count(voti "i dietro j").
- [ ] **4.2.2** Stima ranking globale:
  - **v1:** winsorized Borda count o greedy topological sort
  - **v1.1:** Bradley-Terry / rank centrality se vuoi più robustezza
- [ ] **4.2.3** Output per team:
  - `depth_rank[canonical_id]` ∈ [1..N]
  - `width_rank[canonical_id]` ∈ [1..N]
  - `rank_confidence[canonical_id]` = min(voti su coppie che lo coinvolgono)

### 4.3 Gestione disconnessione grafo

- [ ] **4.3.1** Se due gruppi di canonical non hanno mai co-occorrenza (grafo disconnesso): rank separati + flag `low_confidence`.
- [ ] **4.3.2** Non forzare ruolo se `rank_confidence < soglia`.

**Done when:** su clip 2 min, i 2–3 difensori più arretrati hanno `depth_rank` basso in modo stabile tra run.

**File:** `pair_ranking.py`.

---

## Fase 5 — Template formazione + assegnazione ruoli

Sostituisce `_assign_roles` basato su quantili in `tactical_roles.py`.

### 5.1 Definizione template

- [ ] **5.1.1** File config minimale, es. `config/formation_9v9_323.yaml`:

  ```yaml
  slots:
    - { id: POR,  depth: 0.05, width: 0.50 }
    - { id: TS,   depth: 0.20, width: 0.15 }
    - { id: DC,   depth: 0.20, width: 0.50 }
    - { id: TD,   depth: 0.20, width: 0.85 }
    - { id: MED_SX, depth: 0.45, width: 0.35 }
    - { id: MED_DX, depth: 0.45, width: 0.65 }
    - { id: AS,   depth: 0.75, width: 0.15 }
    - { id: PC,   depth: 0.75, width: 0.50 }
    - { id: AD,   depth: 0.75, width: 0.85 }
  ```

  (`depth`/`width` normalizzati 0–1, non metri — adatti al ranking)

- [ ] **5.1.2** Parametro CLI `--formation` per altri moduli (3-3-2, ecc.) in futuro.

### 5.2 Costruzione feature per canonical

- [ ] **5.2.1** Da ranking: `feat[c] = (depth_rank[c]/N, width_rank[c]/N)`.
- [ ] **5.2.2** GK separato:
  - mantieni euristica attuale (`gk_gap`) **oppure**
  - slot `POR` = canonical con `depth_rank` minimo + gap di confidenza
  - escludi GK dall'Hungarian outfield

### 5.3 Hungarian assignment

- [ ] **5.3.1** Costo `cost[c, slot] = w_d * |feat_d - slot_d| + w_w * |feat_w - slot_w|`.
- [ ] **5.3.2** `scipy.optimize.linear_sum_assignment` (o implementazione minima se vuoi zero dipendenze).
- [ ] **5.3.3** Slot non assegnati / canonical non assegnati → `role=""` o `role=UNK`.
- [ ] **5.3.4** Soglia massima costo: sopra soglia → non assegnare (evita forzature).

### 5.4 Ruolo stabile nel tempo

- [ ] **5.4.1** Calcolo ruoli **una volta** a fine clip (o fine finestra, es. 60 s).
- [ ] **5.4.2** Propagazione: ogni riga CSV eredita `role` dal `canonical_id` (come ora).
- [ ] **5.4.3** **Non** ricalcolare ruolo sui soli visibili del frame.

**Done when:** con 5–7 giocatori visibili per gran parte del clip, i ruoli assegnati restano stabili e non "rotano" quando esce un difensore.

**File:** `tactical_roles.py` (refactor `_assign_roles` → `assign_roles_template`), `config/formation_9v9_323.yaml`.

---

## Fase 6 — Homography (fase 2 del progetto, opzionale ma consigliata su Veo)

Da fare **dopo** che ranking + template funzionano in pixel/rank.

- [ ] **6.1** Integrare keypoint pitch (modello es. `Adit-jain/Soccana_Keypoint` o roboflow field detection).
- [ ] **6.2** Modulo `pitch_homography.py`:
  - per frame: keypoints → `H` (RANSAC)
  - `pitch_x, pitch_y = apply(H, foot_point)`
  - flag `homography_ok` se abbastanza inliers
- [ ] **6.3** Aggiungere `pitch_x`, `pitch_y` a `unified.csv` (solo se `homography_ok`).
- [ ] **6.4** Usare coordinate pitch per:
  - depth/width nelle coppie (invece di pixel)
  - template in metri (0–105, 0–68)
- [ ] **6.5** Smoothing `H` o posizioni pitch tra frame (EMA) per ridurre jitter.

**Done when:** radar 2D ragionevole su frame con campo visibile; ranking meno sensibile allo zoom.

---

## Fase 7 — Integrazione pipeline e CLI

- [ ] **7.1** Nuovo entrypoint o estensione `tactical_roles.py`:

  ```bash
  python tactical_roles.py \
    --csv output/unified.csv \
    --video data/test_2min.mp4 \
    --output output/unified_roles.csv \
    --presence-output output/presence.csv \
    --formation config/formation_9v9_323.yaml \
    --roster-size 10 \
    --v-max-px-per-sec 600
  ```

- [ ] **7.2** Ordine interno `run()`:
  1. load CSV
  2. merge tracklet (Fase 2)
  3. presence diff (Fase 1)
  4. pair ranking (Fase 4)
  5. template roles (Fase 5)
  6. write outputs + debug

- [ ] **7.3** Aggiornare `README.md` (sezione Tactical roles + colonne nuove).

- [ ] **7.4** `run_pipeline.py`: nessun obbligo di cambiare in v1; homography/ReID in fasi successive.

---

## Fase 8 — Validazione e tuning

- [ ] **8.1** Tabella tuning su `test_2min`:
  - `--max-gap`, `--v-max-px-per-sec`, `--min-coappearances`, soglia costo Hungarian
- [ ] **8.2** Casi limite da testare:
  - rientro dal bordo sinistro/destro
  - sostituzione (nuovo giocatore, stesso numero canonical > 10)
  - portiere fuori inquadratura a lungo
  - pressing con difensori avanzati (non devono diventare attaccanti permanenti)
- [ ] **8.3** Confronto visivo: `debug_roles_N.jpg` con `role + in_frame + canonical`.
- [ ] **8.4** Regression: `test_10s` deve ancora produrre ~10 canonical/squadra.

---

## Ordine di implementazione consigliato

```text
Fase 0 (baseline)
  → Fase 2 (merge migliorato)     # identità prima di tutto
  → Fase 1 (presenza)            # diff che avevi in mente
  → Fase 3 + 4 (coppie + rank)   # cuore follow-cam
  → Fase 5 (template ruoli)      # sostituisce quantili
  → Fase 7 (integrazione)
  → Fase 8 (tuning)
  → Fase 6 (homography)          # quando il resto è stabile
  → Fase 2.3 (ReID export)       # se serve ancora
```

---

## Cosa NON implementare nella v1

- OCR numero maglia
- Attesa frame con "tutti e 9 visibili" come gate
- Ruoli ricalcolati per-frame sul sottoinsieme visibile
- Panoramic (non disponibile su export Veo)
- Dipendenza da sn-gamestate/TrackLab (troppo pesante per Jetson)

---

## Deliverable finali v1

| Artefatto | Contenuto |
|-----------|-----------|
| `unified_roles.csv` | + `in_frame` (o merge con presence) |
| `presence.csv` | stato presenza per frame/canonical |
| `tactical_roles.py` | merge fisico, ranking coppie, template |
| `config/formation_9v9_323.yaml` | slot ruoli |
| Debug JPG | canonical + role + in/out |
