# Corpus Manifest

The corpus consists of approximately 20 papers in the field of HVAC engineering. The original PDFs are **not distributed with this repository** (copyright reasons).
The table below lists all corpus files with their metadata. To reproduce the experiments, obtain the papers via the DOI/publisher pages, place them in the `corpus/` directory, and run the ingestion scripts under `ingest/` to rebuild the indexes.

> Note: The information below was compiled from the file names. Fields marked `[TBD]` must be completed from the title page of each PDF
> (full title, authors, year, journal/conference, DOI); file names truncated during export are marked with `…`.

| # | File name | Lang. | Size | Full paper title | Authors | Year | Journal / Conference | DOI / Link |
|---|-----------|------|------|------------------|---------|------|----------------------|------------|
| 1 | An_Explainable_Transformer-based_Mixture_of_Expert….pdf | EN | 25.3 MB | An Explainable Transformer-based Mixture of Expert … [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 2 | Analytische und numerische Untersuchung der Geschwindigkeitsprofile an perforierten Doppelböden in Reinräumen mit Verdrängungsströmung.pdf | DE | 8.50 MB | Analytische und numerische Untersuchung der Geschwindigkeitsprofile an perforierten Doppelböden in Reinräumen mit Verdrängungsströmung | [TBD] | [TBD] | [TBD] | [TBD] |
| 3 | Automated building energy modeling for energy….pdf | EN | 10.4 MB | Automated building energy modeling for energy … [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 4 | Economic viability of low- and ultra-low-temperature district heating under….pdf | EN | 5.69 MB | Economic viability of low- and ultra-low-temperature district heating under … [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 5 | Emission rate and particle size of bioaerosols during breathing, speaking and….pdf | EN | 320 KB | Emission rate and particle size of bioaerosols during breathing, speaking and … [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 6 | Energetic comparison of the inactivation of airborne corona viruses using….pdf | EN | 2.14 MB | Energetic comparison of the inactivation of airborne corona viruses using … [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 7 | Experimental analysis of airborne contaminant distribution in an operating….pdf | EN | 10.4 MB | Experimental analysis of airborne contaminant distribution in an operating … [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 8 | Experimentelle Untersuchung der Leckage und Abscheideleistung von typischen….pdf | DE | 555 KB | Experimentelle Untersuchung der Leckage und Abscheideleistung von typischen … [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 9 | Explainable AI for cooling load prediction and heat flow estimation.pdf | EN | 2.08 MB | Explainable AI for cooling load prediction and heat flow estimation | [TBD] | [TBD] | [TBD] | [TBD] |
| 10 | Explainable multi-step heating load forecasting Using SHAP values and….pdf | EN | 1.92 MB | Explainable multi-step heating load forecasting using SHAP values and … [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 11 | hartmann_kriegel_risikobewertung_alltagssituationen_v2.pdf | DE | 942 KB | Risikobewertung Alltagssituationen (Hartmann / Kriegel) [full official title TBD] | Hartmann; Kriegel | [TBD] | [TBD] | [TBD] |
| 12 | HeatShift-HP A Techno-Economic framework for assessing heat pump load….pdf | EN | 3.53 MB | HeatShift-HP: A Techno-Economic Framework for Assessing Heat Pump Load … [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 13 | Improving heat demand forecasting with feature reduction in an….pdf | EN | 1.79 MB | Improving heat demand forecasting with feature reduction in an … [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 14 | Imputing the long-term missing heating load data using a generative network.pdf | EN | 2.64 MB | Imputing the long-term missing heating load data using a generative network | [TBD] | [TBD] | [TBD] | [TBD] |
| 15 | Influence of occupancy data on heating and….pdf | EN | 1.53 MB | Influence of occupancy data on heating and … [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 16 | Nahfeldexposition respiratorischer Partikel anhand statistischer Auswer….pdf | DE | 695 KB | Nahfeldexposition respiratorischer Partikel anhand statistischer Auswer… [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 17 | Operational regime dependence of model performance and feature….pdf | EN | 14.6 MB | Operational regime dependence of model performance and feature … [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 18 | Relationship betweenfeature importanceandbuildingcharacteristics for….pdf | EN | 1.85 MB | Relationship between feature importance and building characteristics for … [TBD] (original file name missing spaces) | [TBD] | [TBD] | [TBD] | [TBD] |
| 19 | Schulen_inhalierte_Dosis_Aufenthaltszeit.pdf | DE | 1.32 MB | Schulen: inhalierte Dosis & Aufenthaltszeit [full official title TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 20 | TCN-BiLSTM-CEAninterdisciplinaryapproachformissingenergydata.pdf | EN | 2.83 MB | TCN-BiLSTM-CEA: An Interdisciplinary Approach for Missing Energy Data [full official title TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

## Topic Distribution (quick overview of corpus coverage)

- **Building load / energy forecasting & machine learning**: #1, #9, #10, #13, #15, #17, #18, #20
- **Missing energy data imputation**: #14, #20
- **District heating & heat pump economics**: #4, #12
- **Indoor air quality / aerosols & ventilation**: #2, #5, #6, #7, #8, #11, #16, #19
- **Automated building energy modeling**: #3

## Metadata Specification (aligned with the `ingest/` pipeline)

Each paper is registered with the following fields at ingestion time; they are written into every chunk's metadata (citation cards and provenance display depend on them):

```json
{
  "doc_id": "stable ID (first 8 chars of file-name sha1)",
  "title": "full paper title",
  "authors": ["..."],
  "year": 2024,
  "venue": "journal or conference name",
  "doi": "10.xxxx/xxxxx",
  "language": "en | de",
  "source_file": "corpus/original_file_name.pdf"
}
```

Recommendation: once completed, this manifest serves as the single source of truth for the metadata; a script can generate `corpus/metadata.jsonl` from this table.
