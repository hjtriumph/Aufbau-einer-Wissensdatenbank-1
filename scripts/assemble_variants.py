"""Build the English/German question variants (5 base questions × 2 languages).

Variants share the gold evidence with their Chinese base question and are
reported separately by language; they are not part of the 30 × 16 main run.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"


def load_questions() -> dict[str, dict]:
    return {
        item["id"]: item
        for item in (
            json.loads(line)
            for line in (CORPUS / "questions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


VARIANTS = [
    {
        "base": "q005",
        "language": "en",
        "question": "In the Cambridge office building case study, by how much did Data2BEM reduce total modeling time relative to professional practice, and what were the before and after durations?",
        "reference_answer": "Data2BEM reduced total modeling time by over 90%, from 8–32 hours in professional practice to about 48 minutes, with minimal human input.",
        "key_points": ["over 90% reduction", "from 8–32 hours", "about 48 minutes", "minimal human input"],
    },
    {
        "base": "q005",
        "language": "de",
        "question": "Um wie viel reduzierte Data2BEM im Cambridge-Bürogebäudefall die gesamte Modellierungszeit gegenüber der professionellen Praxis, und wie lang waren die Zeiten vorher und nachher?",
        "reference_answer": "Data2BEM reduzierte die gesamte Modellierungszeit um über 90 %, von 8–32 Stunden in der professionellen Praxis auf etwa 48 Minuten, bei minimalem menschlichem Eingriff.",
        "key_points": ["über 90 % Reduktion", "von 8–32 Stunden", "etwa 48 Minuten", "minimaler Eingriff"],
    },
    {
        "base": "q007",
        "language": "en",
        "question": "In the cooling load study, local SHAP values cannot reproduce the negative Qtrans+vent heat flow at night. What does the paper attribute this to, and which other heat flow is reported to have the same problem?",
        "reference_answer": "It is attributed to the storage and release of energy in the thermal mass (QTM): the selected input features (Tout, Isol,so, occupancy) do not explicitly include thermal-mass temperature or storage state, so the negative night-time heat flow is not reflected in the SHAP values. Internal heat gains (QIG) show the same problem: they are poorly represented by occupancy SHAP values, with R² of about 0.28 (TS) and 0.17 (KS).",
        "key_points": ["thermal mass storage and release", "features lack thermal-mass state", "no negative night Qtrans+vent", "QIG poorly represented by occupancy"],
    },
    {
        "base": "q007",
        "language": "de",
        "question": "In der Kühllaststudie können lokale SHAP-Werte den negativen Wärmestrom Qtrans+vent nachts nicht abbilden. Worauf führt die Studie dies zurück, und welcher weitere Wärmestrom zeigt dasselbe Problem?",
        "reference_answer": "Die Studie führt dies auf die Speicherung und Freisetzung von Energie in der thermischen Masse (QTM) zurück: Die ausgewählten Eingangsmerkmale (Tout, Isol,so, occupancy) enthalten keine Information über Temperatur oder Speicherzustand der thermischen Masse, daher erscheint der negative nächtliche Wärmestrom nicht in den SHAP-Werten. Dasselbe Problem besteht bei den internen Wärmegewinnen (QIG), die durch die occupancy-SHAP-Werte nur schlecht abgebildet werden (R² etwa 0,28 bei TS und 0,17 bei KS).",
        "key_points": ["Speicherung in thermischer Masse", "Merkmale ohne Massezustand", "kein negativer Nachtstrom", "QIG schlecht durch occupancy abgebildet"],
    },
    {
        "base": "q010",
        "language": "en",
        "question": "Which input features does the explainable transformer-based Mixture-of-Experts heat load forecasting framework use, and which load regime does each of the three experts correspond to?",
        "reference_answer": "The inputs are outdoor temperature, the historical heat load time series, and four temporal encodings: sine and cosine of month and of hour. K-means clusters buildings into low, medium, and high load regimes based on historical consumption; each regime is modelled by one lightweight Informer encoder expert.",
        "key_points": ["outdoor temperature", "historical heat load", "sin/cos month and hour", "low, medium, high regimes", "one Informer expert per regime"],
    },
    {
        "base": "q010",
        "language": "de",
        "question": "Welche Eingangsmerkmale verwendet das erklärbare transformerbasierte Mixture-of-Experts-Framework zur Wärmelastprognose, und welchem Lastregime entspricht jeder der drei Experten?",
        "reference_answer": "Die Eingänge sind die Außentemperatur, die historische Wärmelastzeitreihe sowie vier zeitliche Kodierungen: Sinus und Kosinus von Monat und Stunde. K-means clustert Gebäude anhand des historischen Verbrauchs in niedrige, mittlere und hohe Lastregime; jedes Regime wird von einem leichten Informer-Encoder-Experten modelliert.",
        "key_points": ["Außentemperatur", "historische Wärmelast", "Sinus/Kosinus Monat und Stunde", "niedriges, mittleres, hohes Regime", "ein Informer-Experte je Regime"],
    },
    {
        "base": "q013",
        "language": "en",
        "question": "Combining HeatShift-HP and Ni et al.'s study on the economics of LTDH/ULTDH: through which mechanisms does lowering the supply temperature affect cost in a single-building heat pump system versus a district heating network, and where do the two papers agree and differ on whether lower temperature is always more economical?",
        "reference_answer": "For a single building, HeatShift-HP finds that lowering the nominal supply temperature raises the heat pump COP, reducing electricity use and paid heat cost: 40→35 °C lowers median paid specific heat cost by about 3.7% in Berlin and 4.4% in Chicago, and nominal system temperature is the dominant driver; storage and price-based control add only limited gains. For district heating, Ni et al. find that lowering network temperature reduces distribution losses and improves central heat pump efficiency, but these benefits can be partly or fully offset by the investment and operating cost of building-side booster heat pumps. LTDH remains competitive in most scenarios, while ULTDH is advantageous only when the high-temperature demand share falls below about 30%; at very low linear heat density (below about 1.28 MWh/(m·yr)) decentralized heat pumps are preferable. Both agree that lower temperature improves efficiency and reduces losses; they differ in that the single-building benefit is direct, whereas the district-heating benefit is conditional on HT demand share, heat density, and booster costs.",
        "key_points": ["single building: higher COP", "3.7% Berlin, 4.4% Chicago", "network: lower distribution losses", "booster heat pump cost offsets", "ULTDH only if HT share below 30%"],
    },
    {
        "base": "q013",
        "language": "de",
        "question": "Mit Blick auf HeatShift-HP und die Studie von Ni et al. zur Wirtschaftlichkeit von LTDH/ULTDH: Über welche Mechanismen beeinflusst eine niedrigere Vorlauftemperatur die Kosten bei einer Wärmepumpe im Einzelgebäude bzw. in einem Fernwärmenetz, und worin stimmen beide Arbeiten überein bzw. unterscheiden sie sich bei der Aussage „niedrigere Temperatur ist immer wirtschaftlicher“?",
        "reference_answer": "Für das Einzelgebäude zeigt HeatShift-HP: Eine niedrigere nominelle Vorlauftemperatur erhöht die COP der Wärmepumpe und senkt den Stromverbrauch und die gezahlten Wärmekosten; 40→35 °C senken die medianen spezifischen Wärmekosten um etwa 3,7 % in Berlin und 4,4 % in Chicago, wobei die nominelle Systemtemperatur der dominierende Faktor ist; Speicher und preisbasierte Regelung bringen nur begrenzte Zusatzgewinne. Für Fernwärme zeigt Ni et al.: Eine niedrigere Netztemperatur verringert Verteilverluste und verbessert die Effizienz zentraler Wärmepumpen, doch diese Vorteile können durch Investitions- und Betriebskosten gebäudeseitiger Nachheiz-Wärmepumpen teilweise oder vollständig aufgezehrt werden. LTDH bleibt in den meisten Szenarien wettbewerbsfähig, ULTDH ist nur vorteilhaft, wenn der Anteil des Hochtemperaturbedarfs unter etwa 30 % fällt; bei sehr geringer linearer Wärmedichte (unter etwa 1,28 MWh/(m·a)) sind dezentrale Wärmepumpen günstiger. Beide stimmen überein, dass niedrigere Temperaturen die Effizienz erhöhen und Verluste senken; sie unterscheiden sich darin, dass der Nutzen im Einzelgebäude direkt ist, während er im Fernwärmenetz vom HT-Anteil, der Wärmedichte und den Kosten der Nachheizung abhängt.",
        "key_points": ["Einzelgebäude: höhere COP", "3,7 % Berlin, 4,4 % Chicago", "Netz: geringere Verteilverluste", "Kosten der Nachheizung", "ULTDH nur bei HT-Anteil unter 30 %"],
    },
    {
        "base": "q021",
        "language": "en",
        "question": "How does the UVC study derive the cumulative UV dose over 10 minutes from the single-pass dose? Give the calculation chain, key parameters, and results, and explain how the result should be interpreted.",
        "reference_answer": "First the single-pass dose Dp = I · tr: at a chamber depth of 0.13 m and 200 m³/h the mean flow velocity is about 0.26 m/s, giving a mean residence time in the irradiation zone of about 0.5 s; with a mean irradiance of 81.5 W/m² the single-pass dose is about 40.75 J/m² (CFD median 42 J/m²). For recirculation, at an air change rate of 28.5 h⁻¹ the room air passes the device about N = 4.73 times in 10 min, so the cumulative dose is Dc = Dp · N ≈ 192 J/m². The authors stress that this is a representative estimate subject to measurement and model uncertainty in irradiance, residence-time distribution, and airflow, not an absolute inactivation threshold.",
        "key_points": ["Dp = intensity × residence time", "residence time about 0.5 s", "single-pass dose 40.75 J/m²", "about 4.73 passes in 10 min", "cumulative dose about 192 J/m²"],
    },
    {
        "base": "q021",
        "language": "de",
        "question": "Wie leitet die UVC-Studie die kumulative UV-Dosis über 10 Minuten aus der Einzeldurchgangsdosis ab? Geben Sie die Berechnungskette, die wichtigsten Parameter und Ergebnisse an und erläutern Sie, wie das Ergebnis zu interpretieren ist.",
        "reference_answer": "Zunächst die Einzeldurchgangsdosis Dp = I · tr: Bei einer Kammertiefe von 0,13 m und 200 m³/h beträgt die mittlere Strömungsgeschwindigkeit etwa 0,26 m/s, woraus eine mittlere Verweilzeit im Bestrahlungsbereich von etwa 0,5 s folgt; bei einer mittleren Bestrahlungsstärke von 81,5 W/m² ergibt sich eine Einzeldosis von etwa 40,75 J/m² (CFD-Median 42 J/m²). Für die Rezirkulation passiert die Raumluft das Gerät bei einer Luftwechselrate von 28,5 h⁻¹ in 10 Minuten etwa N = 4,73-mal, sodass die kumulative Dosis Dc = Dp · N ≈ 192 J/m² beträgt. Die Autoren betonen, dass dies eine repräsentative Schätzung ist, die von Mess- und Modellunsicherheiten bei Bestrahlungsstärke, Verweilzeitverteilung und Luftströmung abhängt, und kein absoluter Inaktivierungsschwellenwert.",
        "key_points": ["Dp = Intensität × Verweilzeit", "Verweilzeit etwa 0,5 s", "Einzeldosis 40,75 J/m²", "etwa 4,73 Durchgänge in 10 min", "kumulative Dosis etwa 192 J/m²"],
    },
]


def main() -> None:
    questions = load_questions()
    out = []
    counters: dict[str, int] = {}
    for item in VARIANTS:
        base = questions[item["base"]]
        counters[item["language"]] = counters.get(item["language"], 0) + 1
        out.append(
            {
                "id": f"v-{item['language']}-{counters[item['language']]:02d}",
                "base_id": base["id"],
                "question": item["question"],
                "type": base["type"],
                "reference_answer": item["reference_answer"],
                "key_points": item["key_points"],
                "gold_evidence": base["gold_evidence"],
                "answerable": True,
                "language": item["language"],
                "notes": f"variant of {base['id']}",
            }
        )
    path = CORPUS / "variants.jsonl"
    path.write_text(
        "\n".join(json.dumps(v, ensure_ascii=False) for v in out) + "\n", encoding="utf-8"
    )
    print(f"[variants] {len(out)} -> {path}")


if __name__ == "__main__":
    main()
