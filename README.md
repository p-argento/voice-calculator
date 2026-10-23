# Voice Calculator

Calcolatrice vocale: parli nel microfono, un agente LangGraph con Gemma fa il calcolo e la risposta viene letta ad alta voce con ElevenLabs.

Flusso: **STT** (ElevenLabs Scribe) → **agente** (LangGraph + Gemma con tool `add`, `multiply`, `divide`) → **TTS** (ElevenLabs `eleven_v4`).

Lo schema completo del grafo è in [FLOW.md](FLOW.md).

## Requisiti

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- Chiave API di Google AI Studio (Gemma)
- Chiave API di ElevenLabs

## Configurazione

Crea un file `.env` nella cartella del progetto:

```
GOOGLE_API_KEY=la-tua-chiave-google
ELEVENLABS_API_KEY=la-tua-chiave-elevenlabs
```

Il file `.env` è escluso da git e non va mai committato.

## Avvio

```bash
uv run --env-file .env python app.py
```

Apri `http://127.0.0.1:7860`, premi **Avvia registrazione**, fai la domanda (per esempio "quanto fa tre più quattro") e premi **Interrompi e invia**.

Nella pagina trovi tre riquadri: i tool disponibili con quelli usati nell'ultima domanda, il modello LLM e il grafo LangGraph.

## Versione da terminale

`voice_calculator.py` esegue lo stesso flusso senza interfaccia:

```bash
uv run --env-file .env python voice_calculator.py              # registra dal microfono per 5 secondi
uv run --env-file .env python voice_calculator.py domanda.mp3  # usa un file audio
```

`calculator.py` è la versione solo testuale dell'agente.
