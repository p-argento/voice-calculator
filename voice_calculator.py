"""
Calcolatrice vocale minimale: STT -> agente LangGraph (Gemma) -> TTS.

Flusso del grafo (vedi FLOW.md per lo schema completo):

    START -> stt -> llm_call <-> tool_node
                       |
                       v
                      tts -> END

Uso:
    uv run --env-file .env python voice_calculator.py              # registra dal microfono
    uv run --env-file .env python voice_calculator.py domanda.mp3  # usa un file audio
"""

# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

import operator
import subprocess
import sys
import wave
from typing import Annotated, Literal

import sounddevice as sd
from dotenv import load_dotenv
from typing_extensions import TypedDict

# Il client ElevenLabs legge ELEVENLABS_API_KEY dall'ambiente.
from elevenlabs import ElevenLabs
from langchain.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

# Carica .env (GOOGLE_API_KEY, ELEVENLABS_API_KEY) se non è già nell'ambiente.
load_dotenv()


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

# Voce ElevenLabs (George). Puoi sceglierne un'altra dalla dashboard.
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"

# Registrazione dal microfono: durata in secondi e frequenza di campionamento.
RECORD_SECONDS = 5
SAMPLE_RATE = 16000  # 16 kHz è sufficiente per il parlato
QUESTION_WAV = "domanda.wav"

# Client ElevenLabs: usato sia per STT sia per TTS.
eleven = ElevenLabs()

# Modello LLM. Stesso modello usato in calculator.py.
model = ChatGoogleGenerativeAI(model="gemma-4-26b-a4b-it", temperature=0)


# ---------------------------------------------------------------------------
# Step 1: tool matematici
# ---------------------------------------------------------------------------
# La docstring è importante: LangChain la passa al modello come descrizione
# del tool, e il modello decide quando chiamarlo in base a quella.


@tool
def add(a: int, b: int) -> int:
    """Somma `a` e `b`.

    Args:
        a: primo numero intero
        b: secondo numero intero
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Moltiplica `a` per `b`.

    Args:
        a: primo numero intero
        b: secondo numero intero
    """
    return a * b


@tool
def divide(a: int, b: int) -> str:
    """Divide `a` per `b`.

    Args:
        a: numeratore
        b: denominatore
    """
    # Restituiamo una stringa invece di sollevare un'eccezione:
    # il modello riceve il messaggio e può rispondere all'utente.
    if b == 0:
        return "Errore: divisione per zero"
    return str(a / b)


# Colleghiamo il modello ai tool: il modello può ora "chiedere" di eseguirne uno.
tools = [add, multiply, divide]
tools_by_name = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)


# ---------------------------------------------------------------------------
# Step 2: stato del grafo
# ---------------------------------------------------------------------------
# Lo stato è la "memoria" condivisa tra i nodi. Ogni nodo legge lo stato
# e restituisce solo le chiavi che vuole aggiornare.


class CalcState(TypedDict):
    audio_in: str                                   # percorso dell'audio della domanda
    question: str                                   # testo trascritto dall'STT
    messages: Annotated[list[AnyMessage], operator.add]  # cronologia dei messaggi
    llm_calls: int                                  # quante volte ha chiamato il modello
    answer: str                                     # risposta testuale finale
    audio_out: str                                  # percorso dell'audio della risposta


SYSTEM_PROMPT = (
    "Sei un assistente che esegue calcoli aritmetici. "
    "Usa i tool per fare i conti e rispondi in italiano, in una frase breve "
    "e adatta a essere letta ad alta voce."
)


# ---------------------------------------------------------------------------
# Step 3: nodo STT (speech-to-text)
# ---------------------------------------------------------------------------


def stt_node(state: CalcState):
    """Trascrive l'audio della domanda in testo con ElevenLabs Scribe."""
    with open(state["audio_in"], "rb") as audio_file:
        result = eleven.speech_to_text.convert(file=audio_file, model_id="scribe_v2")

    print(f"[STT] {result.text}")

    # Il testo diventa il primo messaggio umano della conversazione.
    return {
        "question": result.text,
        "messages": [HumanMessage(content=result.text)],
    }


# ---------------------------------------------------------------------------
# Step 4: nodo LLM (il "cervello": decide se usare un tool)
# ---------------------------------------------------------------------------


def llm_call(state: CalcState):
    """Chiede al modello la prossima mossa: rispondere o chiamare un tool."""
    response = model_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Step 5: nodo tool (esegue le tool call richieste dal modello)
# ---------------------------------------------------------------------------


def tool_node(state: CalcState):
    """Esegue ogni tool call dell'ultimo messaggio AI e restituisce i risultati."""
    results = []
    for call in state["messages"][-1].tool_calls:
        observation = tools_by_name[call["name"]].invoke(call["args"])
        # ToolMessage collega il risultato alla tool call tramite tool_call_id.
        results.append(ToolMessage(content=str(observation), tool_call_id=call["id"]))
    return {"messages": results}


# ---------------------------------------------------------------------------
# Step 6: nodo TTS (text-to-speech)
# ---------------------------------------------------------------------------


def tts_node(state: CalcState):
    """Converte la risposta finale del modello in un file audio MP3."""
    answer = state["messages"][-1].content
    if isinstance(answer, list):
        # Alcuni modelli restituiscono il contenuto come lista di blocchi.
        answer = " ".join(
            block["text"] for block in answer if block.get("type") == "text"
        )

    audio_path = "risposta.mp3"
    # convert() restituisce l'audio a pezzi (stream): li scriviamo nel file.
    audio = eleven.text_to_speech.convert(
        text=answer,
        voice_id=VOICE_ID,
        model_id="eleven_v4",
    )
    with open(audio_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)

    return {"answer": answer, "audio_out": audio_path}


# ---------------------------------------------------------------------------
# Step 7: logica di routing (dopo llm_call)
# ---------------------------------------------------------------------------


def should_continue(state: CalcState) -> Literal["tool_node", "tts"]:
    """Se il modello ha chiesto un tool, va al tool_node; altrimenti passa al TTS."""
    if state["messages"][-1].tool_calls:
        return "tool_node"
    return "tts"


# ---------------------------------------------------------------------------
# Step 8: costruzione e compilazione del grafo
# ---------------------------------------------------------------------------

builder = StateGraph(CalcState)

builder.add_node("stt", stt_node)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)
builder.add_node("tts", tts_node)

builder.add_edge(START, "stt")
builder.add_edge("stt", "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", "tts"])
builder.add_edge("tool_node", "llm_call")  # dopo il tool, il modello rielabora
builder.add_edge("tts", END)

graph = builder.compile()


# ---------------------------------------------------------------------------
# Step 9: esecuzione
# ---------------------------------------------------------------------------


def record_question(path: str, seconds: int = RECORD_SECONDS) -> str:
    """Registra dal microfono per `seconds` secondi e salva un file WAV."""
    print(f"Parla ora, registro per {seconds} secondi...")

    # sd.rec restituisce un array numpy di campioni int16 (un canale, mono).
    samples = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()  # aspetta la fine della registrazione

    # Il WAV è un formato semplice: intestazione + campioni grezzi.
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)       # mono
        wav_file.setsampwidth(2)       # 2 byte = int16
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(samples.tobytes())

    print(f"Registrazione salvata in {path}")
    return path


def main():
    # Senza argomenti registra dal microfono; con un file usa quello.
    if len(sys.argv) == 1:
        audio_in = record_question(QUESTION_WAV)
    elif len(sys.argv) == 2:
        audio_in = sys.argv[1]
    else:
        print("Uso: python voice_calculator.py [domanda.mp3]")
        sys.exit(1)

    result = graph.invoke({"audio_in": audio_in, "llm_calls": 0, "messages": []})
    print(f"[Risposta] {result['answer']}")

    # Riproduzione audio solo su macOS (comando di sistema `afplay`).
    if sys.platform == "darwin":
        subprocess.run(["afplay", result["audio_out"]])


if __name__ == "__main__":
    main()
