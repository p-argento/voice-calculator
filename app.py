"""
Interfaccia web per la calcolatrice vocale.

Riusa il grafo, il modello e i tool di voice_calculator.py: qui c'è solo la UI.

Avvio:
    uv run --env-file .env python app.py
Poi apri l'URL che compare nel terminale (di solito http://127.0.0.1:7860).
"""

import io

import gradio as gr
from langchain.messages import AIMessage, ToolMessage
from PIL import Image

from voice_calculator import VOICE_ID, graph, model, tools


def run_agent(audio_path):
    """Esegue il grafo su un file audio e raccoglie la traccia dei tool usati."""
    result = graph.invoke({"audio_in": audio_path, "llm_calls": 0, "messages": []})

    # Abbiniamo ogni tool call (nel messaggio AI) al suo risultato (ToolMessage),
    # usando tool_call_id come chiave.
    calls = {}
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                calls[call["id"]] = {"name": call["name"], "args": call["args"], "result": None}
        elif isinstance(message, ToolMessage):
            if message.tool_call_id in calls:
                calls[message.tool_call_id]["result"] = message.content

    return {
        "transcript": result["question"],
        "answer": result["answer"],
        "audio_out": result["audio_out"],
        "tool_log": list(calls.values()),
    }


def render_tools(tool_log):
    """Riquadro dei tool: elenco di quelli disponibili, con il dettaglio di quelli usati."""
    used_names = {entry["name"] for entry in tool_log}
    lines = ["### Tool disponibili\n"]
    for t in tools:
        # La prima riga della docstring è la descrizione che il modello legge.
        summary = t.description.split("\n")[0]
        mark = " ✅ usato" if t.name in used_names else ""
        lines.append(f"- **`{t.name}`**: {summary}{mark}")

    lines.append("\n### Chiamate nell'ultima domanda\n")
    if not tool_log:
        lines.append("Nessuna chiamata ancora. Fai una domanda.")
    for entry in tool_log:
        lines.append(f"- `{entry['name']}` con `{entry['args']}` → `{entry['result']}`")
    return "\n".join(lines)


def render_model():
    """Riquadro del modello LLM e della voce TTS."""
    return (
        f"- **Classe**: `{type(model).__name__}`\n"
        f"- **Modello**: `{model.model}`\n"
        f"- **Temperatura**: `{model.temperature}`\n"
        f"- **Tool collegati**: {len(tools)}\n\n"
        f"**Voce TTS**: ElevenLabs `eleven_v4`, voce `{VOICE_ID}`"
    )


def render_graph():
    """Riquadro del grafo: PNG del grafo compilato, generato con Mermaid."""
    try:
        # draw_mermaid_png() usa il servizio mermaid.ink, quindi richiede internet.
        png = graph.get_graph().draw_mermaid_png()
        return Image.open(io.BytesIO(png))
    except Exception as e:
        print(f"[grafo] impossibile generare il PNG: {e}")
        return None


def ask(audio_path, state):
    """Eseguita quando la registrazione termina: chiama l'agente e aggiorna la UI."""
    if not audio_path:
        return "", "Registra una domanda prima.", None, state, render_tools(state["tool_log"])

    out = run_agent(audio_path)
    state = {"tool_log": out["tool_log"]}
    return (
        out["transcript"],
        out["answer"],
        out["audio_out"],
        state,
        render_tools(state["tool_log"]),
    )


# ---------------------------------------------------------------------------
# Layout dell'interfaccia
# ---------------------------------------------------------------------------

# Nasconde i controlli nativi del microfono: li usano i nostri pulsanti via JavaScript.
HIDE_NATIVE_MIC_CSS = """
.record-button, .stop-button {
    display: none !important;
}
"""

with gr.Blocks(title="Calcolatrice vocale") as demo:
    gr.Markdown("# Calcolatrice vocale\nSTT → Gemma + LangGraph → TTS.")

    # Stato della sessione: traccia dei tool dell'ultima domanda.
    state = gr.State({"tool_log": []})

    with gr.Row():
        with gr.Column():
            # Solo microfono: niente upload di file.
            mic = gr.Audio(sources=["microphone"], type="filepath", label="Domanda")
            with gr.Row():
                start_btn = gr.Button("Avvia registrazione", variant="primary")
                stop_btn = gr.Button("Interrompi e invia", variant="stop")
        with gr.Column():
            transcript = gr.Textbox(label="Trascrizione", interactive=False)
            answer = gr.Textbox(label="Risposta", interactive=False)
            answer_audio = gr.Audio(label="Risposta audio", type="filepath", autoplay=True)

    gr.Markdown("---")

    # Tre riquadri sempre visibili, senza dropdown.
    with gr.Row(equal_height=False):
        with gr.Column():
            with gr.Group():
                tools_box = gr.Markdown(render_tools([]))
        with gr.Column():
            with gr.Group():
                gr.Markdown("### Modello LLM")
                gr.Markdown(render_model())
        with gr.Column():
            with gr.Group():
                gr.Markdown("### Grafo LangGraph")
                gr.Image(value=render_graph(), show_label=False, interactive=False)

    # Avvia e Interrompi cliccano i pulsanti nativi del microfono di Gradio.
    # Impostare `recording` dal Python non basta: il frontend non avvia la registrazione.
    start_btn.click(
        None,
        js="() => document.querySelector('.record-button')?.click()",
    )
    stop_btn.click(
        None,
        js="() => document.querySelector('.stop-button')?.click()",
    )
    # ...e a registrazione chiusa e file pronto parte la richiesta all'agente.
    mic.stop_recording(
        ask,
        inputs=[mic, state],
        outputs=[transcript, answer, answer_audio, state, tools_box],
    )


if __name__ == "__main__":
    demo.launch(css=HIDE_NATIVE_MIC_CSS)
