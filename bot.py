"""
Bot de Telegram con RAG (Retrieval-Augmented Generation)
Version mejorada — Groq API + memoria de conversacion + prompt especializado
Temario de metodologia de investigacion para alumnos universitarios de ciencias de la salud.
"""

import os
import time
import logging
import threading
import hashlib
import requests as req
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from collections import defaultdict

from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
from sentence_transformers import SentenceTransformer

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─── Variables de entorno ─────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY")
PORT           = int(os.environ.get("PORT", 8080))

# Modelo a usar
GROQ_MODEL = "llama-3.1-8b-instant"

# ─── Carpeta de PDFs ──────────────────────────────────────────────────────────
PDF_FOLDER = "manuales"

# ─── Memoria de conversacion (ultimos 4 mensajes por usuario) ─────────────────
historial = defaultdict(list)
MAX_HISTORIAL = 4

# ─── Clientes ─────────────────────────────────────────────────────────────────
groq_client   = Groq(api_key=GROQ_API_KEY)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

class SemanticEmbeddingFunction:
    """Embedding semantico ligero — entiende significado, no solo palabras exactas."""
    def __init__(self):
        self.model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

    def __call__(self, input):
        embeddings = self.model.encode(input, normalize_embeddings=True)
        return embeddings.tolist()

embedding_fn = SemanticEmbeddingFunction()

# Intentar usar la coleccion con mas fragmentos, si no existe usar la actual
def obtener_coleccion():
    colecciones_preferidas = ["manuales"]
    mejor = None
    mejor_count = 0
    for nombre in colecciones_preferidas:
        try:
            col = chroma_client.get_collection(name=nombre, embedding_function=embedding_fn)
            count = col.count()
            logger.info(f"Coleccion '{nombre}' encontrada con {count} fragmentos.")
            if count > mejor_count:
                mejor = col
                mejor_count = count
        except Exception:
            pass
    if mejor:
        logger.info(f"Usando coleccion con {mejor_count} fragmentos.")
        return mejor
    # Si no existe ninguna, crear nueva
    return chroma_client.get_or_create_collection(
        name="manuales_v5",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

collection = chroma_client.get_or_create_collection(
    name="manuales_v5",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)


# ─── Servidor web minimo (necesario para Render Web Service) ──────────────────
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot activo")

    def log_message(self, format, *args):
        pass

def iniciar_servidor_web():
    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    logger.info(f"Servidor web arrancado en puerto {PORT}")
    server.serve_forever()


# ─── Indexar PDFs ─────────────────────────────────────────────────────────────
def indexar_pdfs():
    pdf_folder = Path(PDF_FOLDER)
    if not pdf_folder.exists():
        pdf_folder.mkdir()
        logger.warning(f"Carpeta '{PDF_FOLDER}' creada. Anade tus PDFs y reinicia.")
        return

    pdfs = list(pdf_folder.glob("*.pdf"))
    if not pdfs:
        logger.warning(f"No hay PDFs en '{PDF_FOLDER}'.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60)

    for pdf_path in pdfs:
        pdf_name = pdf_path.stem
        existing = collection.get(where={"source": pdf_name})
        if existing["ids"]:
            logger.info(f"'{pdf_name}' ya indexado.")
            continue

        logger.info(f"Indexando '{pdf_name}'...")
        loader    = PyPDFLoader(str(pdf_path))
        pages     = loader.load()
        fragments = splitter.split_documents(pages)

        ids       = [f"{pdf_name}__{i}" for i in range(len(fragments))]
        texts     = [f.page_content for f in fragments]
        metadatas = [{"source": pdf_name, "page": f.metadata.get("page", 0)} for f in fragments]

        for start in range(0, len(texts), 100):
            collection.add(
                ids=ids[start:start+100],
                documents=texts[start:start+100],
                metadatas=metadatas[start:start+100]
            )
        logger.info(f"'{pdf_name}' indexado: {len(fragments)} fragmentos.")

    logger.info(f"Indexacion completada. Total fragmentos: {collection.count()}")


# ─── Buscar contexto ──────────────────────────────────────────────────────────
def buscar_contexto(pregunta: str, n_resultados: int = 5) -> str:
    resultados = collection.query(query_texts=[pregunta], n_results=n_resultados)
    if not resultados["documents"] or not resultados["documents"][0]:
        return ""
    fragmentos = []
    for doc, meta in zip(resultados["documents"][0], resultados["metadatas"][0]):
        fuente = meta.get("source", "desconocido")
        pagina = meta.get("page", "?")
        fragmentos.append(f"[{fuente} - pag. {pagina}]\n{doc}")
    return "\n\n---\n\n".join(fragmentos)


# ─── Generar respuesta con Groq ───────────────────────────────────────────────
def generar_respuesta(user_id: int, pregunta: str, contexto: str, reintentos: int = 2) -> str:
    if not contexto:
        return (
            "No he encontrado informacion sobre eso en el temario. "
            "Intenta reformular la pregunta usando terminos del temario, "
            "o consulta directamente el material de la asignatura."
        )

    system_prompt = """Eres un asistente academico especializado en metodologia de investigacion para estudiantes universitarios de ciencias de la salud.

Tu funcion es responder preguntas sobre el temario de la asignatura basandote EXCLUSIVAMENTE en los fragmentos del temario proporcionados.

INSTRUCCIONES:
1. Usa SOLO la informacion del CONTEXTO dado. No anyadas conocimiento externo ni inventes contenido.
2. Si la informacion no esta en el contexto, responde exactamente: "Esta informacion no aparece en el temario disponible. Te recomiendo consultar directamente el material de la asignatura."
3. Adapta el nivel de la respuesta a un estudiante universitario de ciencias de la salud.
4. Usa un lenguaje claro y academico, pero accesible.
5. Si la pregunta es sobre un concepto, explica primero la definicion y luego su aplicacion practica en investigacion en salud.
6. Cita siempre la fuente (nombre del tema o manual y pagina) al final de la respuesta.
7. Responde siempre en el mismo idioma en que te hagan la pregunta (español, catalan, ingles, etc.)
8. Si el alumno hace una pregunta de seguimiento (como "puedes explicarlo mejor" o "y en ese caso..."), ten en cuenta el contexto de la conversacion anterior."""

    # Construir mensajes con historial de conversacion
    messages = [{"role": "system", "content": system_prompt}]

    # Añadir historial previo del usuario
    for msg in historial[user_id]:
        messages.append(msg)

    # Añadir la pregunta actual con el contexto
    messages.append({
        "role": "user",
        "content": f"FRAGMENTOS DEL TEMARIO:\n{contexto}\n\nPREGUNTA DEL ALUMNO:\n{pregunta}"
    })

    for intento in range(reintentos + 1):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=1024,
                temperature=0.2
            )
            respuesta = response.choices[0].message.content

            # Guardar en historial
            historial[user_id].append({"role": "user", "content": pregunta})
            historial[user_id].append({"role": "assistant", "content": respuesta})

            # Mantener solo los ultimos MAX_HISTORIAL mensajes
            if len(historial[user_id]) > MAX_HISTORIAL * 2:
                historial[user_id] = historial[user_id][-(MAX_HISTORIAL * 2):]

            return respuesta

        except Exception as e:
            if intento < reintentos:
                logger.warning(f"Intento {intento+1} fallido, reintentando en 2 seg: {e}")
                time.sleep(2)
            else:
                raise


# ─── Handlers de Telegram ─────────────────────────────────────────────────────
async def comando_inicio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    total = collection.count()
    await update.message.reply_text(
        f"Este chatbot ha sido diseñado para resolver dudas y responder preguntas exclusivamente a partir del contenido de los apuntes y documentos proporcionados.\n\n"
	    f"Recomendaciones de uso:\n\n"
        f"- Realice preguntas directas y concretas sobre los contenidos."
        f"- Cuanto más específica sea la pregunta, más precisa será la respuesta."
        f"- El sistema responderá únicamente utilizando la información disponible en los documentos cargados."
        f"- Si una cuestión no aparece en los apuntes, el chatbot puede no disponer de información suficiente para responder correctamente."
        f"Tengo {total} fragmentos de tus manuales disponibles.\n\n"
        f"El chatbot puede:\n\n"
        f"- Explicar conceptos incluidos en los apuntes."
        f"- Resumir contenidos."
        f"- Resolver dudas concretas."
        f"- Comparar conceptos presentes en la documentación."
        f"- Generar preguntas tipo test para practicar, si se solicita expresamente."
        f"Ejemplos de preguntas útiles:\n\n"
        f"Explícame la diferencia entre…"
        f"Resume el tema 3."
        f"¿Qué significa… según los apuntes?"
        f"Genera 10 preguntas tipo test sobre este tema."
        f"Hazme preguntas de examen sobre…"
        f"Escribeme tu pregunta.\n\n"
    )


async def comando_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    total = collection.count()
    todos = collection.get(include=["metadatas"])
    fuentes = set(m.get("source", "?") for m in todos["metadatas"])
    lista = "\n".join(f"- {f}" for f in sorted(fuentes)) if fuentes else "Ninguno aun."
    await update.message.reply_text(
        f"Material disponible ({total} fragmentos en total):\n\n{lista}"
    )

async def comando_limpiar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    historial[user_id] = []
    await update.message.reply_text(
        "Historial de conversacion borrado. Puedes empezar una nueva consulta."
    )

async def manejar_mensaje(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pregunta  = update.message.text.strip()
    user_id   = update.message.from_user.id
    if not pregunta:
        return
    await update.message.reply_chat_action("typing")
    try:
        contexto  = buscar_contexto(pregunta)
        respuesta = generar_respuesta(user_id, pregunta, contexto)
        await update.message.reply_text(respuesta)
    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text(
            "Ha ocurrido un error procesando tu pregunta. Por favor, intentalo de nuevo en unos segundos."
        )


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("Falta TELEGRAM_TOKEN")
    if not GROQ_API_KEY:
        raise ValueError("Falta GROQ_API_KEY")

    # Forzar cierre de cualquier sesion anterior
    for _ in range(5):
        try:
            req.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook?drop_pending_updates=true", timeout=5)
            req.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates", json={"offset": -1, "timeout": 0}, timeout=5)
        except Exception:
            pass
        time.sleep(2)

    # Arrancar servidor web en hilo separado
    hilo_web = threading.Thread(target=iniciar_servidor_web, daemon=True)
    hilo_web.start()

    # Indexar PDFs
    logger.info("Indexando PDFs...")
    indexar_pdfs()

    # Arrancar bot
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",   comando_inicio))
    app.add_handler(CommandHandler("info",    comando_info))
    app.add_handler(CommandHandler("limpiar", comando_limpiar))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, manejar_mensaje))

    logger.info("Bot iniciado. Esperando mensajes...")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True, close_loop=False)


if __name__ == "__main__":
    main()
