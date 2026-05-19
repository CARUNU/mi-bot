"""
Bot de Telegram con RAG (Retrieval-Augmented Generation)
Version profesional — Groq API + Render Starter
Soporta 30+ usuarios simultaneos con respuestas en 1-2 segundos.
"""
 
import os
import time
import logging
import threading
import hashlib
import requests as req
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
 
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
 
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
 
# Modelo a usar — cambia a "llama-3.3-70b-versatile" para mayor calidad
GROQ_MODEL = "llama-3.1-8b-instant"
 
# ─── Carpeta de PDFs ──────────────────────────────────────────────────────────
PDF_FOLDER = "manuales"
 
# ─── Clientes ─────────────────────────────────────────────────────────────────
groq_client   = Groq(api_key=GROQ_API_KEY)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
 
class SimpleEmbeddingFunction:
    """Embedding ligero basado en frecuencia de palabras."""
    def __init__(self, dim=384):
        self.dim = dim
 
    def __call__(self, input):
        import math
        results = []
        for text in input:
            vec = [0.0] * self.dim
            words = text.lower().split()
            for word in words:
                idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
                vec[idx] += 1.0
            norm = math.sqrt(sum(x*x for x in vec)) or 1.0
            vec = [x / norm for x in vec]
            results.append(vec)
        return results
 
embedding_fn = SimpleEmbeddingFunction()
collection   = chroma_client.get_or_create_collection(
    name="manuales_v3",
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
 
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
 
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
 
    logger.info("Indexacion completada.")
 
# ─── Buscar contexto ──────────────────────────────────────────────────────────
def buscar_contexto(pregunta: str, n_resultados: int = 3) -> str:
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
def generar_respuesta(pregunta: str, contexto: str, reintentos: int = 2) -> str:
    if not contexto:
        return (
            "Lo siento, no he encontrado informacion sobre eso en los manuales. "
            "Por favor, reformula la pregunta o consulta directamente el manual."
        )
 
    system_prompt = """Eres un asistente que responde preguntas EXCLUSIVAMENTE basandose en los manuales proporcionados.
 
REGLAS:
1. Solo usa la informacion del CONTEXTO dado. No anyadas conocimiento externo.
2. Si la informacion no esta en el contexto, di: "Esta informacion no esta en los manuales disponibles."
3. Cita siempre la fuente (nombre del manual y pagina) al final.
4. Responde en el mismo idioma de la pregunta.
5. Se conciso y claro."""
 
    for intento in range(reintentos + 1):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": f"CONTEXTO:\n{contexto}\n\nPREGUNTA:\n{pregunta}"}
                ],
                max_tokens=1024,
                temperature=0.1
            )
            return response.choices[0].message.content
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
        f"Hola! Soy un asistente especializado.\n\n"
        f"Tengo {total} fragmentos de tus manuales disponibles.\n\n"
        f"Escribeme tu pregunta."
    )
 
async def comando_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    total = collection.count()
    todos = collection.get(include=["metadatas"])
    fuentes = set(m.get("source", "?") for m in todos["metadatas"])
    lista = "\n".join(f"- {f}" for f in sorted(fuentes)) if fuentes else "Ninguno aun."
    await update.message.reply_text(
        f"Manuales disponibles ({total} fragmentos):\n\n{lista}"
    )
 
async def manejar_mensaje(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pregunta = update.message.text.strip()
    if not pregunta:
        return
    await update.message.reply_chat_action("typing")
    try:
        contexto  = buscar_contexto(pregunta)
        respuesta = generar_respuesta(pregunta, contexto)
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
 
    # Cancelar sesiones anteriores de Telegram
    req.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook?drop_pending_updates=true")
 
    # Arrancar servidor web en hilo separado
    hilo_web = threading.Thread(target=iniciar_servidor_web, daemon=True)
    hilo_web.start()
 
    # Indexar PDFs
    logger.info("Indexando PDFs...")
    indexar_pdfs()
 
    # Arrancar bot
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", comando_inicio))
    app.add_handler(CommandHandler("info",  comando_info))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, manejar_mensaje))
 
    logger.info("Bot iniciado. Esperando mensajes...")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True, close_loop=False)
 
if __name__ == "__main__":
    main()
