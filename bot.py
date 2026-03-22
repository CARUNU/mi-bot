"""
Bot de Telegram con RAG (Retrieval-Augmented Generation)
Responde preguntas basándose ÚNICAMENTE en los PDFs proporcionados.

Requisitos:
    pip install python-telegram-bot chromadb langchain langchain-community
    pip install pypdf sentence-transformers groq

Variables de entorno necesarias:
    TELEGRAM_TOKEN  → Token de tu bot (de @BotFather)
    GROQ_API_KEY    → API key de groq.com (gratuita)
"""

import os
import logging
from pathlib import Path
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq

# ─── Configuración de logging ─────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─── Variables de entorno ─────────────────────────────────────────────────────
Variables de entorno necesarias:
    TELEGRAM_TOKEN  → 8329164294:AAEddXzyvjUkQIkrUZNRGo4TJ37KFmAsmF8
    GROQ_API_KEY    → gsk_dp0hJzugiPRERSYyyDXHWGdyb3FYROrsB73ag7q7psdqe8q71Gy

# ─── Carpeta donde pones tus PDFs ─────────────────────────────────────────────
PDF_FOLDER = "manuales"   # Crea esta carpeta y pon tus PDFs dentro

# ─── Inicializar clientes ─────────────────────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)

# Base de datos vectorial local (se guarda en ./chroma_db)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn   = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = chroma_client.get_or_create_collection(
    name="manuales",
    embedding_function=embedding_fn
)


# ─── Indexar PDFs ─────────────────────────────────────────────────────────────
def indexar_pdfs():
    """
    Lee todos los PDFs de la carpeta 'manuales', los divide en fragmentos
    y los guarda en ChromaDB para búsqueda semántica.
    Solo indexa PDFs que aún no estén en la base de datos.
    """
    pdf_folder = Path(PDF_FOLDER)
    if not pdf_folder.exists():
        pdf_folder.mkdir()
        logger.warning(f"Carpeta '{PDF_FOLDER}' creada. Añade tus PDFs y reinicia el bot.")
        return

    pdfs = list(pdf_folder.glob("*.pdf"))
    if not pdfs:
        logger.warning(f"No hay PDFs en la carpeta '{PDF_FOLDER}'.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # Caracteres por fragmento
        chunk_overlap=100,    # Solapamiento para no perder contexto entre fragmentos
    )

    for pdf_path in pdfs:
        pdf_name = pdf_path.stem

        # Comprobar si ya está indexado
        existing = collection.get(where={"source": pdf_name})
        if existing["ids"]:
            logger.info(f"'{pdf_name}' ya está indexado, omitiendo.")
            continue

        logger.info(f"Indexando '{pdf_name}'...")
        loader    = PyPDFLoader(str(pdf_path))
        pages     = loader.load()
        fragments = splitter.split_documents(pages)

        ids       = [f"{pdf_name}__{i}" for i in range(len(fragments))]
        texts     = [f.page_content for f in fragments]
        metadatas = [{"source": pdf_name, "page": f.metadata.get("page", 0)} for f in fragments]

        # Insertar en lotes de 100 para no sobrecargar memoria
        batch = 100
        for start in range(0, len(texts), batch):
            collection.add(
                ids=ids[start:start+batch],
                documents=texts[start:start+batch],
                metadatas=metadatas[start:start+batch]
            )

        logger.info(f"'{pdf_name}' indexado: {len(fragments)} fragmentos.")

    logger.info("Indexación completada.")


# ─── Buscar contexto relevante ────────────────────────────────────────────────
def buscar_contexto(pregunta: str, n_resultados: int = 5) -> str:
    """
    Busca en ChromaDB los fragmentos más relevantes para la pregunta.
    Devuelve un texto con el contexto encontrado.
    """
    resultados = collection.query(
        query_texts=[pregunta],
        n_results=n_resultados
    )

    if not resultados["documents"] or not resultados["documents"][0]:
        return ""

    fragmentos = []
    for doc, meta in zip(resultados["documents"][0], resultados["metadatas"][0]):
        fuente = meta.get("source", "desconocido")
        pagina = meta.get("page", "?")
        fragmentos.append(f"[{fuente} - pág. {pagina}]\n{doc}")

    return "\n\n---\n\n".join(fragmentos)


# ─── Generar respuesta con Groq ───────────────────────────────────────────────
def generar_respuesta(pregunta: str, contexto: str) -> str:
    """
    Llama a la API de Groq con el contexto de los PDFs y la pregunta.
    Si no hay contexto relevante, lo indica claramente.
    """
    if not contexto:
        return (
            "Lo siento, no he encontrado información sobre eso en los manuales disponibles. "
            "Por favor, reformula la pregunta o consulta directamente el manual."
        )

    system_prompt = """Eres un asistente especializado que responde preguntas EXCLUSIVAMENTE 
basándose en la información de los manuales proporcionados.

REGLAS ESTRICTAS:
1. Solo usa la información del CONTEXTO dado. No añadas conocimiento externo.
2. Si la información no está en el contexto, di exactamente: "Esta información no está en los manuales disponibles."
3. Cita siempre la fuente (nombre del manual y página) al final de tu respuesta.
4. Responde en el mismo idioma en que te hagan la pregunta.
5. Sé conciso y claro."""

    user_message = f"""CONTEXTO DE LOS MANUALES:
{contexto}

PREGUNTA DEL USUARIO:
{pregunta}

Responde basándote únicamente en el contexto anterior."""

    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",   # Modelo gratuito de Groq
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ],
        max_tokens=1024,
        temperature=0.1   # Baja temperatura = respuestas más precisas y conservadoras
    )

    return response.choices[0].message.content


# ─── Handlers de Telegram ─────────────────────────────────────────────────────
async def comando_inicio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mensaje de bienvenida cuando el usuario escribe /start."""
    total = collection.count()
    await update.message.reply_text(
        f"Hola! Soy un asistente especializado.\n\n"
        f"Puedo responder preguntas basándome en {total} fragmentos de los manuales disponibles.\n\n"
        f"Escríbeme tu pregunta y la buscaré en los manuales."
    )


async def comando_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Muestra información sobre los manuales indexados."""
    total = collection.count()
    # Obtener fuentes únicas
    todos = collection.get(include=["metadatas"])
    fuentes = set(m.get("source", "?") for m in todos["metadatas"])
    lista = "\n".join(f"• {f}" for f in sorted(fuentes)) if fuentes else "Ninguno aún."
    await update.message.reply_text(
        f"Manuales disponibles ({total} fragmentos en total):\n\n{lista}"
    )


async def manejar_mensaje(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Procesa cada mensaje del usuario y devuelve la respuesta."""
    pregunta = update.message.text.strip()
    if not pregunta:
        return

    # Indicar que el bot está procesando
    await update.message.reply_chat_action("typing")

    try:
        contexto  = buscar_contexto(pregunta)
        respuesta = generar_respuesta(pregunta, contexto)
        await update.message.reply_text(respuesta)
    except Exception as e:
        logger.error(f"Error procesando mensaje: {e}")
        await update.message.reply_text(
            "Ha ocurrido un error al procesar tu pregunta. Por favor, inténtalo de nuevo."
        )


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("Falta la variable de entorno TELEGRAM_TOKEN")
    if not GROQ_API_KEY:
        raise ValueError("Falta la variable de entorno GROQ_API_KEY")

    # Indexar PDFs al arrancar
    logger.info("Iniciando indexación de PDFs...")
    indexar_pdfs()

    # Crear y configurar la aplicación de Telegram
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", comando_inicio))
    app.add_handler(CommandHandler("info",  comando_info))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, manejar_mensaje))

    logger.info("Bot iniciado. Esperando mensajes...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
