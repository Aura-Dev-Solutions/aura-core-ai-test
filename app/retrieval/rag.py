from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.retrieval.searcher import SemanticSearcher
from app.config import settings
from loguru import logger

class RAGService:
    def __init__(self):
        self.searcher = SemanticSearcher()
        
        # Configuración de OpenRouter
        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
            temperature=0
        )
        
        self.prompt = ChatPromptTemplate.from_template(
            """Eres un asistente experto en análisis de documentos. 
            Usa el siguiente contexto recuperado para responder a la pregunta del usuario.
            Si no encuentras la respuesta en el contexto, di que no tienes suficiente información.
            
            Contexto:
            {context}
            
            Pregunta: {question}
            
            Respuesta:"""
        )
        
        self.chain = (
            {"context": lambda x: self._format_context(x["query"]), "question": lambda x: x["query"]}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_context(self, query: str) -> str:
        """Recupera chunks y los formatea como string único."""
        results = self.searcher.search(query, limit=5)
        formatted_docs = []
        for res in results:
            source = res["metadata"].get("source", "unknown")
            text = res["text"]
            formatted_docs.append(f"--- Documento: {source} ---\n{text}\n")
        
        return "\n".join(formatted_docs)

    def generate_answer(self, query: str):
        """Genera una respuesta usando RAG."""
        logger.info(f"Generando respuesta RAG para: {query}")
        
        # Recuperamos documentos primero para devolverlos en la respuesta
        sources = self.searcher.search(query, limit=5)
        
        # Ejecutamos la cadena
        answer = self.chain.invoke({"query": query})
        
        return {
            "answer": answer,
            "sources": [s["metadata"] for s in sources]
        }
