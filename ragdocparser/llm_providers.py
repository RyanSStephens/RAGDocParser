"""
LLM provider integrations for RAGDocParser.
"""

import os
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_response(self, query: str, context: List[str], **kwargs) -> str:
        """Generate response using retrieved context."""
        pass
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI provider for GPT models."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """Initialize OpenAI provider."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
    
    def generate_response(self, query: str, context: List[str], **kwargs) -> str:
        """Generate response using GPT model."""
        context_text = "\n".join(context)
        
        prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say "I don't have enough information to answer that question."

Context:
{context_text}

Question: {query}

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=kwargs.get("max_tokens", 500),
                temperature=kwargs.get("temperature", 0.7)
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {e}")
            return "Error generating response."
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            
            return [item.embedding for item in response.data]
        
        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {e}")
            return []

class AnthropicProvider(LLMProvider):
    """Anthropic provider for Claude models."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-2.1"):
        """Initialize Anthropic provider."""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("Anthropic API key not found")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
    
    def generate_response(self, query: str, context: List[str], **kwargs) -> str:
        """Generate response using Claude model."""
        context_text = "\n".join(context)
        
        prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say "I don't have enough information to answer that question."

Context:
{context_text}

Question: {query}

Answer:"""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 500),
                temperature=kwargs.get("temperature", 0.7),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text.strip()
        
        except Exception as e:
            logger.error(f"Error generating Anthropic response: {e}")
            return "Error generating response."
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings (Anthropic doesn't provide embeddings API)."""
        logger.warning("Anthropic doesn't provide embeddings API. Use OpenAI or local models.")
        return []

class CohereProvider(LLMProvider):
    """Cohere provider for command models."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "command"):
        """Initialize Cohere provider."""
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("Cohere API key not found")
        
        try:
            import cohere
            self.client = cohere.Client(self.api_key)
        except ImportError:
            raise ImportError("Cohere package not installed. Install with: pip install cohere")
    
    def generate_response(self, query: str, context: List[str], **kwargs) -> str:
        """Generate response using Cohere model."""
        context_text = "\n".join(context)
        
        prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say "I don't have enough information to answer that question."

Context:
{context_text}

Question: {query}

Answer:"""
        
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                max_tokens=kwargs.get("max_tokens", 500),
                temperature=kwargs.get("temperature", 0.7),
                stop_sequences=kwargs.get("stop_sequences", [])
            )
            
            return response.generations[0].text.strip()
        
        except Exception as e:
            logger.error(f"Error generating Cohere response: {e}")
            return "Error generating response."
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Cohere."""
        try:
            response = self.client.embed(
                texts=texts,
                model="embed-english-v3.0"
            )
            
            return response.embeddings
        
        except Exception as e:
            logger.error(f"Error generating Cohere embeddings: {e}")
            return []

class RAGManager:
    """Manages RAG pipeline with different LLM providers."""
    
    def __init__(self, provider: LLMProvider, vectordb_manager):
        """Initialize RAG manager."""
        self.provider = provider
        self.vectordb = vectordb_manager
    
    def ask_question(self, 
                    question: str, 
                    collection_name: str = None,
                    k: int = 5,
                    **kwargs) -> Dict[str, Any]:
        """Ask a question and get an answer with sources."""
        
        # Retrieve relevant documents
        search_results = self.vectordb.search_similar(
            question, 
            k=k, 
            collection_name=collection_name
        )
        
        if not search_results:
            return {
                "answer": "No relevant documents found for your question.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Extract context
        context = [result["content"] for result in search_results]
        
        # Generate answer
        answer = self.provider.generate_response(question, context, **kwargs)
        
        # Calculate confidence based on similarity scores
        if search_results and "distance" in search_results[0]:
            avg_similarity = sum(1 - r.get("distance", 1) for r in search_results) / len(search_results)
            confidence = min(avg_similarity, 1.0)
        else:
            confidence = 0.5  # Default confidence when no distance available
        
        # Prepare sources
        sources = []
        for result in search_results:
            source = {
                "content": result["content"][:200] + "...",
                "metadata": result.get("metadata", {}),
                "similarity": 1 - result.get("distance", 0) if "distance" in result else None
            }
            sources.append(source)
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "question": question
        }
    
    def chat(self, 
            question: str,
            chat_history: List[Dict[str, str]] = None,
            collection_name: str = None,
            k: int = 5) -> Dict[str, Any]:
        """Enhanced chat with conversation history."""
        
        chat_history = chat_history or []
        
        # Build context from chat history
        history_context = ""
        if chat_history:
            history_context = "\n".join([
                f"Human: {turn['question']}\nAssistant: {turn['answer']}"
                for turn in chat_history[-3:]  # Last 3 turns
            ])
        
        # Get answer for current question
        result = self.ask_question(question, collection_name, k)
        
        # Add to chat history
        chat_turn = {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"]
        }
        
        return chat_turn 