"""
Advanced retrieval techniques for RAG systems.
Implements multiple retrieval strategies for improved context selection.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Enhanced retrieval result with scoring information."""
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    relevance_score: float
    diversity_score: float
    final_score: float
    retrieval_strategy: str

class RetrievalStrategy(ABC):
    """Abstract base class for retrieval strategies."""
    
    @abstractmethod
    def retrieve(
        self, 
        query: str, 
        vectordb, 
        k: int = 5,
        collection_name: str = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents using this strategy."""
        pass

class SemanticRetrievalStrategy(RetrievalStrategy):
    """Standard semantic similarity retrieval."""
    
    def retrieve(
        self, 
        query: str, 
        vectordb, 
        k: int = 5,
        collection_name: str = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """Perform semantic similarity search."""
        try:
            raw_results = vectordb.search_similar(
                query, k=k * 2, collection_name=collection_name
            )
            
            results = []
            for result in raw_results[:k]:
                similarity_score = 1 - result.get("distance", 1.0)
                
                retrieval_result = RetrievalResult(
                    content=result["content"],
                    metadata=result.get("metadata", {}),
                    similarity_score=similarity_score,
                    relevance_score=similarity_score,
                    diversity_score=1.0,
                    final_score=similarity_score,
                    retrieval_strategy="semantic"
                )
                results.append(retrieval_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic retrieval: {e}")
            return []

class HybridRetrievalStrategy(RetrievalStrategy):
    """Hybrid retrieval combining semantic and keyword matching."""
    
    def __init__(self, semantic_weight: float = 0.7, keyword_weight: float = 0.3):
        """Initialize hybrid retrieval.
        
        Args:
            semantic_weight: Weight for semantic similarity (0-1)
            keyword_weight: Weight for keyword matching (0-1)
        """
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        
        if abs(semantic_weight + keyword_weight - 1.0) > 0.01:
            logger.warning("Semantic and keyword weights should sum to 1.0")
    
    def _calculate_keyword_score(self, query: str, content: str) -> float:
        """Calculate keyword matching score."""
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        content_terms = set(re.findall(r'\b\w+\b', content.lower()))
        
        if not query_terms:
            return 0.0
        
        intersection = query_terms.intersection(content_terms)
        return len(intersection) / len(query_terms)
    
    def retrieve(
        self, 
        query: str, 
        vectordb, 
        k: int = 5,
        collection_name: str = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """Perform hybrid retrieval."""
        try:
            # Get more candidates for reranking
            raw_results = vectordb.search_similar(
                query, k=k * 3, collection_name=collection_name
            )
            
            results = []
            for result in raw_results:
                similarity_score = 1 - result.get("distance", 1.0)
                keyword_score = self._calculate_keyword_score(query, result["content"])
                
                final_score = (
                    self.semantic_weight * similarity_score + 
                    self.keyword_weight * keyword_score
                )
                
                retrieval_result = RetrievalResult(
                    content=result["content"],
                    metadata=result.get("metadata", {}),
                    similarity_score=similarity_score,
                    relevance_score=keyword_score,
                    diversity_score=1.0,
                    final_score=final_score,
                    retrieval_strategy="hybrid"
                )
                results.append(retrieval_result)
            
            # Sort by final score and return top k
            results.sort(key=lambda x: x.final_score, reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            return []

class DiversityRetrievalStrategy(RetrievalStrategy):
    """Retrieval strategy that promotes diversity in results."""
    
    def __init__(self, diversity_threshold: float = 0.8):
        """Initialize diversity retrieval.
        
        Args:
            diversity_threshold: Similarity threshold for diversity filtering
        """
        self.diversity_threshold = diversity_threshold
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple content similarity (Jaccard similarity)."""
        words1 = set(re.findall(r'\b\w+\b', content1.lower()))
        words2 = set(re.findall(r'\b\w+\b', content2.lower()))
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def retrieve(
        self, 
        query: str, 
        vectordb, 
        k: int = 5,
        collection_name: str = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """Perform diversity-aware retrieval."""
        try:
            # Get more candidates
            raw_results = vectordb.search_similar(
                query, k=k * 4, collection_name=collection_name
            )
            
            selected_results = []
            
            for result in raw_results:
                similarity_score = 1 - result.get("distance", 1.0)
                
                # Calculate diversity score
                diversity_score = 1.0
                for selected in selected_results:
                    content_sim = self._calculate_content_similarity(
                        result["content"], selected.content
                    )
                    if content_sim > self.diversity_threshold:
                        diversity_score *= (1 - content_sim)
                
                final_score = similarity_score * diversity_score
                
                retrieval_result = RetrievalResult(
                    content=result["content"],
                    metadata=result.get("metadata", {}),
                    similarity_score=similarity_score,
                    relevance_score=similarity_score,
                    diversity_score=diversity_score,
                    final_score=final_score,
                    retrieval_strategy="diversity"
                )
                
                selected_results.append(retrieval_result)
                
                if len(selected_results) >= k:
                    break
            
            # Sort by final score
            selected_results.sort(key=lambda x: x.final_score, reverse=True)
            return selected_results[:k]
            
        except Exception as e:
            logger.error(f"Error in diversity retrieval: {e}")
            return []

class MMRRetrievalStrategy(RetrievalStrategy):
    """Maximal Marginal Relevance retrieval strategy."""
    
    def __init__(self, lambda_param: float = 0.5):
        """Initialize MMR retrieval.
        
        Args:
            lambda_param: Balance between relevance and diversity (0-1)
        """
        self.lambda_param = lambda_param
    
    def retrieve(
        self, 
        query: str, 
        vectordb, 
        k: int = 5,
        collection_name: str = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """Perform MMR retrieval."""
        try:
            # Get candidates
            raw_results = vectordb.search_similar(
                query, k=k * 3, collection_name=collection_name
            )
            
            if not raw_results:
                return []
            
            candidates = [
                {
                    "content": result["content"],
                    "metadata": result.get("metadata", {}),
                    "similarity": 1 - result.get("distance", 1.0)
                }
                for result in raw_results
            ]
            
            selected = []
            
            # Select first document (highest similarity)
            candidates.sort(key=lambda x: x["similarity"], reverse=True)
            selected.append(candidates.pop(0))
            
            # Select remaining documents using MMR
            while len(selected) < k and candidates:
                best_mmr_score = -1
                best_idx = 0
                
                for i, candidate in enumerate(candidates):
                    # Calculate max similarity to already selected documents
                    max_sim_to_selected = 0
                    for selected_doc in selected:
                        sim = self._calculate_content_similarity(
                            candidate["content"], selected_doc["content"]
                        )
                        max_sim_to_selected = max(max_sim_to_selected, sim)
                    
                    # Calculate MMR score
                    mmr_score = (
                        self.lambda_param * candidate["similarity"] - 
                        (1 - self.lambda_param) * max_sim_to_selected
                    )
                    
                    if mmr_score > best_mmr_score:
                        best_mmr_score = mmr_score
                        best_idx = i
                
                selected.append(candidates.pop(best_idx))
            
            # Convert to RetrievalResult objects
            results = []
            for i, doc in enumerate(selected):
                retrieval_result = RetrievalResult(
                    content=doc["content"],
                    metadata=doc["metadata"],
                    similarity_score=doc["similarity"],
                    relevance_score=doc["similarity"],
                    diversity_score=1.0,  # Could be calculated more precisely
                    final_score=doc["similarity"],
                    retrieval_strategy="mmr"
                )
                results.append(retrieval_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in MMR retrieval: {e}")
            return []
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple content similarity."""
        words1 = set(re.findall(r'\b\w+\b', content1.lower()))
        words2 = set(re.findall(r'\b\w+\b', content2.lower()))
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

class AdvancedRetriever:
    """Advanced retriever with multiple strategies."""
    
    def __init__(self):
        """Initialize advanced retriever."""
        self.strategies = {
            "semantic": SemanticRetrievalStrategy(),
            "hybrid": HybridRetrievalStrategy(),
            "diversity": DiversityRetrievalStrategy(),
            "mmr": MMRRetrievalStrategy()
        }
        self.default_strategy = "hybrid"
    
    def retrieve(
        self,
        query: str,
        vectordb,
        k: int = 5,
        collection_name: str = None,
        strategy: str = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """Retrieve documents using specified strategy.
        
        Args:
            query: Search query
            vectordb: Vector database instance
            k: Number of documents to retrieve
            collection_name: Collection to search in
            strategy: Retrieval strategy to use
            **kwargs: Additional strategy-specific parameters
        """
        strategy_name = strategy or self.default_strategy
        
        if strategy_name not in self.strategies:
            logger.warning(f"Unknown strategy '{strategy_name}', using default")
            strategy_name = self.default_strategy
        
        retrieval_strategy = self.strategies[strategy_name]
        
        try:
            return retrieval_strategy.retrieve(
                query=query,
                vectordb=vectordb,
                k=k,
                collection_name=collection_name,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in retrieval with strategy '{strategy_name}': {e}")
            return []
    
    def add_strategy(self, name: str, strategy: RetrievalStrategy):
        """Add a custom retrieval strategy."""
        self.strategies[name] = strategy
        logger.info(f"Added retrieval strategy: {name}")
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available retrieval strategies."""
        return list(self.strategies.keys())
    
    def ensemble_retrieve(
        self,
        query: str,
        vectordb,
        k: int = 5,
        collection_name: str = None,
        strategies: List[str] = None,
        weights: List[float] = None
    ) -> List[RetrievalResult]:
        """Perform ensemble retrieval using multiple strategies.
        
        Args:
            query: Search query
            vectordb: Vector database instance
            k: Number of documents to retrieve
            collection_name: Collection to search in
            strategies: List of strategies to combine
            weights: Weights for each strategy (should sum to 1.0)
        """
        if strategies is None:
            strategies = ["semantic", "hybrid"]
        
        if weights is None:
            weights = [1.0 / len(strategies)] * len(strategies)
        
        if len(strategies) != len(weights):
            raise ValueError("Number of strategies must match number of weights")
        
        # Collect results from all strategies
        all_results = {}  # content -> RetrievalResult
        
        for strategy_name, weight in zip(strategies, weights):
            strategy_results = self.retrieve(
                query=query,
                vectordb=vectordb,
                k=k * 2,  # Get more candidates
                collection_name=collection_name,
                strategy=strategy_name
            )
            
            for result in strategy_results:
                content = result.content
                if content in all_results:
                    # Combine scores
                    existing = all_results[content]
                    existing.final_score += weight * result.final_score
                else:
                    # New result
                    result.final_score *= weight
                    all_results[content] = result
        
        # Sort by combined scores and return top k
        final_results = list(all_results.values())
        final_results.sort(key=lambda x: x.final_score, reverse=True)
        
        # Update strategy name for ensemble results
        for result in final_results[:k]:
            result.retrieval_strategy = f"ensemble({'+'.join(strategies)})"
        
        return final_results[:k] 