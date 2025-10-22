"""
Query transformation module for RAG enhancement
Includes multi-query generation and HyDE (Hypothetical Document Embeddings)
"""
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import time
import asyncio


class QueryTransformer:
    """
    Service for transforming queries to improve retrieval
    """

    def __init__(self, llm_model: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Initialize query transformer

        Args:
            llm_model: LLM model name
            temperature: Temperature for generation
        """
        # Initialize with connection pooling if httpx is available
        try:
            import httpx
            http_client = httpx.Client(
                timeout=30.0,
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10,
                    keepalive_expiry=30.0
                )
            )
            self.llm = ChatOpenAI(model=llm_model, temperature=temperature, http_client=http_client)
            self.llm_deterministic = ChatOpenAI(model=llm_model, temperature=0.3, http_client=http_client)
        except ImportError:
            # Fallback without connection pooling
            self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
            self.llm_deterministic = ChatOpenAI(model=llm_model, temperature=0.3)

    def generate_multi_queries(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Generate multiple variations of the input query

        Args:
            query: Original user query
            num_queries: Number of query variations to generate

        Returns:
            List of query variations (including original)
        """
        try:
            system_prompt = """You are a helpful assistant that generates query variations.
Given a user query, generate {num} alternative phrasings that capture the same intent.

Rules:
- Keep queries concise and clear
- Maintain the original intent
- Vary the wording and structure
- Return only the queries, one per line, numbered"""

            user_prompt = f"""Original query: {query}

Generate {num_queries - 1} alternative queries:"""

            response = self.llm.invoke([
                SystemMessage(content=system_prompt.format(num=num_queries - 1)),
                HumanMessage(content=user_prompt)
            ])

            # Parse response to extract queries
            generated_queries = []
            for line in response.content.strip().split('\n'):
                # Remove numbering (e.g., "1. ", "2. ", etc.)
                clean_line = line.strip()
                if clean_line:
                    # Remove leading numbers and punctuation
                    import re
                    clean_query = re.sub(r'^\d+[\.\)]\s*', '', clean_line)
                    if clean_query and clean_query != query:
                        generated_queries.append(clean_query)

            # Combine with original query
            all_queries = [query] + generated_queries[:num_queries - 1]

            print(f"  ✓ Generated {len(all_queries)} query variations")
            return all_queries

        except Exception as e:
            print(f"  ⚠️  Multi-query generation failed: {e}")
            print(f"  ℹ️  Falling back to original query")
            return [query]

    def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document (HyDE) that would answer the query

        Args:
            query: User query

        Returns:
            Hypothetical document text
        """
        try:
            system_prompt = """You are a helpful assistant for an educational institution.
Given a question, write a brief hypothetical answer (2-3 sentences) that a relevant document might contain.

Write as if you are quoting from an official document or webpage. Be factual and informative."""

            user_prompt = f"Question: {query}\n\nHypothetical answer:"

            response = self.llm_deterministic.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            hypothetical_doc = response.content.strip()
            print(f"  ✓ Generated hypothetical document ({len(hypothetical_doc)} chars)")

            return hypothetical_doc

        except Exception as e:
            print(f"  ⚠️  HyDE generation failed: {e}")
            print(f"  ℹ️  Falling back to original query")
            return query

    def transform_query(
        self,
        query: str,
        use_multi_query: bool = True,
        use_hyde: bool = True,
        num_variations: int = 3
    ) -> Dict[str, Any]:
        """
        Transform query using multiple techniques

        Args:
            query: Original user query
            use_multi_query: Whether to generate query variations
            use_hyde: Whether to generate hypothetical document
            num_variations: Number of query variations (if multi-query enabled)

        Returns:
            Dict containing:
                - original_query: Original query
                - query_variations: List of query variations (if enabled)
                - hypothetical_doc: Hypothetical document (if enabled)
        """
        result = {
            "original_query": query,
            "query_variations": None,
            "hypothetical_doc": None
        }

        # Generate query variations
        if use_multi_query:
            start = time.time()
            result["query_variations"] = self.generate_multi_queries(query, num_variations)
            elapsed = time.time() - start
            print(f"  ⏱️  [QUERY_TRANSFORM] Multi-query: {elapsed:.3f}s")

        # Generate hypothetical document
        if use_hyde:
            start = time.time()
            result["hypothetical_doc"] = self.generate_hypothetical_document(query)
            elapsed = time.time() - start
            print(f"  ⏱️  [QUERY_TRANSFORM] HyDE: {elapsed:.3f}s")

        return result

    async def generate_multi_queries_async(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Generate multiple variations of the input query (async version)

        Args:
            query: Original user query
            num_queries: Number of query variations to generate

        Returns:
            List of query variations (including original)
        """
        try:
            system_prompt = """You are a helpful assistant that generates query variations.
Given a user query, generate {num} alternative phrasings that capture the same intent.

Rules:
- Keep queries concise and clear
- Maintain the original intent
- Vary the wording and structure
- Return only the queries, one per line, numbered"""

            user_prompt = f"""Original query: {query}

Generate {num_queries - 1} alternative queries:"""

            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt.format(num=num_queries - 1)),
                HumanMessage(content=user_prompt)
            ])

            # Parse response to extract queries
            generated_queries = []
            for line in response.content.strip().split('\n'):
                # Remove numbering (e.g., "1. ", "2. ", etc.)
                clean_line = line.strip()
                if clean_line:
                    # Remove leading numbers and punctuation
                    import re
                    clean_query = re.sub(r'^\d+[\.\)]\s*', '', clean_line)
                    if clean_query and clean_query != query:
                        generated_queries.append(clean_query)

            # Combine with original query
            all_queries = [query] + generated_queries[:num_queries - 1]

            print(f"  ✓ Generated {len(all_queries)} query variations")
            return all_queries

        except Exception as e:
            print(f"  ⚠️  Multi-query generation failed: {e}")
            print(f"  ℹ️  Falling back to original query")
            return [query]

    async def generate_hypothetical_document_async(self, query: str) -> str:
        """
        Generate a hypothetical document (HyDE) that would answer the query (async version)

        Args:
            query: User query

        Returns:
            Hypothetical document text
        """
        try:
            system_prompt = """You are a helpful assistant for an educational institution.
Given a question, write a brief hypothetical answer (2-3 sentences) that a relevant document might contain.

Write as if you are quoting from an official document or webpage. Be factual and informative."""

            user_prompt = f"Question: {query}\n\nHypothetical answer:"

            response = await self.llm_deterministic.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            hypothetical_doc = response.content.strip()
            print(f"  ✓ Generated hypothetical document ({len(hypothetical_doc)} chars)")

            return hypothetical_doc

        except Exception as e:
            print(f"  ⚠️  HyDE generation failed: {e}")
            print(f"  ℹ️  Falling back to original query")
            return query

    async def transform_query_async(
        self,
        query: str,
        use_multi_query: bool = True,
        use_hyde: bool = True,
        num_variations: int = 3
    ) -> Dict[str, Any]:
        """
        Transform query using multiple techniques (async version with parallel execution)

        Args:
            query: Original user query
            use_multi_query: Whether to generate query variations
            use_hyde: Whether to generate hypothetical document
            num_variations: Number of query variations (if multi-query enabled)

        Returns:
            Dict containing:
                - original_query: Original query
                - query_variations: List of query variations (if enabled)
                - hypothetical_doc: Hypothetical document (if enabled)
        """
        result = {
            "original_query": query,
            "query_variations": None,
            "hypothetical_doc": None
        }

        # Run both transformations in parallel
        tasks = []
        task_names = []

        if use_multi_query:
            tasks.append(self.generate_multi_queries_async(query, num_variations))
            task_names.append("multi_query")

        if use_hyde:
            tasks.append(self.generate_hypothetical_document_async(query))
            task_names.append("hyde")

        if tasks:
            start = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.time() - start
            print(f"  ⏱️  [QUERY_TRANSFORM] Parallel execution: {elapsed:.3f}s")

            # Map results back
            for task_name, task_result in zip(task_names, results):
                if isinstance(task_result, Exception):
                    print(f"  ⚠️  [QUERY_TRANSFORM] {task_name} failed: {task_result}")
                else:
                    if task_name == "multi_query":
                        result["query_variations"] = task_result
                    elif task_name == "hyde":
                        result["hypothetical_doc"] = task_result

        return result


# Global instance
query_transformer: QueryTransformer = None


def get_query_transformer(llm_model: str = "gpt-4o-mini") -> QueryTransformer:
    """
    Get or create global query transformer instance

    Args:
        llm_model: LLM model name

    Returns:
        QueryTransformer instance
    """
    global query_transformer

    if query_transformer is None:
        query_transformer = QueryTransformer(llm_model=llm_model)

    return query_transformer
