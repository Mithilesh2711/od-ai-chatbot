from typing import TypedDict, List, Dict, Any, Annotated, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from vectorStore.qdrantClient import qdrant_service
from vectorStore.embeddings import embedding_service
from config import settings
from rag.reranker import get_reranker_service
from rag.query_transformer import get_query_transformer
from rag.sentence_window import get_sentence_window_service

# State Definition
class LeadsState(TypedDict):
    """State for leads chat agent"""
    messages: Annotated[List[BaseMessage], add_messages]
    entity: str
    entity_name: Optional[str]
    query: str
    query_variations: Optional[List[str]]
    hypothetical_doc: Optional[str]
    retrieved_docs: List[Dict[str, Any]]
    answer: str
    model_name: Optional[str]  # Model name from chatBotConfig
    llm_instance: Optional[Any]  # LLM instance created from model_name

def get_llm_instance(model_name: str = None):
    """
    Initialize LLM instance based on model name from chatBotConfig.
    Falls back to default GPT-3.5 turbo if model_name not provided or invalid.

    Args:
        model_name: Model name from chatBotConfig.model (e.g., 'gpt-3.5-turbo', 'claude-sonnet', 'gemini-pro')

    Returns:
        LLM instance (ChatOpenAI, ChatAnthropic, or ChatGoogleGenerativeAI)
    """
    # Use default model if not provided
    if not model_name:
        model_name = settings.DEFAULT_MODEL

    # Get model config from MODEL_NAMES mapping
    model_config = settings.MODEL_NAMES.get(model_name)

    if not model_config:
        print(f"Warning: Unknown model '{model_name}', falling back to {settings.DEFAULT_MODEL}")
        model_name = settings.DEFAULT_MODEL
        model_config = settings.MODEL_NAMES[model_name]

    provider = model_config["provider"]
    api_key_name = model_config["api_key"]
    model = model_config["model"]

    # Get API key from settings
    api_key = getattr(settings, api_key_name, None)

    if not api_key:
        print(f"Warning: API key '{api_key_name}' not found in settings, falling back to OpenAI")
        return ChatOpenAI(model=settings.OPENAI_MODEL, temperature=0.7)

    print(f"Initializing LLM: provider={provider}, model={model}")

    # Initialize LLM based on provider
    try:
        if provider == "openai":
            return ChatOpenAI(model=model, temperature=0.7, api_key=api_key)
        elif provider == "anthropic":
            return ChatAnthropic(model=model, temperature=0.7, api_key=api_key)
        elif provider == "google":
            return ChatGoogleGenerativeAI(model=model, temperature=0.7, google_api_key=api_key)
        elif provider == "deepseek":
            # DeepSeek uses OpenAI-compatible API
            return ChatOpenAI(
                model=model,
                temperature=0.7,
                api_key=api_key,
                base_url="https://api.deepseek.com/v1"
            )
        else:
            print(f"Warning: Unknown provider '{provider}', falling back to OpenAI")
            return ChatOpenAI(model=settings.OPENAI_MODEL, temperature=0.7)
    except Exception as e:
        print(f"Error initializing {provider} LLM: {e}, falling back to OpenAI")
        return ChatOpenAI(model=settings.OPENAI_MODEL, temperature=0.7)

def query_transform_node(state: LeadsState) -> LeadsState:
    """
    Transform query using multi-query generation and HyDE (with async parallelization)
    """
    import time
    import asyncio
    node_start = time.time()

    query = state["query"]
    llm = state.get("llm_instance")

    if not llm:
        print("  ⚠️  [QUERY_TRANSFORM] No LLM instance in state, using default")
        llm = get_llm_instance()

    print(f"  ℹ️  [QUERY_TRANSFORM] Transforming query: {query[:50]}...")

    try:
        # Get model name for transformer (use generic model name for now)
        model_name = state.get("model_name", settings.DEFAULT_MODEL)

        # Get query transformer
        transformer = get_query_transformer(llm_model=model_name)

        # Transform query using async version for parallel execution
        # Run the async function in the event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        transformed = loop.run_until_complete(
            transformer.transform_query_async(
                query=query,
                use_multi_query=True,
                use_hyde=True,
                num_variations=3
            )
        )

        node_end = time.time()
        print(f"  ⏱️  [QUERY_TRANSFORM] Total node time: {node_end - node_start:.3f}s")

        return {
            **state,
            "query_variations": transformed.get("query_variations"),
            "hypothetical_doc": transformed.get("hypothetical_doc")
        }

    except Exception as e:
        print(f"  ⚠️  [QUERY_TRANSFORM] Query transformation failed: {e}")
        print(f"  ℹ️  [QUERY_TRANSFORM] Continuing with original query")
        return state

def retrieve_node(state: LeadsState) -> LeadsState:
    """
    Retrieve relevant documents from vector DB based on entity
    Supports multi-query and HyDE retrieval if query transformation is enabled
    """
    import time
    node_start = time.time()

    query = state["query"]
    entity = state["entity"]
    query_variations = state.get("query_variations")
    hypothetical_doc = state.get("hypothetical_doc")

    print(f"Retrieving docs for entity={entity}")

    filters = {
        "entity": entity
    }

    # Adjust top_k based on whether reranking is enabled
    if settings.ENABLE_RERANKING:
        top_k = settings.INITIAL_RETRIEVAL_K  # Retrieve more for reranking
        print(f"  ℹ️  [RETRIEVE] Reranking enabled - retrieving {top_k} docs per query")
    else:
        top_k = 5  # Default retrieval without reranking

    all_results = []
    seen_ids = set()

    # Strategy: If query transformation is enabled, search with multiple queries
    if settings.ENABLE_QUERY_TRANSFORMATION and (query_variations or hypothetical_doc):
        print(f"  ℹ️  [RETRIEVE] Multi-query/HyDE retrieval enabled")

        # OPTIMIZATION: Batch all embeddings together
        texts_to_embed = []
        if query_variations:
            texts_to_embed.extend(query_variations)
        if hypothetical_doc:
            texts_to_embed.append(hypothetical_doc)

        # Generate all embeddings in a single batch call
        embed_start = time.time()
        all_embeddings = embedding_service.generate_embeddings(texts_to_embed)
        embed_end = time.time()
        print(f"  ⏱️  [RETRIEVE] Batch embedding ({len(texts_to_embed)} texts): {embed_end - embed_start:.3f}s")

        # Now perform searches with the embeddings
        embedding_idx = 0

        # Search with query variations
        if query_variations:
            for idx, q_var in enumerate(query_variations, 1):
                query_embedding = all_embeddings[embedding_idx]
                embedding_idx += 1

                search_start = time.time()
                results = qdrant_service.search(
                    query_embedding=query_embedding,
                    top_k=top_k // 2 if query_variations and hypothetical_doc else top_k,  # Split budget
                    filters=filters
                )
                search_end = time.time()
                print(f"  ⏱️  [RETRIEVE] Search #{idx}: {search_end - search_start:.3f}s ({len(results)} docs)")

                # Deduplicate by ID
                for result in results:
                    if result['id'] not in seen_ids:
                        seen_ids.add(result['id'])
                        all_results.append(result)

        # Search with hypothetical document
        if hypothetical_doc:
            hyde_embedding = all_embeddings[embedding_idx]

            search_start = time.time()
            hyde_results = qdrant_service.search(
                query_embedding=hyde_embedding,
                top_k=top_k // 2 if query_variations else top_k,
                filters=filters
            )
            search_end = time.time()
            print(f"  ⏱️  [RETRIEVE] HyDE search: {search_end - search_start:.3f}s ({len(hyde_results)} docs)")

            # Deduplicate by ID
            for result in hyde_results:
                if result['id'] not in seen_ids:
                    seen_ids.add(result['id'])
                    all_results.append(result)

    else:
        # Standard single-query retrieval
        embed_start = time.time()
        query_embedding = embedding_service.generate_embedding(query)
        embed_end = time.time()
        print(f"  ⏱️  [RETRIEVE] Embedding generation: {embed_end - embed_start:.3f}s")

        search_start = time.time()
        all_results = qdrant_service.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )
        search_end = time.time()
        print(f"  ⏱️  [RETRIEVE] Vector DB search: {search_end - search_start:.3f}s")

    node_end = time.time()
    print(f"  ⏱️  [RETRIEVE] Total node time: {node_end - node_start:.3f}s")
    print(f"Retrieved {len(all_results)} unique documents")
    return {
        **state,
        "retrieved_docs": all_results
    }

def rerank_node(state: LeadsState) -> LeadsState:
    """
    Rerank retrieved documents using Cohere or fallback method
    """
    import time
    node_start = time.time()

    query = state["query"]
    retrieved_docs = state["retrieved_docs"]

    if not retrieved_docs:
        print("  ⚠️  [RERANK] No documents to rerank")
        return state

    print(f"  ℹ️  [RERANK] Reranking {len(retrieved_docs)} documents")

    try:
        # Get reranker service
        reranker = get_reranker_service(
            api_key=settings.COHERE_API_KEY if hasattr(settings, 'COHERE_API_KEY') else None,
            model=settings.RERANK_MODEL if hasattr(settings, 'RERANK_MODEL') else "rerank-english-v3.0"
        )

        # Rerank documents
        reranked_docs = reranker.rerank(
            query=query,
            documents=retrieved_docs,
            top_n=settings.RERANK_TOP_N if hasattr(settings, 'RERANK_TOP_N') else 5
        )

        node_end = time.time()
        print(f"  ⏱️  [RERANK] Total node time: {node_end - node_start:.3f}s")
        print(f"  ✓ Reranked to top {len(reranked_docs)} documents")

        return {
            **state,
            "retrieved_docs": reranked_docs
        }

    except Exception as e:
        print(f"  ⚠️  [RERANK] Reranking failed: {e}")
        print(f"  ℹ️  [RERANK] Falling back to original documents")
        # Fallback: use original top-5 documents
        fallback_docs = retrieved_docs[:5]
        return {
            **state,
            "retrieved_docs": fallback_docs
        }

def expand_context_node(state: LeadsState) -> LeadsState:
    """
    Expand retrieved documents with sentence window context
    """
    import time
    node_start = time.time()

    retrieved_docs = state["retrieved_docs"]
    entity = state["entity"]

    if not retrieved_docs:
        print("  ⚠️  [EXPAND_CONTEXT] No documents to expand")
        return state

    print(f"  ℹ️  [EXPAND_CONTEXT] Expanding context for {len(retrieved_docs)} documents")

    try:
        # Get sentence window service
        sw_service = get_sentence_window_service(
            window_size=settings.SENTENCE_WINDOW_SIZE if hasattr(settings, 'SENTENCE_WINDOW_SIZE') else 3
        )

        # Prepare filters for fetching related chunks
        filters = {
            "entity": entity
        }

        # Expand context
        expanded_docs = sw_service.expand_context(
            documents=retrieved_docs,
            qdrant_service=qdrant_service,
            filters=filters
        )

        node_end = time.time()
        print(f"  ⏱️  [EXPAND_CONTEXT] Total node time: {node_end - node_start:.3f}s")

        # Count how many were actually expanded
        expanded_count = sum(1 for doc in expanded_docs if doc.get('metadata', {}).get('context_expanded'))
        if expanded_count > 0:
            print(f"  ✓ Expanded context for {expanded_count} documents")
        else:
            print(f"  ℹ️  No sentence window metadata found (backward compatible mode)")

        return {
            **state,
            "retrieved_docs": expanded_docs
        }

    except Exception as e:
        print(f"  ⚠️  [EXPAND_CONTEXT] Context expansion failed: {e}")
        print(f"  ℹ️  [EXPAND_CONTEXT] Continuing with original documents")
        return state

def generate_node(state: LeadsState) -> LeadsState:
    """
    Generate response using retrieved documents as context
    """
    import time
    node_start = time.time()

    query = state["query"]
    retrieved_docs = state["retrieved_docs"]
    entity = state["entity"]
    entity_name = state.get("entity_name") or entity

    # Get LLM instance from state
    llm = state.get("llm_instance")
    if not llm:
        print("  ⚠️  [GENERATE] No LLM instance in state, using default")
        llm = get_llm_instance()

    # Build context from retrieved documents
    context_start = time.time()
    if retrieved_docs:
        context_parts = []
        for idx, doc in enumerate(retrieved_docs, 1):
            context_parts.append(
                f"Document {idx} (Score: {doc['score']:.3f}):\n"
                f"Source: {doc['metadata'].get('url', 'N/A')}\n"
                f"Content: {doc['text']}\n"
            )
        context = "\n---\n".join(context_parts)
    else:
        context = "No relevant information found in the knowledge base."
    context_end = time.time()
    print(f"  ⏱️  [GENERATE] Context building: {context_end - context_start:.3f}s")

    # Create system prompt
    prompt_start = time.time()
    system_prompt = f"""You are a representative from {entity_name} communicating via WhatsApp with prospective students and families.

Communication Style:
- Respond naturally and conversationally, as if you're a staff member from the institution
- Keep messages concise and WhatsApp-friendly (avoid long paragraphs)
- Use a warm, welcoming, and professional tone
- Never mention AI, context, documents, or data sources - just respond as the institution

When You Have Information:
- Provide clear, accurate answers directly
- Share relevant details about programs, admissions, facilities, fees, etc.
- Be specific with numbers, dates, and requirements when available

When You Don't Have Information:
- guide them: "For that information, please call our office"
- Never say things like "based on the provided context" or "I don't have that in my database"

Important:
- Respond as {entity_name} itself, not as an assistant or bot
- Keep the conversation natural and human-like
- If asked about sensitive matters (admissions decisions, personal records), direct them to contact the appropriate office directly"""

    # Create user prompt with context
    user_prompt = f"""Context from knowledge base:
{context}

Question: {query}

Please provide a helpful answer based on the context above. This will be whatsapped to the user so keep it concise and clear."""

    # Generate response
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    prompt_end = time.time()
    print(f"  ⏱️  [GENERATE] Prompt creation: {prompt_end - prompt_start:.3f}s")

    llm_start = time.time()
    response = llm.invoke(messages)
    answer = response.content
    llm_end = time.time()
    print(f"  ⏱️  [GENERATE] LLM API call: {llm_end - llm_start:.3f}s")

    print(f"Generated answer: {answer[:200]}...")

    # Update messages
    message_start = time.time()
    updated_messages = state["messages"] + [
        HumanMessage(content=query),
        AIMessage(content=answer)
    ]
    message_end = time.time()
    print(f"  ⏱️  [GENERATE] Message update: {message_end - message_start:.3f}s")

    node_end = time.time()
    print(f"  ⏱️  [GENERATE] Total node time: {node_end - node_start:.3f}s")

    return {
        **state,
        "messages": updated_messages,
        "answer": answer
    }

def build_leads_graph():
    """
    Build and compile the leads chat graph

    Flow (with enhancements):
    START → [query_transform] → retrieve → [rerank] → [expand_context] → generate → END

    Nodes in brackets are optional based on config flags
    """
    # Create graph
    workflow = StateGraph(LeadsState)

    # Add core nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    # Conditionally add query transformation node
    if settings.ENABLE_QUERY_TRANSFORMATION:
        workflow.add_node("query_transform", query_transform_node)
        print("[+] RAG Enhancement: Query transformation (multi-query + HyDE) enabled")

    # Conditionally add reranking node
    if settings.ENABLE_RERANKING:
        workflow.add_node("rerank", rerank_node)
        print("[+] RAG Enhancement: Reranking enabled")

    # Conditionally add sentence window expansion node
    if settings.ENABLE_SENTENCE_WINDOW:
        workflow.add_node("expand_context", expand_context_node)
        print("[+] RAG Enhancement: Sentence window expansion enabled")

    # Define edges based on enabled features
    # 1. START → query_transform (if enabled) OR retrieve
    if settings.ENABLE_QUERY_TRANSFORMATION:
        workflow.add_edge(START, "query_transform")
        workflow.add_edge("query_transform", "retrieve")
    else:
        workflow.add_edge(START, "retrieve")

    # 2. retrieve → rerank (if enabled) OR next step
    if settings.ENABLE_RERANKING:
        workflow.add_edge("retrieve", "rerank")
        last_node = "rerank"
    else:
        last_node = "retrieve"

    # 3. Last node → expand_context (if enabled) OR generate
    if settings.ENABLE_SENTENCE_WINDOW:
        workflow.add_edge(last_node, "expand_context")
        workflow.add_edge("expand_context", "generate")
    else:
        workflow.add_edge(last_node, "generate")

    # 4. generate → END
    workflow.add_edge("generate", END)

    # Compile
    graph = workflow.compile()

    return graph

# Create global graph instance
leads_graph = build_leads_graph()
