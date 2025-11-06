from typing import TypedDict, List, Dict, Any, Annotated, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, RemoveMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
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
    query_type: Optional[str]  # "general", "chat_history", "rag"
    confidence_score: Optional[float]  # 0.0 to 1.0
    query_variations: Optional[List[str]]
    hypothetical_doc: Optional[str]
    retrieved_docs: List[Dict[str, Any]]
    answer: str
    needs_rag: Optional[bool]  # Whether RAG retrieval is needed
    model_name: Optional[str]  # Model name from chatBotConfig
    llm_instance: Optional[Any]  # LLM instance created from model_name

def classify_query_node(state: LeadsState) -> LeadsState:
    """
    Classify query to determine if RAG retrieval is needed.
    Checks if query is:
    1. General query (greeting, thanks, small talk) - confidence > 0.8
    2. Can be answered from chat history - confidence > 0.8
    3. Needs RAG retrieval - confidence < 0.8
    """
    import time
    node_start = time.time()

    query = state["query"]
    messages = state.get("messages", [])  # Default to empty if not in state
    llm = state.get("llm_instance")

    if not llm:
        llm = get_llm_instance()

    print(f"  ℹ️  [CLASSIFY] Classifying query: {query[:50]}...")

    # Build conversation context for classification
    chat_context = ""
    if messages:
        recent_messages = messages[-6:]  # Last 3 exchanges
        for msg in recent_messages:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            chat_context += f"{role}: {msg.content}\n"

    classification_prompt = f"""Analyze this query and determine its type and confidence score.

Query: "{query}"

Recent conversation:
{chat_context if chat_context else "No previous conversation"}

Classify as ONE of:
1. GENERAL - Greetings, thanks, small talk, yes/no responses (e.g., "hi", "thanks", "okay", "yes", "hello")
2. CHAT_HISTORY - Can be answered from the conversation history above (e.g., "what did you say?", "repeat that", "tell me more about what you mentioned")
3. RAG_NEEDED - Requires looking up specific information (e.g., "what are the fees?", "admission requirements", "courses offered")

Respond in this EXACT format:
TYPE: [GENERAL/CHAT_HISTORY/RAG_NEEDED]
CONFIDENCE: [0.0-1.0]
REASON: [brief explanation]"""

    try:
        response = llm.invoke([{"role": "user", "content": classification_prompt}])
        classification_text = response.content

        # Parse classification
        query_type = "rag"  # default
        confidence = 0.5  # default
        needs_rag = True

        if "TYPE:" in classification_text:
            type_line = [line for line in classification_text.split('\n') if 'TYPE:' in line][0]
            if "GENERAL" in type_line.upper():
                query_type = "general"
            elif "CHAT_HISTORY" in type_line.upper():
                query_type = "chat_history"
            else:
                query_type = "rag"

        if "CONFIDENCE:" in classification_text:
            conf_line = [line for line in classification_text.split('\n') if 'CONFIDENCE:' in line][0]
            try:
                confidence = float(conf_line.split(':')[1].strip())
            except:
                confidence = 0.5

        # Determine if RAG is needed
        if query_type in ["general", "chat_history"] and confidence > 0.8:
            needs_rag = False

        node_end = time.time()
        print(f"  ✓ [CLASSIFY] Type: {query_type}, Confidence: {confidence:.2f}, Needs RAG: {needs_rag}")
        print(f"  ⏱️  [CLASSIFY] Total node time: {node_end - node_start:.3f}s")

        return {
            **state,
            "query_type": query_type,
            "confidence_score": confidence,
            "needs_rag": needs_rag
        }

    except Exception as e:
        print(f"  ⚠️  [CLASSIFY] Classification failed: {e}, defaulting to RAG")
        return {
            **state,
            "query_type": "rag",
            "confidence_score": 0.5,
            "needs_rag": True
        }

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

def trim_messages_sliding_window(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Trim messages using sliding window technique.
    Keeps only the last N messages based on SLIDING_WINDOW_SIZE.

    Args:
        messages: List of messages to trim

    Returns:
        Trimmed list of messages and list of messages to remove
    """
    if not settings.ENABLE_CHAT_MEMORY:
        return messages

    window_size = settings.SLIDING_WINDOW_SIZE

    # If messages are within window size, return as is
    if len(messages) <= window_size:
        return messages

    # Keep only last N messages
    print(f"  📊 [MEMORY] Trimming messages: {len(messages)} -> {window_size} (sliding window)")
    return messages[-window_size:]

def generate_node(state: LeadsState) -> LeadsState:
    """
    Generate response based on query type and context
    """
    import time
    node_start = time.time()

    query = state["query"]
    retrieved_docs = state["retrieved_docs"]
    entity = state["entity"]
    entity_name = state.get("entity_name") or entity
    query_type = state.get("query_type", "rag")
    needs_rag = state.get("needs_rag", True)

    # Get LLM instance from state
    llm = state.get("llm_instance")
    if not llm:
        print("  ⚠️  [GENERATE] No LLM instance in state, using default")
        llm = get_llm_instance()

    # Trim messages using sliding window before adding new ones
    current_messages = state.get("messages", [])  # Default to empty if not in state
    if settings.ENABLE_CHAT_MEMORY and len(current_messages) > 0:
        current_messages = trim_messages_sliding_window(current_messages)

    # Build context from retrieved documents (only if RAG was performed)
    context_start = time.time()
    context = ""
    if needs_rag and retrieved_docs:
        context_parts = []
        for idx, doc in enumerate(retrieved_docs, 1):
            context_parts.append(
                f"Document {idx}:\n{doc['text']}\n"
            )
        context = "\n---\n".join(context_parts)
    context_end = time.time()
    print(f"  ⏱️  [GENERATE] Context building: {context_end - context_start:.3f}s")

    # Create system prompt based on query type
    prompt_start = time.time()

    # Different prompts for different query types
    if query_type == "general":
        system_prompt = f"""You are a friendly representative from {entity_name}.
Keep responses brief, warm, and natural. This is a WhatsApp conversation."""

    elif query_type == "chat_history":
        system_prompt = f"""You are a representative from {entity_name}.
Answer based on what was already discussed in this conversation.
Keep responses concise and conversational."""

    else:  # RAG query
        system_prompt = f"""You are a representative from {entity_name}.

CRITICAL RULES:
1. ONLY use information from the Knowledge Base below
2. If information is NOT in the Knowledge Base, say: "I don't have that information. Please contact our office at [office number/email] for details."
3. NEVER make up names, numbers, or details
4. NEVER use placeholders like [Name], [Number], [Details]
5. Be brief - 2-3 sentences maximum
6. Don't end with "contact us" or "feel free to ask" unless no information found

Knowledge Base:
{context if context else "No relevant information available."}

If Knowledge Base is empty or doesn't answer the question, be honest and direct them to contact the office."""

    # Convert conversation history to LLM format
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history (if exists)
    for msg in current_messages:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})

    # Add current query
    messages.append({"role": "user", "content": query})

    prompt_end = time.time()
    print(f"  ⏱️  [GENERATE] Prompt creation: {prompt_end - prompt_start:.3f}s")
    print(f"  📝 [GENERATE] Query type: {query_type}, Using {len(current_messages)} previous messages")

    llm_start = time.time()
    response = llm.invoke(messages)
    answer = response.content
    llm_end = time.time()
    print(f"  ⏱️  [GENERATE] LLM API call: {llm_end - llm_start:.3f}s")

    # Validate response for hallucinations
    if query_type == "rag" and ('[' in answer or 'Name of' in answer or 'please contact our office directly for accurate information' in answer.lower()):
        print(f"  ⚠️  [GENERATE] Possible hallucination detected, cleaning response...")
        # Remove placeholder patterns
        import re
        answer = re.sub(r'\[.*?\]', '', answer)
        if not answer.strip() or len(answer.strip()) < 20:
            answer = "I don't have specific information about that. Please contact our office for accurate details."

    print(f"Generated answer: {answer[:200]}...")

    # Update messages with trimmed history
    message_start = time.time()
    updated_messages = current_messages + [
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

def get_postgres_checkpointer():
    """
    Initialize PostgresSaver for chat memory persistence.
    Creates connection pool and sets up checkpoint tables.
    """
    if not settings.ENABLE_CHAT_MEMORY:
        print("[!] Chat memory disabled in settings")
        return None

    try:
        from psycopg_pool import ConnectionPool
        import socket

        # Create PostgreSQL connection
        connection_string = settings.POSTGRES_URL

        print(f"[+] Initializing PostgreSQL checkpointer")
        print(f"    Connection: {connection_string.split('@')[-1] if '@' in connection_string else 'local'}")

        # Test DNS resolution first
        # try:
        #     hostname = connection_string.split('@')[1].split(':')[0]
        #     print(f"[+] Testing DNS resolution for: {hostname}")
        #     ip = socket.gethostbyname(hostname)
        #     print(f"[+] DNS resolved to: {ip}")
        # except socket.gaierror as dns_error:
        #     print(f"[!] DNS resolution failed: {dns_error}")
        #     print(f"[!] Check your internet connection and DNS settings")
        #     print(f"[!] Continuing without chat memory (stateless mode)")
        #     return None

        # Create connection pool with timeout and lazy opening
        pool = ConnectionPool(
            conninfo=connection_string,
            min_size=1,  # Minimum connections
            max_size=20,  # Maximum connections
            open=False,  # Don't open connections immediately
            timeout=10.0,  # Connection timeout
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "connect_timeout": 10,
            }
        )

        # Open the pool (creates initial connections)
        print(f"[+] Opening connection pool...")
        pool.open()

        # Create checkpointer with pool
        checkpointer = PostgresSaver(pool)

        # Setup tables (creates if not exists)
        print(f"[+] Setting up checkpoint tables...")
        with pool.connection() as conn:
            checkpointer.setup()

        print(f"[✓] PostgreSQL checkpointer initialized successfully")
        print(f"[+] Chat Memory: Sliding window enabled (last {settings.SLIDING_WINDOW_SIZE} messages)")

        return checkpointer

    except Exception as e:
        print(f"[!] Failed to initialize PostgreSQL checkpointer: {e}")
        print(f"[!] Continuing without chat memory (stateless mode)")
        import traceback
        traceback.print_exc()
        return None

def route_after_classification(state: LeadsState) -> str:
    """
    Route based on query classification.
    If needs_rag is False, skip directly to generate.
    Otherwise, proceed with RAG pipeline.
    """
    needs_rag = state.get("needs_rag", True)
    if needs_rag:
        # Go through RAG pipeline
        if settings.ENABLE_QUERY_TRANSFORMATION:
            return "query_transform"
        else:
            return "retrieve"
    else:
        # Skip RAG, go directly to generate
        return "generate"

def build_leads_graph():
    """
    Build and compile the leads chat graph

    Flow (with query classification):
    START → classify →
        if needs_rag: [query_transform] → retrieve → [rerank] → [expand_context] → generate → END
        else: generate → END

    Nodes in brackets are optional based on config flags
    """
    # Create graph
    workflow = StateGraph(LeadsState)

    # Add query classification node (first step)
    workflow.add_node("classify", classify_query_node)
    print("[+] Query Classification: Enabled (detects general/chat queries)")

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
    # 1. START → classify (always first)
    workflow.add_edge(START, "classify")

    # 2. classify → conditional routing
    workflow.add_conditional_edges(
        "classify",
        route_after_classification,
        {
            "query_transform": "query_transform",
            "retrieve": "retrieve",
            "generate": "generate"
        }
    )

    # 3. query_transform → retrieve (if query transformation enabled)
    if settings.ENABLE_QUERY_TRANSFORMATION:
        workflow.add_edge("query_transform", "retrieve")

    # 4. retrieve → rerank (if enabled) OR next step
    if settings.ENABLE_RERANKING:
        workflow.add_edge("retrieve", "rerank")
        last_node = "rerank"
    else:
        last_node = "retrieve"

    # 5. Last node → expand_context (if enabled) OR generate
    if settings.ENABLE_SENTENCE_WINDOW:
        workflow.add_edge(last_node, "expand_context")
        workflow.add_edge("expand_context", "generate")
    else:
        workflow.add_edge(last_node, "generate")

    # 6. generate → END (always)
    workflow.add_edge("generate", END)

    # Initialize checkpointer if chat memory is enabled
    checkpointer = get_postgres_checkpointer()

    # Compile with or without checkpointer
    if checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)
    else:
        graph = workflow.compile()

    return graph

# Create global graph instance
leads_graph = build_leads_graph()
