from typing import TypedDict, List, Dict, Any, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from vectorStore.qdrantClient import qdrant_service
from vectorStore.embeddings import embedding_service
from config import settings

# State Definition
class LeadsState(TypedDict):
    """State for leads chat agent"""
    messages: Annotated[List[BaseMessage], add_messages]
    entity: str
    session: str
    query: str
    retrieved_docs: List[Dict[str, Any]]
    answer: str

# Initialize LLM
llm = ChatOpenAI(model=settings.OPENAI_MODEL, temperature=0.7)

def retrieve_node(state: LeadsState) -> LeadsState:
    """
    Retrieve relevant documents from vector DB based on entity and session
    """
    import time
    node_start = time.time()

    query = state["query"]
    entity = state["entity"]
    session = state["session"]

    print(f"Retrieving docs for entity={entity}, session={session}, query={query}")

    # Generate query embedding
    embed_start = time.time()
    query_embedding = embedding_service.generate_embedding(query)
    embed_end = time.time()
    print(f"  ⏱️  [RETRIEVE] Embedding generation: {embed_end - embed_start:.3f}s")

    # Search vector DB with filters
    search_start = time.time()
    filters = {
        "entity": entity,
        "session": session
    }

    results = qdrant_service.search(
        query_embedding=query_embedding,
        top_k=5,
        filters=filters
    )
    search_end = time.time()
    print(f"  ⏱️  [RETRIEVE] Vector DB search: {search_end - search_start:.3f}s")

    node_end = time.time()
    print(f"  ⏱️  [RETRIEVE] Total node time: {node_end - node_start:.3f}s")
    print(f"Retrieved {len(results)} documents")

    return {
        **state,
        "retrieved_docs": results
    }

def generate_node(state: LeadsState) -> LeadsState:
    """
    Generate response using retrieved documents as context
    """
    import time
    node_start = time.time()

    query = state["query"]
    retrieved_docs = state["retrieved_docs"]
    entity = state["entity"]

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
    system_prompt = f"""You are a helpful assistant for {entity}.
Your role is to answer questions about this educational institution based on the provided context.

If the context contains relevant information, use it to provide a detailed and accurate answer.
If the context doesn't contain enough information, acknowledge this and provide what you can.

Always be helpful, professional, and accurate in your responses."""

    # Create user prompt with context
    user_prompt = f"""Context from knowledge base:
{context}

Question: {query}

Please provide a helpful answer based on the context above."""

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

    print(f"Generated answer: {answer[:100]}...")

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

    Flow:
    START → retrieve_node → generate_node → END
    """
    # Create graph
    workflow = StateGraph(LeadsState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    # Define edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # Compile
    graph = workflow.compile()

    return graph

# Create global graph instance
leads_graph = build_leads_graph()
