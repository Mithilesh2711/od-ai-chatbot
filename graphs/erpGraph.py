from typing import Annotated, List, Dict, Any, Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from tools.erpTools import get_all_tools, get_tools_for_user_type
from tools.toolExecutor import make_tool_call
from config.settings import OPENAI_MODEL
import json

# LangGraph State definition
class State(BaseModel):
    messages: Annotated[list, add_messages]
    session_id: Optional[str] = None
    tool_results: Optional[List[Dict[str, Any]]] = None
    user_data: Optional[Dict[str, Any]] = None
    chat_config: Optional[Dict[str, Any]] = None
    from_number: Optional[str] = None
    to_number: Optional[str] = None

# Initialize LLM with tools filtered by user type
def get_llm_with_tools(user_type: str = None):
    """Initialize and return LLM with tools bound based on user type"""
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.7
    )

    # Get tools filtered by user type
    if user_type:
        tools = get_tools_for_user_type(user_type)
    else:
        tools = get_all_tools()

    if tools:
        return llm.bind_tools(tools)
    else:
        return llm  # Return LLM without tools if no tools available

# LangGraph nodes
async def chatbot_node(state: State) -> State:
    """
    Main chatbot node that processes messages and may call tools.
    Filters tools based on user type and handles low confidence scenarios.
    """
    # Get the last message
    last_message = state.messages[-1] if state.messages else None

    if not last_message:
        return state

    # Get user type from state
    user_type = state.user_data.get("userType") if state.user_data else None

    # Add system message to guide tool selection based on user type
    system_msg = SystemMessage(content=f"""You are a helpful AI assistant for an educational institution.

    Current user type: {user_type}

    IMPORTANT RULES:
    1. You can ONLY use tools that are appropriate for the user type:
       - If user is a STUDENT: Only use tools starting with 'stud' (studFeeDues, studAttToday, etc.)
       - If user is a USER (admin/staff): Only use tools starting with 'user' (userFeeToday, userAdmOverall, etc.)

    2. CONFIDENCE THRESHOLD:
       - Above 70% confident: Call tool directly
       - 50-70% confident: Ask for confirmation first
       - Below 50% confident: Provide short options (under 20 words)

    3. For general queries, greetings, or non-tool conversations:
       - Be polite, helpful, and conversational
       - Keep responses relevant to educational context (fees, attendance, admissions)
       - Respond naturally but stay within your domain
       - Greetings and basic pleasantries are allowed
    """)

    # Create messages with system prompt
    messages_with_context = [system_msg] + state.messages

    # Get LLM with filtered tools based on user type
    llm_with_tools = get_llm_with_tools(user_type)

    # First, check confidence with a preliminary query
    confidence_check_llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.1
    )

    confidence_prompt = f"""Analyze this query and determine:
    1. What is the user asking for?
    2. Which tool would be most appropriate?
    3. How confident are you (0-1 scale)?

    User type: {user_type}
    Query: {last_message.content if hasattr(last_message, 'content') else str(last_message)}

    Available tools for this user:
    {', '.join([tool.name for tool in get_tools_for_user_type(user_type)])}

    Respond in JSON format:
    {{"intent": "brief action description (under 10 words)", "suggested_tool": "tool_name or null", "confidence": 0.0-1.0}}
    """

    confidence_check = await confidence_check_llm.ainvoke([SystemMessage(content=confidence_prompt)])

    try:
        # Parse confidence response
        import re
        json_match = re.search(r'\{.*\}', confidence_check.content, re.DOTALL)
        if json_match:
            confidence_data = json.loads(json_match.group())
            confidence_score = float(confidence_data.get("confidence", 0))
        else:
            confidence_score = 0.5  # Default to medium confidence
    except:
        confidence_score = 0.5  # Default if parsing fails

    # Handle confidence-based routing
    if confidence_score > 0.7:
        # High confidence - directly call the tool
        response = await llm_with_tools.ainvoke(messages_with_context)

    elif confidence_score >= 0.5:
        # Medium confidence - provide a natural response instead of confirmation
        context_llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.7
        )

        intent = confidence_data.get("intent", "") if 'confidence_data' in locals() else ""

        # Check if it's a greeting
        greeting_words = ['hi', 'hello', 'hey', 'greet', 'good morning', 'good afternoon', 'good evening']
        is_greeting = any(word in intent.lower() for word in greeting_words) or \
                     any(word in last_message.content.lower() for word in greeting_words)

        if is_greeting:
            # Handle greetings naturally
            context_prompt = SystemMessage(content=f"""You are a helpful educational institution assistant.
            User type: {user_type}

            The user is greeting you. Respond warmly and briefly mention how you can help.
            Keep response under 20 words and conversational.
            Focus on educational services (fees, attendance, admissions).
            """)
        else:
            # Handle other medium confidence cases
            context_prompt = SystemMessage(content=f"""You are a helpful educational institution assistant.
            User type: {user_type}

            The user seems to be asking about: {intent}
            Provide a helpful response or clarification.
            Keep response under 20 words and conversational.
            """)

        messages_for_response = [context_prompt, last_message]
        response = await context_llm.ainvoke(messages_for_response)

        return State(
            messages=[response],
            session_id=state.session_id,
            tool_results=None,
            user_data=state.user_data,
            chat_config=state.chat_config,
            from_number=state.from_number,
            to_number=state.to_number
        )

    else:
        # Low confidence - provide context-aware conversational response
        # Let LLM generate natural response based on context
        context_llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.7
        )

        context_prompt = SystemMessage(content=f"""You are a helpful educational institution assistant.
        User type: {user_type}

        The user asked something but it's unclear. Respond naturally and helpfully:
        - If greeting, respond warmly
        - If unclear about educational services, briefly mention what you can help with
        - Keep response under 20 words and conversational
        - Stay within educational context (fees, attendance, admissions)
        """)

        messages_for_response = [context_prompt, last_message]
        clarification_response = await context_llm.ainvoke(messages_for_response)

        return State(
            messages=[clarification_response],
            session_id=state.session_id,
            tool_results=None,
            user_data=state.user_data,
            chat_config=state.chat_config,
            from_number=state.from_number,
            to_number=state.to_number
        )

    # Continue with normal flow after confidence check
    response = response if 'response' in locals() else await llm_with_tools.ainvoke(messages_with_context)

    # Check if tools were called
    tool_calls = []
    messages_to_add = [response]

    # If no tool calls, ensure response is conversational and helpful
    if not hasattr(response, 'tool_calls') or not response.tool_calls:
        # Generate a natural, context-aware response
        if confidence_score < 0.7 and confidence_score >= 0.5:
            # Already handled in confirmation branch above
            pass
        else:
            # Let the LLM respond naturally for non-tool queries
            pass

    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']

            # Validate tool is appropriate for user type
            allowed_tools = [t.name for t in get_tools_for_user_type(user_type)]

            if tool_name not in allowed_tools:
                # Tool not allowed for this user type
                error_msg = f"Tool '{tool_name}' is not available for {user_type} users."
                tool_calls.append({
                    "tool": tool_name,
                    "args": tool_call.get('args', {}),
                    "result": {"status": "error", "message": error_msg}
                })

                tool_message = ToolMessage(
                    content=json.dumps({"status": "error", "message": error_msg}),
                    tool_call_id=tool_call.get('id', tool_name)
                )
                messages_to_add.append(tool_message)
                continue

            # Make actual API call to the tool endpoint
            if state.user_data and state.from_number and state.to_number:
                result = await make_tool_call(
                    action_id=tool_name,
                    user_data=state.user_data,
                    from_number=state.from_number,
                    to_number=state.to_number,
                    chat_config=state.chat_config
                )

                tool_calls.append({
                    "tool": tool_name,
                    "args": tool_call.get('args', {}),
                    "result": result
                })

                # Add tool result as ToolMessage
                tool_message = ToolMessage(
                    content=json.dumps(result),
                    tool_call_id=tool_call.get('id', tool_name)
                )
                messages_to_add.append(tool_message)

        # If tools were called, get final response from LLM
        if tool_calls:
            # Add all messages including tool results to state
            state_with_tools = State(
                messages=state.messages + messages_to_add,
                session_id=state.session_id,
                user_data=state.user_data,
                chat_config=state.chat_config,
                from_number=state.from_number,
                to_number=state.to_number
            )

            # Get final response after tool execution
            final_response = await llm_with_tools.ainvoke(state_with_tools.messages)
            messages_to_add.append(final_response)

    # Update state with response and tool results
    new_state = State(
        messages=messages_to_add,
        session_id=state.session_id,
        tool_results=tool_calls if tool_calls else None,
        user_data=state.user_data,
        chat_config=state.chat_config,
        from_number=state.from_number,
        to_number=state.to_number
    )

    return new_state

# Build the graph
def build_graph():
    """
    Builds and compiles the LangGraph workflow.
    """
    graph_builder = StateGraph(State)

    # Add nodes
    graph_builder.add_node("chatbot", chatbot_node)

    # Add edges
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    # Compile the graph
    return graph_builder.compile()
