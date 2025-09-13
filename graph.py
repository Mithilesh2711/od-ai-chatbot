from typing import Annotated, List, Dict, Any, Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from tools import get_all_tools, get_tools_for_user_type, make_tool_call
import os
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
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
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
    system_msg = SystemMessage(content=f"""You are an AI assistant for an educational institution's chatbot.

    Current user type: {user_type}

    IMPORTANT RULES:
    1. You can ONLY use tools that are appropriate for the user type:
       - If user is a STUDENT: Only use tools starting with 'stud' (studFeeDues, studAttToday, etc.)
       - If user is a USER (admin/staff): Only use tools starting with 'user' (userFeeToday, userAdmOverall, etc.)

    2. CONFIDENCE THRESHOLD:
       - If you are less than 50% confident about which tool to use, DO NOT call any tool.
       - Instead, ask clarifying questions or suggest relevant options.
       - Example: "I understand you're asking about fees. Could you please clarify:
         • Do you want to check your personal fee dues?
         • Do you want to see your payment history?
         • Do you need information about fee structure?"

    3. If the query doesn't match any available tools for this user type, politely explain what you can help with.
    """)

    # Create messages with system prompt
    messages_with_context = [system_msg] + state.messages

    # Get LLM with filtered tools based on user type
    llm_with_tools = get_llm_with_tools(user_type)

    # First, check confidence with a preliminary query
    confidence_check_llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
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
    {{"intent": "description", "suggested_tool": "tool_name or null", "confidence": 0.0-1.0}}
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

    # If confidence is too low, provide clarification instead of calling tools
    if confidence_score < 0.5:
        # Get available options for this user type
        if user_type == "student":
            options = """I can help you with:
• **Fee Information**: Check your fee dues, payment history
• **Attendance**: View today's attendance, weekly/monthly summary

Please specify what you'd like to know more clearly."""
        elif user_type == "user":
            options = """I can help you with:
• **Fee Reports**: Today's collection, monthly summary, payment modes, dues by course
• **Attendance Reports**: Overall attendance statistics
• **Admission Reports**: Overall summary, course-wise, category-wise data

Please specify which report or information you need."""
        else:
            options = "Please clarify your request so I can assist you better."

        clarification_response = AIMessage(content=f"I'm not entirely sure what specific information you're looking for. {options}")

        return State(
            messages=[clarification_response],
            session_id=state.session_id,
            tool_results=None,
            user_data=state.user_data,
            chat_config=state.chat_config,
            from_number=state.from_number,
            to_number=state.to_number
        )

    # Proceed with normal tool invocation if confidence is high enough
    response = await llm_with_tools.ainvoke(messages_with_context)

    # Check if tools were called
    tool_calls = []
    messages_to_add = [response]

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
                    to_number=state.to_number
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