import torch
import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

vector_store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
branch_store = FAISS.load_local(
    "faiss_branch", embeddings, allow_dangerous_deserialization=True
)
product_store = FAISS.load_local( 
    "faiss_product", embeddings, allow_dangerous_deserialization=True
)


PROJECT_ID = "ai-last-project-2024"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)

from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model="gemini-1.5-pro")

graph_builder = StateGraph(MessagesState)

keywords_df = pd.read_csv("../data/classifier.csv")

keyword_dict = {}
for _, row in keywords_df.iterrows():
    keyword = row['text'].lower().strip()
    label = row['label'].strip()

    if label not in keyword_dict:
        keyword_dict[label] = set()
    keyword_dict[label].add(keyword)
def classify_query(query: str):
    query_lower = query.lower()

    for label, keywords in keyword_dict.items():
        if any(keyword in query_lower for keyword in keywords):
            return label

    return "General_Request"


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Classify the query using keyword matching."""
    classification = classify_query(query)

    if classification == "branch_Request":
        retrieved_docs = branch_store.similarity_search(query, k=5)
    elif classification == "product_Request":
        retrieved_docs = product_store.similarity_search(query, k=5)
    else:
        retrieved_docs = vector_store.similarity_search(query, k=5)

    # Serialize the results
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    # return state['messages']
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])

# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an mongolian assistant for question-answering tasks. "
        "the queries will be in mongolian"
        "if you don't know the answer, use the tools once."
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise, use mongolian, and always give the resource from metadata as answer if available"
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}


memory = MemorySaver()
config = {"configurable": {"thread_id": "gemini000"}}

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile(checkpointer=memory)

input_message = "төв салбар хаана байрладаг вэ"
def invoke_gemini(input_message: str):
    results = graph.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config = config
    )
    for item in results:
        last_item = item
    return last_item












