from typing import Dict, TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
import json

load_dotenv()

# Initialize LLM with Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.68.114:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3")

llm = OllamaLLM(
    base_url=OLLAMA_BASE_URL,
    model=MODEL_NAME,
    temperature=0.75,
)

# Initialize embeddings with Ollama
embeddings = OllamaEmbeddings(
    base_url=OLLAMA_BASE_URL,
    model=MODEL_NAME,
)

# Initialize vector store
CHROMA_PATH = "./chroma_db"
if not os.path.exists(CHROMA_PATH):
    os.makedirs(CHROMA_PATH)

vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
    collection_name="rag_collection"
)

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    current_step: str
    context: str
    research_summary: str
    final_answer: str
    agent_outputs: Dict

def retriever_agent(state: AgentState) -> AgentState:
    """Retriever agent that finds relevant documents."""
    print("\n=== Retriever Agent ===")
    print(f"Input state: {state}")
    
    messages = state["messages"]
    query = messages[-1].content
    print(f"Processing query: {query}")
    
    # Get relevant documents
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    print(f"Found {len(docs)} documents")
    
    # Initialize agent outputs if not present
    if "agent_outputs" not in state:
        state["agent_outputs"] = {}
    
    # Format markdown response
    response = {
        "retriever": f"""<details open>
<summary>### 📚 Retrieved Documents</summary>

Found {len(docs)} relevant documents.

<details>
<summary>Document Content</summary>

```
{context}
```

</details>

</details>
"""
    }
    
    # Update state while preserving existing fields
    state["current_step"] = "researcher"
    state["context"] = context
    state["query"] = query
    state["agent_outputs"]["retriever"] = response["retriever"]
    
    print(f"Output state: {state}")
    return state

def researcher_agent(state: AgentState) -> AgentState:
    """Researcher agent that analyzes context and query."""
    print("\n=== Researcher Agent ===")
    print(f"Input state: {state}")
    
    query = state.get("query")
    if not query:
        query = state["messages"][-1].content
        state["query"] = query
    
    context = state.get("context", "")
    
    # Analyze context and query
    analysis = llm.invoke(
        f"Analyze this query and context, identifying key points and relationships:\n\nQuery: {query}\n\nContext: {context}"
    )
    print(f"Generated analysis: {analysis}")
    
    # Format markdown response
    response = {
        "research": f"""<details open>
<summary>### 🔍 Research Analysis</summary>

{analysis}

<details>
<summary>Analysis Details</summary>

- Query: {query}
- Context Analysis: Present
- Key Points Identified
- Relationships Mapped

</details>

</details>
"""
    }
    
    # Update state while preserving existing fields
    state["current_step"] = "writer"
    state["analysis"] = analysis
    state["agent_outputs"]["research"] = response["research"]
    
    print(f"Output state: {state}")
    return state

def writer_agent(state: AgentState) -> AgentState:
    """Writer agent that drafts the initial response."""
    print("\n=== Writer Agent ===")
    print(f"Input state: {state}")
    
    query = state.get("query")
    if not query:
        query = state["messages"][-1].content
        state["query"] = query
    
    context = state.get("context", "")
    analysis = state.get("analysis", "")
    
    # Generate initial draft
    draft = llm.invoke(
        f"Draft a comprehensive response that addresses this query using the context and analysis:\n\nQuery: {query}\nContext: {context}\nAnalysis: {analysis}"
    )
    print(f"Generated draft: {draft}")
    
    # Format markdown response
    response = {
        "writer": f"""<details open>
<summary>### ✍️ Initial Draft</summary>

{draft}

<details>
<summary>Writing Process</summary>

1. Analyzed query requirements
2. Incorporated context
3. Applied research insights
4. Structured response

</details>

</details>
"""
    }
    
    # Update state while preserving existing fields
    state["current_step"] = "critic"
    state["draft"] = draft
    state["agent_outputs"]["writer"] = response["writer"]
    
    print(f"Output state: {state}")
    return state

def critic_agent(state: AgentState) -> AgentState:
    """Critic agent that refines and finalizes the response."""
    print("\n=== Critic Agent ===")
    print(f"Input state: {state}")
    
    query = state.get("query")
    if not query:
        query = state["messages"][-1].content
        state["query"] = query
    
    draft = state.get("draft", "")
    context = state.get("context", "")
    analysis = state.get("analysis", "")
    
    # Review and refine
    feedback = llm.invoke(
        f"""Review this draft response and provide specific feedback:
        Query: {query}
        Context: {context}
        Analysis: {analysis}
        Draft: {draft}
        
        Provide feedback on:
        1. Accuracy and factual correctness
        2. Completeness of response
        3. Clarity and structure
        4. Areas for improvement
        """
    )
    print(f"Generated feedback: {feedback}")
    
    final = llm.invoke(
        f"""Create a final polished response incorporating this feedback:
        Original Query: {query}
        Draft: {draft}
        Feedback: {feedback}
        
        Requirements:
        1. Address all feedback points
        2. Maintain clear structure
        3. Ensure completeness
        4. Polish language and flow
        """
    )
    print(f"Generated final response: {final}")
    
    # Format final markdown response with detailed sections
    response = {
        "critic": f"""<details open>
<summary>### 🎯 Review & Refinement</summary>

<details>
<summary>Feedback Summary</summary>

{feedback}

</details>

<details>
<summary>Review Process</summary>

- Evaluated accuracy and completeness
- Assessed clarity and structure
- Identified improvements
- Refined content and flow

</details>

</details>
""",
        "response": f"""<details open>
<summary>### 🎬 Final Response</summary>

{final}

---
*Generated using multi-agent processing with context-aware analysis.*

</details>
"""
    }
    
    # Update state while preserving existing fields
    state["current_step"] = END
    state["feedback"] = feedback
    state["final_response"] = final
    state["agent_outputs"]["critic"] = response["critic"]
    state["agent_outputs"]["response"] = response["response"]
    
    print(f"Final state: {state}")
    return state

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes for each agent
workflow.add_node("retriever", retriever_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)
workflow.add_node("critic", critic_agent)

# Add edges to connect the workflow
workflow.add_edge("retriever", "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "critic")

# Set the entry point
workflow.set_entry_point("retriever")

# Compile the graph
chain = workflow.compile()

if __name__ == "__main__":
    # Test the workflow
    print("\n=== Testing Agent Workflow ===")
    test_query = "What is the latest document in store?"
    result = chain.invoke({
        "messages": [HumanMessage(content=test_query)],
        "current_step": "retriever",
        "context": "",
        "research_summary": "",
        "final_answer": "",
        "agent_outputs": {}
    })
    
    print("\n=== Final Result ===")
    print(json.dumps(result["agent_outputs"], indent=2))
