import json
import polars as pl
import pathway as pw
pw.set_license_key('12403C-5891EB-94B6EC-CC90D1-C3B479-V3')
from pathway.xpacks.llm.embedders import OpenAIEmbedder
from pathway.xpacks.llm.parsers import UnstructuredParser
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.stdlib.indexing.nearest_neighbors import BruteForceKnnFactory
from pathway.xpacks.llm.document_store import DocumentStore
from pathway.xpacks.llm.mcp_server import PathwayMcp

from agno.agent import Agent
from agno.workflow import Workflow, Step, Parallel
from agno.tools.reasoning import ReasoningTools
from agno.models.message import Message
from agno.tools import Tool

from fastmcp import Client
import asyncio
from concurrent.futures import ThreadPoolExecutor

# ==================== CONFIG ====================
GDRIVE_FOLDER_ID = "1Z1Pt3XoF7GAb_QtLksa8q4D_U-wc65e4"
MCP_URL = "http://localhost:8123/mcp/"

# ==================== PATHWAY MCP SERVER ====================
def launch_mcp_server():
    docs = pw.io.gdrive.read(
        object_id=GDRIVE_FOLDER_ID,
        mode="static",
        format="plaintext",
        with_metadata=True
    )

    parser = UnstructuredParser(chunking_mode="basic")
    splitter = TokenCountSplitter(min_tokens=400, max_tokens=1600, encoding_name="cl100k_base")
    embedder = OpenAIEmbedder(model="text-embedding-ada-002")
    retriever = BruteForceKnnFactory(embedder=embedder, k=8)

    store = DocumentStore(
        docs=docs,
        parser=parser,
        splitter=splitter,
        retriever_factory=retriever
    )

    PathwayMcp(
        name="KDSH26_Novel_Store",
        transport="streamable-http",
        host="localhost",
        port=8123,
        serve=[store]
    )
    pw.run()  # Blocks

# Start server in background
ThreadPoolExecutor(max_workers=1).submit(launch_mcp_server)

# ==================== MCP RETRIEVAL TOOL ====================
class NovelRAG(Tool):
    name = "retrieve_novel_passages"
    description = "Retrieve relevant passages from the full novel."

    def __init__(self):
        self.client = Client(MCP_URL)

    async def _call(self, query: str, book_name: str):
        async with self.client:
            args = {
                "query": query,
                "k": 8,
                "metadata_filter": {"name": {"$regex": f".*{book_name.replace(' ', '_')}.*"}}
            }
            result = await self.client.call_tool("retrieve_query", args)
        passages = result.get("result", [])
        return "\n\n---\n\n".join([p["text"] for p in passages]) or "No relevant passages."

    def __call__(self, query: str, book_name: str):
        return asyncio.run(self._call(query, book_name))

rag_tool = NovelRAG()

# ==================== DATA LOADING (Polars) ====================
train_df = pl.read_csv("train.csv")
test_df = pl.read_csv("test.csv")

def build_few_shot() -> list[Message]:
    consistent = train_df.filter(pl.col("label") == "consistent").sample(3)
    contradict = train_df.filter(pl.col("label") == "contradict").sample(3)
    samples = pl.concat([consistent, contradict]).sample(6)

    msgs = []
    for row in samples.iter_rows(named=True):
        msgs.append(Message(
            role="user",
            content=f"Character: {row['char']}\nBackstory: {row['content'][:3000]}"
        ))
        label = 1 if row["label"] == "consistent" else 0
        msgs.append(Message(role="assistant", content=f"Judgment: {label}"))
    return msgs

few_shot = build_few_shot()

# ==================== AGENTS ====================
character_agent = Agent(
    name="CharacterExtractor",
    model="mistral:mistral-large-latest",
    tools=[rag_tool],
    instructions=[
        "Use retrieve_novel_passages with the correct book name. Extract ALL established facts about the central character: traits, relationships, motivations, key life events, family, beliefs."
    ]
)

timeline_agent = Agent(
    name="TimelineExtractor",
    model="mistral:mistral-large-latest",
    tools=[rag_tool],
    instructions=[
        "Use retrieve_novel_passages. Extract major chronological events, causal chains, and outcomes involving or affecting the central character."
    ]
)

world_agent = Agent(
    name="WorldRulesExtractor",
    model="mistral:mistral-large-latest",
    tools=[rag_tool],
    instructions=[
        "Use retrieve_novel_passages. Extract world logic, political context, social rules, and any constraints that govern character behavior and events."
    ]
)

analyzer = Agent(
    name="ConsistencyAnalyzer",
    model="mistral:mistral-large-latest",
    tools=[ReasoningTools(add_instructions=True, add_few_shot=True), rag_tool],
    reasoning=True,
    additional_input=few_shot,
    instructions=[
        "Compare the hypothetical backstory against character facts, timeline, and world rules.",
        "Check for causal plausibility and constraint violations.",
        "Use RAG evidence from multiple sections."
    ]
)

classifier = Agent(
    name="FinalJudge",
    model="mistral:mistral-large-latest",
    reasoning=True,
    instructions=[
        "Output ONLY JSON: {\"judgment\": 1 or 0, \"rationale\": \"1-2 sentence evidence-based explanation\"}",
        "1 = Consistent, 0 = Contradict"
    ]
)

# Evidence aggregator
def aggregate(step_input):
    parts = [
        step_input.get_step_content("CharacterExtractor") or "",
        step_input.get_step_content("TimelineExtractor") or "",
        step_input.get_step_content("WorldRulesExtractor") or "",
        step_input.get_step_content("ConsistencyAnalyzer") or ""
    ]
    synthesis = "\n\n".join([p[:2000] for p in parts if p])
    return {"step_name": "Aggregator", "content": synthesis, "success": True}

# ==================== WORKFLOW ====================
workflow = Workflow(
    name="KDSH26_Consistency",
    steps=[
        Parallel(
            Step("CharacterExtractor", character_agent),
            Step("TimelineExtractor", timeline_agent),
            Step("WorldRulesExtractor", world_agent),
            name="ParallelExtraction"
        ),
        Step("ConsistencyAnalyzer", analyzer),
        Step("Aggregator", executor=aggregate),
        Step("FinalJudge", classifier),
    ]
)

# ==================== PROCESSING ====================
def process(row: dict) -> dict:
    book = row["book_name"]
    char = row["char"]
    backstory = row["content"]

    prompt = f"""Book: {book}
Central Character: {char}
Hypothetical Backstory: {backstory}

All agents: Use retrieve_novel_passages(book_name="{book}") for evidence."""
    
    result = workflow.run(prompt)
    
    try:
        out = json.loads(result.content)
        judgment = out["judgment"]
        rationale = out.get("rationale", "")[:200]
    except:
        judgment = 0
        rationale = "Classification failed"
    
    return {"Story ID": row["id"], "Prediction": judgment, "Rationale": rationale}

# ==================== MAIN ====================
def main():
    # Optional train validation
    if "label" in train_df.columns:
        val = [process(r) for r in train_df.to_dicts()]
        actual = [1 if r["label"] == "consistent" else 0 for r in train_df.to_dicts()]
        acc = sum(v["Prediction"] == a for v, a in zip(val, actual)) / len(val)
        print(f"Train accuracy: {acc:.2%}")

    # Test predictions
    results = [process(r) for r in test_df.to_dicts()]
    pl.DataFrame(results).write_csv("results.csv")
    print("results.csv generated – ready for submission!")

if __name__ == "__main__":
    main()