import os
import json
import polars as pl
import pandas as pd
import pathway as pw
from pathway.xpacks.llm.embedders import LiteLLMEmbedder
from pathway.xpacks.llm.parsers import UnstructuredParser
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.stdlib.indexing.nearest_neighbors import BruteForceKnnFactory
from pathway.xpacks.llm.document_store import DocumentStore

from agno.agent import Agent
from agno.workflow import Workflow, Step, Parallel
from agno.tools.reasoning import ReasoningTools
from agno.models.message import Message
from agno.tools import Toolkit

# ==================== CONFIG ====================
GDRIVE_FOLDER_ID = "1Z1Pt3XoF7GAb_QtLksa8q4D_U-wc65e4"
pw.set_license_key("12403C-5891EB-94B6EC-CC90D1-C3B479-V3")
os.environ['MISTRAL_API_KEY'] = "QVIVwZhP01i2QlrDtQ7U6QU6hFyst4yy"

# ==================== DOCUMENT STORE SETUP ====================
# Using local file system instead of Google Drive to avoid credential issues
docs = pw.io.fs.read(
    path="./novels/Books",
    mode="static",
    format="binary",
    with_metadata=True
)

parser = UnstructuredParser(chunking_mode="basic")
splitter = TokenCountSplitter(min_tokens=400, max_tokens=1600, encoding_name="cl100k_base")
embedder = LiteLLMEmbedder(model="mistral/mistral-embed")
retriever = BruteForceKnnFactory(embedder=embedder)

store = DocumentStore(
    docs=docs,
    parser=parser,
    splitter=splitter,
    retriever_factory=retriever
)

# ==================== DIRECT RETRIEVAL TOOL ====================
class NovelRAG(Toolkit):
    def __init__(self, store):
        self.store = store
        # Register tools in constructor
        super().__init__(name="novel_rag", tools=[self.retrieve_novel_passages])
    
    def retrieve_novel_passages(self, query: str, book_name: str) -> str:
        """Retrieve relevant passages from the full novel directly.
        
        Args:
            query: The query to retrieve passages for.
            book_name: The name of the book to retrieve passages from.
        """
        metadata_filter = {"name": {"$regex": f".*{book_name.replace(' ', '_')}.*"}}
        query_row = {"query": query, "k": 8, "metadata_filter": json.dumps(metadata_filter)}
        query_df = pd.DataFrame([query_row])
        query_table = pw.debug.table_from_pandas(query_df)
        
        retrieved = self.store.retrieve_query(query_table)
        
        # Compute results
        results = pw.debug.compute_and_print(retrieved.select(pw.this.result))
        
        passages = []
        if results:
            for doc in results[0]["result"]:
                passages.append(doc["text"])
        
        return "\n\n---\n\n".join(passages) or "No relevant passages."

rag_tool = NovelRAG(store)

# ==================== DATA LOADING (Polars) ====================
train_df = pl.read_csv("novels/train.csv")
test_df = pl.read_csv("novels/test.csv")

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

# Run Pathway engine
pw.run()