from vanna.core.user import UserResolver, User, RequestContext
from vanna.tools import RunSqlTool
from vanna.integrations.postgres import PostgresRunner
from vanna.integrations.ollama import OllamaLlmService
from vanna.tools.agent_memory import SaveQuestionToolArgsTool, SearchSavedCorrectToolUsesTool, SaveTextMemoryTool
from vanna.integrations.chromadb import ChromaAgentMemory
from vanna.core.registry import ToolRegistry
from vanna import Agent
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Load environment variables
OLLAMA_HOST = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434")
QWEN = os.environ.get("LLM_MODEL", "qwen2.5-coder:32b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "vanna_memory")

import json
import re
from vanna.core.tool import ToolCall
from vanna.core.agent.config import AgentConfig

class CoderOllamaLlmService(OllamaLlmService):
    """Hybrid LLM service that supports both native tool-calling models (e.g. qwen3, llama3.1)
    and JSON-dumping models (e.g. qwen2.5-coder) that lack native tool support.
    
    Native tool calls flow through untouched via the parent OllamaLlmService.
    For models that dump raw JSON in content, we parse it and build a ToolCall manually.
    """
    async def send_request(self, request):
        response = await super().send_request(request)
        # Only attempt JSON parsing if the model didn't already produce native tool calls
        if not response.tool_calls and response.content and "{" in response.content:
            try:
                # Find first { and last } in case there's conversational text around it
                start_idx = response.content.find('{')
                end_idx = response.content.rfind('}')
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response.content[start_idx:end_idx+1]
                    parsed = json.loads(json_str)
                    
                    # Handle both flat format {"sql": "..."} and nested format {"name": "run_sql", "arguments": {"sql": "..."}}
                    args = parsed.get("arguments", parsed)
                    
                    if "sql" in args or "query" in args:
                        sql = args.get("sql", args.get("query"))
                        tool_call = ToolCall(id="manual_parse_1", name="run_sql", arguments={"sql": sql})
                        response.tool_calls = [tool_call]
                        response.content = None
            except Exception:
                pass
        return response

## LLM connection
llm = CoderOllamaLlmService(
    model=QWEN,
    host=OLLAMA_HOST
)

## Embedding function
embedding_fn = OllamaEmbeddingFunction(
    url=OLLAMA_HOST,
    model_name=EMBED_MODEL
)


class SafePostgresRunner(PostgresRunner):
    async def run_sql(self, args, context):
        df = await super().run_sql(args, context)
        
        # Pydantic cannot serialize memoryview objects (like PostgreSQL bytea columns).
        # We replace them with a string placeholder to prevent crashes.
        for col in df.columns:
            # Check if any element in the column is a memoryview
            if any(isinstance(val, memoryview) for val in df[col]):
                df[col] = df[col].apply(lambda x: "<binary_data>" if isinstance(x, memoryview) else x)
                
        return df

#database connection
db_tool = RunSqlTool(
    sql_runner=SafePostgresRunner(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", 5432)),
        database=os.environ.get("DB_NAME", "dvdrental"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "")
    )
)

# Custom memory class to lower the default similarity threshold from 0.7 to 0.4.
# This is necessary because nomic-embed-text distances naturally map to ~0.5-0.6
# which causes Vanna to silently filter out all memories using the 0.7 threshold.
class CustomChromaMemory(ChromaAgentMemory):
    async def search_text_memories(self, query, context, *, limit=10, similarity_threshold=0.4):
        return await super().search_text_memories(query, context, limit=limit, similarity_threshold=similarity_threshold)
        
    async def search_similar_usage(self, question, context, *, limit=10, similarity_threshold=0.4, tool_name_filter=None):
        return await super().search_similar_usage(question, context, limit=limit, similarity_threshold=similarity_threshold, tool_name_filter=tool_name_filter)

# ChromaDB setup with nomic-embed-text embeddings for agent_memory
agent_memory = CustomChromaMemory(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_fn
)

# Register memory tools (they access agent_memory via ToolContext)
tools = ToolRegistry()
tools.register_local_tool(db_tool, access_groups=['admin', 'user'])
tools.register_local_tool(SaveQuestionToolArgsTool(), access_groups=['admin'])
tools.register_local_tool(SearchSavedCorrectToolUsesTool(), access_groups=['admin', 'user'])
tools.register_local_tool(SaveTextMemoryTool(), access_groups=['admin', 'user'])

class SimpleUserResolver(UserResolver):
    async def resolve_user(self, request_context: RequestContext) -> User:
        user_email = request_context.get_cookie('vanna_email') or 'guest@example.com'
        group = 'admin' if user_email == 'admin@example.com' else 'user'
        return User(id=user_email, email=user_email, group_memberships=[group])

user_resolver = SimpleUserResolver()

custom_config = AgentConfig(stream_responses=False)

agent = Agent(
    llm_service=llm,
    tool_registry=tools,
    agent_memory=agent_memory,
    user_resolver=user_resolver,
    config=custom_config
)

