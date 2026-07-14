from vanna.core.user import UserResolver, User, RequestContext
from vanna.tools import RunSqlTool
from vanna.integrations.postgres import PostgresRunner
from vanna.integrations.ollama import OllamaLlmService
from vanna.tools.agent_memory import SaveQuestionToolArgsTool, SearchSavedCorrectToolUsesTool, SaveTextMemoryTool
from vanna.integrations.chromadb import ChromaAgentMemory
from vanna.core.registry import ToolRegistry
from vanna import Agent
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# Ollama host (shared between LLM and embeddings)
OLLAMA_HOST = "http://192.168.19.21:11434"
LLAMA3 = "llama3.3:70b-instruct-q2_K"
QWEN = "qwen2.5-coder:32b"

## LLM connection
llm = OllamaLlmService(
    model=QWEN,
    host=OLLAMA_HOST
)

## Embedding function (nomic-embed-text via Ollama)
embedding_fn = OllamaEmbeddingFunction(
    url=OLLAMA_HOST,
    model_name="nomic-embed-text"
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
        host='localhost',
        port=5432,
        database='dvdrental',
        user='postgres',
        password='MrigajSuman@2015'
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
    collection_name="vanna_memory",
    persist_directory="./chroma_db",
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

agent = Agent(
    llm_service=llm,
    tool_registry=tools,
    agent_memory=agent_memory,
    user_resolver=user_resolver
)

