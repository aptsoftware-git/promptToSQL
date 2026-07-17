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
import json
import re
from vanna.core.tool import ToolCall
from vanna.core.agent.config import AgentConfig
from vanna.core.system_prompt import DefaultSystemPromptBuilder

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
DB_SCHEMA = os.environ.get("DB_SCHEMA", "public")
LLM_NUM_CTX = int(os.environ.get("LLM_NUM_CTX", 32768))

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
                # Strip out any hallucinated multi-step conversation turns 
                content = response.content.split("<|im_start|>")[0].strip()
                
                # Find first { and last } within the isolated first turn
                start_idx = content.find('{')
                end_idx = content.rfind('}')
                
                if start_idx != -1 and end_idx != -1:
                    json_str = content[start_idx:end_idx+1]
                    
                    # Fix unquoted tool names (e.g. {"name": run_sql} -> {"name": "run_sql"})
                    json_str = re.sub(r'("name"\s*:\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*[,}])', r'\1"\2"\3', json_str)
                    
                    parsed = json.loads(json_str)
                    
                    # Handle both flat format {"sql": "..."} and nested format {"name": "run_sql", "arguments": {"..."}}
                    args = parsed.get("arguments", parsed)
                    tool_name = parsed.get("name", "run_sql")
                    
                    # If it's a flat object without a name, but has sql/query, infer it's run_sql
                    if "name" not in parsed and ("sql" in args or "query" in args):
                        args = {"sql": args.get("sql", args.get("query"))}
                        tool_name = "run_sql"
                        
                    tool_call = ToolCall(id="manual_parse_1", name=tool_name, arguments=args)
                    response.tool_calls = [tool_call]
                    response.content = None
            except Exception as e:
                print(f"JSON Parse Error: {e} | Raw string: {content[start_idx:end_idx+1]}")
        return response

## LLM connection
llm = CoderOllamaLlmService(
    model=QWEN,
    host=OLLAMA_HOST,
    num_ctx=LLM_NUM_CTX,
#    num_gpu=0,  # 0 forces Ollama to load 0 layers into VRAM, strictly using the CPU
    stop=["<|im_start|>", "<|im_end|>"]
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
            # Check if column is datetime and convert to string
            elif df[col].dtype == 'datetime64[ns]' or 'datetime' in str(df[col].dtype):
                df[col] = df[col].astype(str)
                
        # Fill NaN/NaT/pd.NA values with None to prevent Pydantic serialization errors
        import pandas as pd
        df = df.astype(object).where(pd.notnull(df), None)
        
        # Try to automatically save successful queries to memory
        try:
            global agent
            if agent:
                convo = await agent.conversation_store.get_conversation(context.conversation_id, context.user)
                if convo and convo.messages:
                    last_user_msg = next((m.content for m in reversed(convo.messages) if m.role == "user"), None)
                    if last_user_msg:
                        # Extract SQL safely, handling both dict and Pydantic models
                        sql = None
                        if isinstance(args, dict):
                            sql = args.get("sql", args.get("query"))
                        elif hasattr(args, "sql"):
                            sql = args.sql
                            
                        if sql:
                            await context.agent_memory.save_tool_usage(
                                question=last_user_msg,
                                tool_name="run_sql",
                                args={"sql": sql},
                                context=context,
                                success=True
                            )
                            print(f"Auto-saved SQL to memory for question: {last_user_msg}")
        except Exception as e:
            print(f"Failed to auto-save SQL: {e}")
                
        return df

# database connection with correct search path
db_tool = RunSqlTool(
    sql_runner=SafePostgresRunner(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", 5432)),
        database=os.environ.get("DB_NAME", "dvdrental"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", ""),
        options=f"-c search_path={DB_SCHEMA}"
    )
)

class SchemaAwareSystemPromptBuilder(DefaultSystemPromptBuilder):
    async def build_system_prompt(self, user, tools):
        base_prompt = await super().build_system_prompt(user, tools)
        schema_hint = (
            f"\n\nCRITICAL INSTRUCTION: The target database schema is '{DB_SCHEMA}'. "
            f"If you query information_schema, you MUST use WHERE table_schema = '{DB_SCHEMA}'. "
            f"Do NOT query the 'public' schema unless explicitly asked."
        )
        return (base_prompt or "") + schema_hint

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
tools.register_local_tool(SaveQuestionToolArgsTool(), access_groups=['admin', 'user'])
tools.register_local_tool(SearchSavedCorrectToolUsesTool(), access_groups=['admin', 'user'])
tools.register_local_tool(SaveTextMemoryTool(), access_groups=['admin', 'user'])

class SimpleUserResolver(UserResolver):
    async def resolve_user(self, request_context: RequestContext) -> User:
        user_email = request_context.get_cookie('vanna_email') or 'guest@example.com'
        group = 'admin' if user_email == 'admin@example.com' else 'user'
        return User(id=user_email, email=user_email, group_memberships=[group])

user_resolver = SimpleUserResolver()

from vanna.core.filter import ConversationFilter

class RecentTurnsFilter(ConversationFilter):
    """Keeps only the system prompt and the last N messages to prevent context overflow from huge SQL results."""
    def __init__(self, keep_last_n: int = 30):
        self.keep_last_n = keep_last_n

    async def filter_messages(self, messages):
        if len(messages) <= self.keep_last_n + 1:
            return messages
            
        filtered = []
        # Always preserve the system prompt (which contains the schema injection)
        if messages and getattr(messages[0], "role", None) == "system":
            filtered.append(messages[0])
            
        # Append only the most recent N messages
        filtered.extend(messages[-self.keep_last_n:])
        return filtered

custom_config = AgentConfig(stream_responses=False)

# Initialize Agent with our custom components
agent = Agent(
    llm_service=llm,
    tool_registry=tools,
    user_resolver=user_resolver,
    agent_memory=agent_memory,
    config=custom_config,
    system_prompt_builder=SchemaAwareSystemPromptBuilder(),
    conversation_filters=[RecentTurnsFilter(keep_last_n=30)],
)
