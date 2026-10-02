import os


from  dataclasses import dataclass,field
from  pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Settings:
    root:Path = ROOT
    neo4j_url:str = "bolt://localhost:7687"
    neo4j_user:str = "neo4j"
    neo4j_password:str = field(default="",repr=False)
    neo4j_database:str = "neo4j"
    project:str = "QAsystem_v1"
    ollama_host:str = "http://localhost:11434"
    llm:str = "qwen:1.8b"
    entity_mode:str = "rule"


    @property
    def lexicon(self):
        return self.root / "data" / "lexicon.json"
    @property
    def checkpoint(self):
        return self.root / "models" / "ner.pt"
    @property
    def base_model(self):
        return self.root / "models" / "chinese-roberta-wwm-ext"
    @property
    def accounts(self):
        return self.root / "runtime" / "accounts.sqlite3"
def load_settings():
    value = Settings(
        neo4j_url=os.getenv("NEW_NEO4J_URL", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEW_NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEW_NEO4J_PASSWORD", "020822whlol"),
        neo4j_database=os.getenv("NEW_NEO4J_DATABASE", "neo4j"),
        project=os.getenv("NEW_PROJECT_ID", "QAsystem_v1"),
        ollama_host=os.getenv("NEW_OLLAMA_HOST", "http://localhost:11434"),
        llm=os.getenv("NEW_LLM", "qwen:1.8b"),
        entity_mode=os.getenv("NEW_ENTITY_MODE", "rule"),
    )
    if not value.project.strip() or value.entity_mode not in {"rule","hybrid"}:
        raise ValueError("项目编号不能为空，NEW_ENTITY_MODE 必须是rule或者hybrid")
    return value
