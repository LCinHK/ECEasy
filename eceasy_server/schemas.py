from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class QueryRequest(BaseModel):
    query: str
    search_uuid: str
    generate_related_questions: Optional[bool] = True
    llm_provider: Optional[str] = None
    api_key: Optional[str] = None
    use_server_key: Optional[bool] = None
    llm_model: Optional[str] = None
    conversation_history: List[ConversationTurn] = Field(default_factory=list, max_length=40)
    memory_turns: int = Field(default=3, ge=0, le=15)


class ImageSuggestion(BaseModel):
    path: str
    description: str
    doc_type: str
    source_relpath: str


class ChatResponse(BaseModel):
    text: Optional[str] = None
    contexts: Optional[List[dict]] = None
    related_questions: Optional[List[str]] = None
    suggested_images: Optional[List[ImageSuggestion]] = None
    flowchart: Optional[str] = None
