from typing import List, Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    search_uuid: str
    generate_related_questions: Optional[bool] = True
    llm_provider: Optional[str] = None
    api_key: Optional[str] = None
    use_server_key: Optional[bool] = None
    llm_model: Optional[str] = None


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
