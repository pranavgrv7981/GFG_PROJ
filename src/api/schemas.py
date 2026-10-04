from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class CommentInput(BaseModel):
    text: str
    platform: Optional[str] = None
    brand: Optional[str] = None

class CommentResult(BaseModel):
    original_text: str
    cleaned_text: str
    sentiment: str
    sentiment_confidence: float
    is_negative: bool
    complaint_cluster: Optional[int] = None
    complaint_category: Optional[str] = None
    platform: Optional[str] = None
    brand: Optional[str] = None

class BatchInput(BaseModel):
    comments: List[CommentInput]

class BatchResult(BaseModel):
    results: List[CommentResult]
    summary: Dict[str, Any]

class InsightsSummary(BaseModel):
    total_comments: int
    positive_count: int
    negative_count: int
    neutral_count: int
    negative_percentage: float
    complaint_categories: Dict[str, int]
    platform_distribution: Dict[str, int]
    brand_distribution: Dict[str, int]

class ModelInfo(BaseModel):
    sentiment_model_loaded: bool
    clustering_model_loaded: bool
    sentiment_model_type: str
    clustering_model_type: str

class HealthResponse(BaseModel):
    status: str
    version: str
