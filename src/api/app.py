from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.api.schemas import (
    CommentInput, CommentResult, BatchInput, BatchResult,
    InsightsSummary, ModelInfo, HealthResponse
)
from src.preprocessing.cleaner import clean_text
from src.sentiment import get_sentiment_analyzer
from src.clustering.clusterer import ComplaintClusterer
from typing import Dict, List, Any

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(
    title="Social Media Brand Sentiment & Complaint Clusterer API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sentiment_analyzer = get_sentiment_analyzer()
complaint_clusterer = ComplaintClusterer()

session_data: List[Dict[str, Any]] = []

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", version="1.0.0")

@app.post("/predict", response_model=CommentResult)
async def predict_single(comment: CommentInput):
    if not comment.text or not comment.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    cleaned_text = clean_text(comment.text)
    
    try:
        sentiment_res = sentiment_analyzer.predict(cleaned_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")
        
    sentiment_label = sentiment_res.get('sentiment', 'neutral')
    confidence = sentiment_res.get('confidence', 0.0)
    is_negative = (sentiment_label.lower() == 'negative')
    
    cluster_id = None
    category = None
    
    if is_negative:
        try:
            cluster_res = complaint_clusterer.predict(cleaned_text)
            cluster_id = cluster_res.get('cluster_id')
            category = cluster_res.get('complaint_category') or cluster_res.get('cluster_label')
        except Exception:
            pass
            
    result = CommentResult(
        original_text=comment.text,
        cleaned_text=cleaned_text,
        sentiment=sentiment_label,
        sentiment_confidence=confidence,
        is_negative=is_negative,
        complaint_cluster=cluster_id,
        complaint_category=category,
        platform=comment.platform,
        brand=comment.brand
    )
    
    session_data.append({
        "result": result,
        "platform": comment.platform,
        "brand": comment.brand
    })
    
    return result

@app.post("/predict/batch", response_model=BatchResult)
async def predict_batch(batch: BatchInput):
    if not batch.comments:
        raise HTTPException(status_code=400, detail="Batch comments cannot be empty")
        
    results = []
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    for comment in batch.comments:
        if not comment.text or not comment.text.strip():
            continue
            
        cleaned_text = clean_text(comment.text)
        
        try:
            sentiment_res = sentiment_analyzer.predict(cleaned_text)
            sentiment_label = sentiment_res.get('sentiment', 'neutral')
            confidence = sentiment_res.get('confidence', 0.0)
        except Exception:
            sentiment_label = 'neutral'
            confidence = 0.0
            
        is_negative = (sentiment_label.lower() == 'negative')
        
        if is_negative:
            negative_count += 1
        elif sentiment_label.lower() == 'positive':
            positive_count += 1
        else:
            neutral_count += 1
            
        cluster_id = None
        category = None
        
        if is_negative:
            try:
                cluster_res = complaint_clusterer.predict(cleaned_text)
                cluster_id = cluster_res.get('cluster_id')
                category = cluster_res.get('complaint_category') or cluster_res.get('cluster_label')
            except Exception:
                pass
                
        result = CommentResult(
            original_text=comment.text,
            cleaned_text=cleaned_text,
            sentiment=sentiment_label,
            sentiment_confidence=confidence,
            is_negative=is_negative,
            complaint_cluster=cluster_id,
            complaint_category=category,
            platform=comment.platform,
            brand=comment.brand
        )
        
        results.append(result)
        
        session_data.append({
            "result": result,
            "platform": comment.platform,
            "brand": comment.brand
        })
        
    summary = {
        "total": len(results),
        "positive": positive_count,
        "negative": negative_count,
        "neutral": neutral_count
    }
    
    return BatchResult(results=results, summary=summary)

@app.get("/insights", response_model=InsightsSummary)
async def get_insights():
    total_comments = len(session_data)
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    complaint_categories = {}
    platform_distribution = {}
    brand_distribution = {}
    
    for item in session_data:
        res = item["result"]
        platform = item["platform"] or "unknown"
        brand = item["brand"] or "unknown"
        
        platform_distribution[platform] = platform_distribution.get(platform, 0) + 1
        brand_distribution[brand] = brand_distribution.get(brand, 0) + 1
        
        if res.is_negative:
            negative_count += 1
            cat = res.complaint_category or "unknown"
            complaint_categories[cat] = complaint_categories.get(cat, 0) + 1
        elif res.sentiment.lower() == 'positive':
            positive_count += 1
        else:
            neutral_count += 1
            
    negative_percentage = (negative_count / total_comments * 100) if total_comments > 0 else 0.0
    
    return InsightsSummary(
        total_comments=total_comments,
        positive_count=positive_count,
        negative_count=negative_count,
        neutral_count=neutral_count,
        negative_percentage=negative_percentage,
        complaint_categories=complaint_categories,
        platform_distribution=platform_distribution,
        brand_distribution=brand_distribution
    )

@app.get("/clusters")
async def get_clusters():
    try:
        labels = complaint_clusterer.get_cluster_labels()
        themes = complaint_clusterer.get_cluster_themes()
        cluster_list = [
            {
                "cluster_id": int(cid),
                "name": themes.get(str(cid), label),
                "keywords": label
            }
            for cid, label in labels.items()
        ]
        return {
            "clusters": cluster_list,
            "labels": labels,
            "themes": themes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch clusters: {str(e)}")

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    model_type = sentiment_analyzer.__class__.__name__
    return ModelInfo(
        sentiment_model_loaded=sentiment_analyzer.is_loaded(),
        clustering_model_loaded=complaint_clusterer.is_loaded(),
        sentiment_model_type=model_type,
        clustering_model_type="ComplaintClusterer"
    )
