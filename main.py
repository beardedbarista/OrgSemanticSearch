from fastapi import FastAPI,HTTPException, Depends, Query, Request, Response
import pandas as pd
import logging
import time
from typing import List, Optional
from functools import lru_cache
from sentence_transformers import SentenceTransformer
import numpy as np


path = "/home/codebrewx/myenv/FastAPI_TEST/organizations-100.csv"
model = SentenceTransformer("all-MiniLM-L6-v2")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# cache function to load the CSV file once and resuse the data
@lru_cache(maxsize=1)
def load_file():
    try:
        logger.info("Loading CSV file...")
        df = pd.read_csv(path)
        logger.info("CSV file loaded successfully")

        # create searchable description field and pre-compute embedding

        df["search_text"] = (
    "name: " + df["Name"].fillna("") + " " +
    "website: " + df["Website"].fillna("") + " " +
    "country: " + df["Country"].fillna("") + " " +
    "industry: " + df["Industry"].fillna("") + " " +
    "description: " + df["Description"].fillna("")
).str.lower()


        logger.info("Creating embeddings for semantic search...")
        df["embedding"] = list(model.encode(df["search_text"].tolist(), show_progress_bar=False, normalize_embeddings=True, batch_size=32))
        logger.info("Embedding ready for semantic search.")
        return df
    except FileNotFoundError:
        logger.error("File not found")
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        load_file.cache_clear()
        logger.error(f"An error occurred while loading the CSV file: {str(e)}") 
        raise HTTPException(status_code=500, detail="An error occurred while loading the CSV file")
async def get_dataframe():
    return load_file()
 
app = FastAPI()
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(f"Unhandled exception during {request.method} {request.url.path}")
        raise

    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.5f}"
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({process_time:.3f}s)")

    return response

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/read-data")
async def read_data(df: pd.DataFrame = Depends(get_dataframe)):
    try:
        logger.info("Received a request to read data from CSV")
        
        data = df.to_dict(orient='records')
        return {"data": data}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(f"An error occurred while processing the data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}") 

@app.get("/column-names")
async def get_column_names(df: pd.DataFrame = Depends(get_dataframe)):
    try:
        logger.info("Received a request to get column names")
        column_names = df.columns.tolist()
        return {"columns": column_names}  
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/unique-countries")
async def get_unique_countries(df: pd.DataFrame = Depends(get_dataframe)):
    try:
        logger.info("Received a request to get unique countries")
        unique_countries = df['Country'].unique().tolist()
        return {"countries": unique_countries}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/number-of-employees")
async def get_number_of_employees(df: pd.DataFrame = Depends(get_dataframe)):
    try:
        logger.info("Received a request for employee stats")
        if 'Number of employees' not in df.columns:
            raise HTTPException(status_code=400, detail="Column 'Number of employees' not found")
        stats = {
            "min": int(df['Number of employees'].min()),
            "max": int(df['Number of employees'].max()),
            "average": float(df['Number of employees'].mean())
        }
        return {"employee_stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    

@app.get("/organizations-filtered")
def get_organizations_filtered(
    request: Request,
    name: Optional[str] = Query(None, description="Partial, case-insensitive name match"),
    country: Optional[str] = Query(None, description="Exact country match"),
    industry: Optional[str] = Query(None, description="Exact or partial industry match"),
    founded_after: Optional[int] = Query(None, ge=1900, le=2025, description="Founded year ≥"),
    founded_before: Optional[int] = Query(None, ge=1900, le=2025, description="Founded year ≤"),
    min_employees: Optional[int] = Query(None, ge=0, description="Minimum number of employees"),
    max_employees: Optional[int] = Query(None, ge=0, description="Maximum number of employees"),
    limit: Optional[int] = Query(100, ge=1, le=1000, description="Max records to return"),
    offset: Optional[int] = Query(0, ge=0, description="Pagination offset"),
    sort_by: Optional[str] = Query("Index", description="Column to sort by"),
    sort_order: Optional[str] = Query("asc", regex="^(asc|desc)$"),
    df: pd.DataFrame = Depends(get_dataframe),
):
  
    start_time = time.time()
    query_params = dict(request.query_params)
    logger.info(f"Filtered query:{query_params}")

    try:
        result_df = df.copy()

        # Vectorised filtering
        if name:
            mask = result_df['Name'].str.contains(name, case=False, na=False)
            result_df = result_df[mask]

        if country:
            result_df = result_df[result_df['Country'].str.lower() == country.lower()]

        if industry:
            exact = result_df['Industry'].str.lower() == industry.lower()
            if exact.any():
                result_df = result_df[exact]
            else:
                result_df = result_df[result_df['Industry'].str.contains(industry, case=False, na=False)]

        if founded_after:
            result_df = result_df[result_df['Founded'] >= founded_after]
        if founded_before:
            result_df = result_df[result_df['Founded'] <= founded_before]

        if min_employees:
            result_df = result_df[result_df['Number of employees'] >= min_employees]
        if max_employees:
            result_df = result_df[result_df['Number of employees'] <= max_employees]

        # Sorting
        ascending = sort_order.lower() == "asc"
        if sort_by not in result_df.columns:
            raise HTTPException(400, f"Invalid sort_by column: {sort_by}")
        result_df = result_df.sort_values(by=sort_by, ascending=ascending)

        # Pagination
        total_matches = len(result_df)
        result_df = result_df.iloc[offset : offset + limit]

        # Response with metadata
        response = {
            "data": result_df.to_dict(orient="records"),
            "pagination": {
                "total": total_matches,
                "returned": len(result_df),
                "offset": offset,
                "limit": limit,
            },
            "query_ms": round((time.time() - start_time) * 1000, 2),
        }

        logger.info(f"Query completed in {response['query_ms']}ms | Returned {response['pagination']['returned']} of {total_matches}")
        return response

    except Exception as e:
        logger.error(f"Error in filtered endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during filtering")
    

@app.get("/org-semantic-search")
async def semantic_search_organizations(
    query: str = Query(..., description="Natural language search query"),
    top_k: Optional[int] = Query(10, ge=1, le=100, description="Number of top results to return"),
    df: pd.DataFrame = Depends(get_dataframe)
):
    
    try:
        start_time = time.time()
        query_vec = model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]
        # Validate query
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        embeddings = np.stack(df["embedding"].values)
        similarities = np.dot(embeddings, query_vec)

        # Handle case with no embeddings
        if embeddings.shape[0] == 0:
            logger.warning("No embeddings in DataFrame")
            return{
                "query": query,
                "results": [],
                "query_ms": round((time.time() - start_time) * 1000, 2),
                "message": "No Data Available"
            }
    
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = df.iloc[top_indices].copy()
        results["score"] = similarities[top_indices]

        logger.info(f"Semantic search for query '{query}' returned {len(results)} results in {round((time.time() - start_time) * 1000, 2)}ms")

        return {
            "query": query,
            "results": results[["Name", "Website", "Industry", "Country", "Description", "score"]]
                    .round({"score": 4})
                    .to_dict(orient="records"),
            "query_ms": round((time.time() - start_time) * 1000, 2)
        }
    except Exception as e:
        logger.error(f"Semantic Search Error for {query}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during semantic search")
    
