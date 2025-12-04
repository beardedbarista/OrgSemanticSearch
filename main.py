from fastapi import FastAPI,HTTPException, Depends, Query, Request, Response
from contextlib import asynccontextmanager
import pandas as pd
import sqlite3
import logging
import time
import json
from typing import Optional
from functools import lru_cache
from sentence_transformers import SentenceTransformer
import numpy as np
import datetime

# Load embedding model and DB once at startup
model = SentenceTransformer("all-MiniLM-L6-v2")
db_connection = None

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_connection
    db_connection = sqlite3.connect("organizations.db", check_same_thread=False)
    db_connection.row_factory = sqlite3.Row
    logger.info("Database connection established at startup")

    yield  

    db_connection.close()
    logger.info("Database connection closed")


app = FastAPI(lifespan=lifespan)



# cache function to load the Database once and resuse
@lru_cache(maxsize=1)
def get_dataframe() -> pd.DataFrame:
    global db_connection

    df = pd.read_sql_query("SELECT * FROM organizations", db_connection)

    if df.empty:
        raise RuntimeError("No data found in database")

    df["search_text"] = (
        "name: " + df["Name"].fillna("") + " " +
        "website: " + df["Website"].fillna("") + " " +
        "country: " + df["Country"].fillna("") + " " +
        "industry: " + df["Industry"].fillna("") + " " +
        "description: " + df["Description"].fillna("")
    ).str.lower()

    logger.info("Computing embeddings from SQLite data...")
    embeddings = model.encode(
        df["search_text"].tolist(),
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=64
    )
    df["embedding"] = list(embeddings)

    logger.info(f"SQLite + embeddings ready: {len(df)} organizations")
    return df
 
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
        
        data = df.to_json(orient='records', date_format='iso')
        return {data: json.loads(df.head(100).to_json(orient="records", date_format="iso"))}
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
async def get_organizations_filtered(
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
    sort_order: Optional[str] = Query("asc", pattern="^(asc|desc)$"),
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

        allowed_columns = {"Name", "Website", "Industry", "Country", "Description", "Founded", "Number of employees"}
        total_matches = len(result_df)
        pagination_df = result_df.iloc[offset : offset + limit]
        display_df = pagination_df[list(allowed_columns)]

        data_records = json.loads(display_df.to_json(orient="records", date_format="iso"))
        
        # Response with metadata
        response = {
            "data": data_records,
            "pagination": {
                "total": total_matches,
                "returned": len(data_records),
                "offset": offset,
                "limit": limit,
            },
            "query_ms": round((time.time() - start_time) * 1000, 2),
        }
        log_filtered_query(
            query_params=dict(request.query_params),
            total=total_matches,
            returned=len(data_records),
            query_ms=response["query_ms"],
            response_json=response  
        )
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
        query_vec = model.encode(query.lower(),show_progress_bar=False,normalize_embeddings=True)
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

        final_response = {
            "query": query,
            "results": results[["Name", "Website", "Industry", "Country", "Description", "score"]]
                        .round({"score": 4})
                        .to_dict(orient="records"),
            "query_ms": round((time.time() - start_time) * 1000, 2)
        }

        # Log the query to trigger helper function
        log_semantic_query(
            query=query,
            top_k=top_k,
            results_count=len(final_response["results"]),
            query_ms=final_response["query_ms"],
            response_json=final_response
        )

        logger.info(f"Semantic search '{query}' → {len(results)} results in {final_response['query_ms']}ms")

        return final_response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Semantic Search Error for {query}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during semantic search")
    
#Helper functions to log queries into the database

def log_filtered_query(query_params: dict, total: int, returned: int, query_ms: float, response_json: dict):
    try:
        conn = sqlite3.connect("organizations.db", check_same_thread=False)
        c = conn.cursor()
        c.execute('''
            INSERT INTO filtered_queries 
            (timestamp, query_params, total_results, returned_results, query_time_ms, response_json)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.datetime.now(datetime.UTC).isoformat(),
            json.dumps(query_params),
            total,
            returned,
            query_ms,
            json.dumps(response_json)
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log filtered query: {e}")  # Now you'll see the real error


def log_semantic_query(query: str, top_k: int, results_count: int, query_ms: float, response_json: dict):
    try:
        conn = sqlite3.connect("organizations.db", check_same_thread=False)
        c = conn.cursor()
        c.execute('''
            INSERT INTO semantic_search_queries 
            (timestamp, query, top_k, results_count, query_time_ms, response_json)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.datetime.now(datetime.UTC).isoformat(),
            query,
            top_k,
            results_count,
            query_ms,
            json.dumps(response_json)
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log semantic query: {e}")