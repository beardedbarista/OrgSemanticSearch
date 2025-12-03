import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import pandas as pd
import numpy as np
import sqlite3
from main import app


client = TestClient(app)

def test_favicon():
    response = client.get("/favicon.ico")
    assert response.status_code in (200, 204, 404)


def test_basic_search_and_pagination():
    response = client.get("/organizations-filtered?name=Good")
    assert response.status_code == 200
    json_data = response.json()

    assert json_data["pagination"]["total"] >= 1
    assert json_data["pagination"]["returned"] <= 2

    first_record = json_data["data"][0]
    assert "Name" in first_record
    assert "Industry" in first_record


def test_country_filter():
    response = client.get("/organizations-filtered?country=USA")
    assert response.status_code == 200
    data = response.json()["data"]
    assert all(org["Country"] == "USA" for org in data)


def test_industry_filter():
    response = client.get("/organizations-filtered?industry=Plastics")
    assert response.status_code == 200
    data = response.json()["data"]
    assert all(org["Industry"] == "Plastics" for org in data)


def test_sorting():
    response = client.get("/organizations-filtered?sort_by=Founded&sort_order=desc")
    assert response.status_code == 200
    data = response.json()["data"]
    founded_years = [org["Founded"] for org in data]
    assert founded_years == sorted(founded_years, reverse=True)


def test_column_names():
    response = client.get("/column-names")
    assert response.status_code == 200
    json_data = response.json()
    returned_columns = json_data["columns"]

    expected_columns = [
    "Index","Organization Id","Name","Website","Country","Description",
    "Founded","Industry","Number of employees","search_text","embedding"
    ]
    assert set(returned_columns) == set(expected_columns)

def test_number_of_employees():
    response = client.get("/number-of-employees")
    assert response.status_code == 200
    json_data = response.json()
    assert "employee_stats" in json_data
    stats = json_data["employee_stats"]

    conn = sqlite3.connect("organizations.db", check_same_thread=False)
    df = pd.read_sql_query("SELECT * FROM organizations", conn)
    conn.close()

    assert stats["min"] == int(df["Number of employees"].min())
    assert stats["max"] == int(df["Number of employees"].max())
    assert stats["average"] == float(df["Number of employees"].mean())

def test_semantic_search():
    response = client.get("/org-semantic-search?query=health")
    assert response.status_code == 200
    json_data = response.json()
    assert "query" in json_data
    assert "results" in json_data
    assert "query_ms" in json_data

    assert json_data["query"] == "health"
    
    assert isinstance(json_data["results"], list)
    if json_data["results"]:
        assert isinstance(json_data["results"][0], dict)

    query_ms = json_data["query_ms"]
    assert isinstance(query_ms, (int, float))


def test_semantic_search_query_reject():
    response = client.get("/org-semantic-search?query=%20")
    assert response.status_code == 400

    json_data = response.json()
    assert json_data["detail"] == "Query cannot be empty"






