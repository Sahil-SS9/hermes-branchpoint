# Reports service (Branchpoint demo)

A tiny FastAPI service whose `GET /report/{customer_id}` is slow (~250 ms).
The decision to explore: **add caching to that endpoint**.

    pip install fastapi httpx pytest
    pytest            # runs in milliseconds
