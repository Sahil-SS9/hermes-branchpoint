# Reports service (Branchpoint demo)

A tiny FastAPI service whose `GET /report/{customer_id}` is slow (~250 ms).
The decision to explore: **add caching to that endpoint**.

    pip install "fastapi>=0.110,<1" "httpx>=0.27,<1" "pytest>=8,<10"
    pytest            # runs in milliseconds
