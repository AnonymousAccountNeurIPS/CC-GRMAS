from pydantic import BaseModel
from typing import Optional

class GraphRequest(BaseModel):
    csv_path: str
    clear_existing: Optional[bool] = False

class GraphResponse(BaseModel):
    success: bool
    message: str
    records_processed: int
    records_failed: int
    total_records: int