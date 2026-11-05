from pydantic import BaseModel, Field


class TagMapIn(BaseModel):
    mapping_csv: str = Field(description="CSV with header site_tag,twin_tag,scale,offset")
