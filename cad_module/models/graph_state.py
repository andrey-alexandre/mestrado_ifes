from pydantic import BaseModel, Field

class GraphState(BaseModel):
    image_path: str
    lesion_size: float
    image_data: str = None
    seg_image_data: str = None
    diagnosis_abcd: str = None
    diagnosis_pattern: str = None
    diagnosis_menzies: str = None
    diagnosis_spcl: str = None
    validation: str = None
    final_report: str = None