# for data validation
from pydantic import BaseModel


class InputValidator(BaseModel):
    lang_data: str