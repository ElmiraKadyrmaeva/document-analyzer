from pydantic import BaseModel


class BoundingBox(BaseModel):
    """Координаты прямоугольной области слова."""

    left: float
    top: float
    right: float
    bottom: float
