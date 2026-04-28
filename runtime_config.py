import os


def get_analytics_product(default: str = "ai_qr_generator") -> str:
    value = os.getenv("ANALYTICS_PRODUCT", default)
    normalized = value.strip()
    return normalized or default
