import os

from runtime_config import get_analytics_product


def test_get_analytics_product_defaults_to_ai_qr_generator(monkeypatch):
    monkeypatch.delenv("ANALYTICS_PRODUCT", raising=False)

    assert get_analytics_product() == "ai_qr_generator"


def test_get_analytics_product_uses_env_override(monkeypatch):
    monkeypatch.setenv("ANALYTICS_PRODUCT", "arti_qrcode_app")

    assert get_analytics_product() == "arti_qrcode_app"


def test_get_analytics_product_strips_whitespace(monkeypatch):
    monkeypatch.setenv("ANALYTICS_PRODUCT", "  arti_qrcode_app  ")

    assert get_analytics_product() == "arti_qrcode_app"
