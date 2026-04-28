from qr_modal_contract import (
    GenerateRequest,
    build_generation_kwargs,
    consume_final_result,
    resolve_request_seed,
)


def test_build_generation_kwargs_includes_short_link_flag_and_wrapper_fields():
    request = GenerateRequest(
        mode="standard",
        prompt="poster",
        qr_text="https://example.com",
        input_type="URL",
        use_temporary_short_link=True,
    )

    kwargs = build_generation_kwargs(request)

    assert kwargs["text_input"] == "https://example.com"
    assert kwargs["use_temporary_short_link"] is True
    assert kwargs["analytics_opt_in"] is False


def test_build_generation_kwargs_for_standard_omits_artistic_only_fields():
    request = GenerateRequest(
        mode="standard",
        prompt="poster",
        qr_text="https://example.com",
        include_svg=True,
        enable_cascade_filter=True,
        freeu_b1=2.0,
    )

    kwargs = build_generation_kwargs(request)

    assert "include_svg" not in kwargs
    assert "enable_cascade_filter" not in kwargs
    assert "freeu_b1" not in kwargs


def test_build_generation_kwargs_for_artistic_keeps_artistic_fields():
    request = GenerateRequest(
        mode="artistic",
        prompt="poster",
        qr_text="https://example.com",
        enable_freeu=False,
        enable_cascade_filter=True,
        freeu_b1=2.0,
    )

    kwargs = build_generation_kwargs(request)

    assert kwargs["enable_freeu"] is False
    assert kwargs["enable_cascade_filter"] is True
    assert kwargs["freeu_b1"] == 2.0


def test_build_generation_kwargs_for_artistic_omits_standard_only_fields():
    request = GenerateRequest(
        mode="artistic",
        prompt="poster",
        qr_text="https://example.com",
        controlnet_strength_standard_first=0.9,
        controlnet_strength_standard_final=0.8,
    )

    kwargs = build_generation_kwargs(request)

    assert "controlnet_strength_standard_first" not in kwargs
    assert "controlnet_strength_standard_final" not in kwargs


def test_consume_final_result_supports_wrapper_style_tuples():
    results = [
        (None, "warming", None, None, None),
        (
            "final-image",
            "done",
            {
                "value": '{"shortener_applied": true, "short_url": "https://qrcut.co/demo"}'
            },
            None,
            None,
        ),
    ]

    image, status, settings = consume_final_result(results)

    assert image == "final-image"
    assert status == "done"
    assert settings == {
        "shortener_applied": True,
        "short_url": "https://qrcut.co/demo",
    }


def test_resolve_request_seed_preserves_custom_seed():
    request = GenerateRequest(
        prompt="poster",
        qr_text="https://example.com",
        use_custom_seed=True,
        seed=123,
    )

    assert resolve_request_seed(request) == 123
