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
