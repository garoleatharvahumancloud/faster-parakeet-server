def format_json(result):
    return {"text": result.text}


def format_verbose_json(result):
    return {
        "language": result.language,
        "duration": result.duration,
        "segments": [
            {
                "id": s.id,
                "start": s.start,
                "end": s.end,
                "text": s.text,
            }
            for s in result.segments
        ],
    }


def format_text(result):
    return result.text


def format_response(result, response_format: str):
    if response_format == "json":
        return format_json(result)
    if response_format == "verbose_json":
        return format_verbose_json(result)
    if response_format == "text":
        return format_text(result)
    raise ValueError("Unsupported response_format")
