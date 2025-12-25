"""
JSON schemas for structured LLM outputs.

These schemas are used with LiteLLM's response_format parameter to ensure
LLMs return valid, parseable JSON for specific tasks.
"""

# Schema for structured diff edits
STRUCTURED_DIFF_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "StructuredDiff",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "description": "List of edits to apply to the code, sorted by start_line",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start_line": {
                                "type": "integer",
                                "description": "First line to replace (1-indexed, inclusive)"
                            },
                            "end_line": {
                                "type": "integer",
                                "description": "Last line to replace (1-indexed, inclusive)"
                            },
                            "new_content": {
                                "type": "string",
                                "description": "New code to replace lines [start_line, end_line]. Empty string deletes lines."
                            },
                            "explanation": {
                                "type": "string",
                                "description": "Brief explanation of this edit"
                            }
                        },
                        "required": ["start_line", "end_line", "new_content", "explanation"],
                        "additionalProperties": False
                    }
                },
                "summary": {
                    "type": "string",
                    "description": "Overall summary of changes"
                }
            },
            "required": ["edits", "summary"],
            "additionalProperties": False
        }
    }
}
