class McpToolParser:

    def parse(self, tool):
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": getattr(tool, "description", "") or "",
                "parameters": getattr(tool, "inputSchema", {"type": "object", "properties": {}})
            }
        }
