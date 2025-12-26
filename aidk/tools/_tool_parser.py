import inspect

class ToolParser:

    def __init__(self):
        pass

    def _get_doc(self, func):
        doc = inspect.getdoc(func)
        if "Args:" in doc:
            doc = doc.split("Args:")[0]
        doc = doc.replace("\n","").strip()
        return doc

    def _get_params(self, func):
        
        types_map = {
            "str":"string"
        }

        sig = inspect.signature(func)
        params = []

        for name, param in sig.parameters.items():
            # Gestione sicura del tipo
            param_type = "string"  # default
            if param.annotation != inspect.Parameter.empty:
                type_name = param.annotation.__name__
                param_type = types_map.get(type_name, "string")
            
            p = {
                    "name":name, 
                    "type": param_type, 
                }
            
            default = param.default

            if default != inspect.Parameter.empty:
                p["default"] = default
                p["required"] = False
            else:
                p["required"] = True

            params.append(p)
        
        if len(params)==0:
            return params

        doc = inspect.getdoc(func)
        doc = doc.split("Args:")[1]

        for param in params:
            delimiter = param["name"]+" ("+param["type"]+")"+":"
            desc = doc[doc.find(delimiter)+len(delimiter):]
            desc = desc.split("\n")[0].strip()
            param["description"] = desc

        return params

    def parse(self, func):

        tool = {
                "type": "function",
                "function": {
                    "name": func.__name__,
                    "description": self._get_doc(func),
                }
        }

        params = self._get_params(func)

        if len(params)==0:
            return tool

        tool["function"]["parameters"] = {"type":"object", "properties":{}}
        required = []
        for param in params:
            tool["function"]["parameters"]["properties"][param["name"]] = {
                "type":param["type"],
                "description":param["description"]
            }
            if param["required"]:
                required.append(param["name"])

        if len(required)>0:
            tool["function"]["parameters"]["required"] = required

        return tool


