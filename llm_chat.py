import ollama
from llm_memory import ChatMemory
import json
import threading
from typing import Callable, Dict, Mapping, List, TypedDict
import uuid


class ToolParameter(TypedDict):
    name: str
    type: str
    description: str


class Tool():
    def __init__(self, name: str, description: str, parameters: List[ToolParameter], required: List[str], callback_fn) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.required = required
        self.callback_fn = callback_fn

    def _parameters_to_blueprint(self):
        parameters = {}
        for parameter in self.parameters:
            parameters[parameter['name']] = {
                "type": parameter['type'],
                "description": parameter["description"]
            }
        return {
            "type": "object",
            "properties": parameters
        }

    def to_blueprint(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._parameters_to_blueprint(),
                "required": self.required
            }
        }


class ChatBot():
    MODEL = "llama3.1"

    @staticmethod
    def _make_system_prompt():
        system_instructions = open(
            "prompts/SYSTEM_INSTRUCTION.md",
            "r"
        ).read()

        return system_instructions

    def __init__(self) -> None:
        self.ollama_client = ollama.Client()

        self.chat_id = str(uuid.uuid1())
        self.memory = ChatMemory(chat_id=self.chat_id)

        self.message_history = []

        system_instruction_msg = {
            "role": "system",
            "content": self._make_system_prompt()
        }

        self.message_history.append(system_instruction_msg)

        self.tool_blue_prints = []
        self.tools_callback_map: Dict[str, Callable] = {}

    def add_tool(self, tool: Tool):
        self.tools_callback_map[tool.name] = tool.callback_fn
        self.tool_blue_prints.append(tool.to_blueprint())

    def _gen_response(self):
        return self.ollama_client.chat(
            model=self.MODEL,
            messages=self.message_history,
            options={
                "temperature": 0,
            },
            tools=self.tool_blue_prints if len(
                self.tool_blue_prints) != 0 else None,
        )

    @staticmethod
    def _is_tool_called(res: Mapping[str, Dict]) -> bool:
        return 'tool_calls' in res['message'].keys() and len(res['message']['tool_calls']) != 0

    def _handle_tool_call(self, tool_calls: List[Mapping[str, Dict]]):
        for tool in tool_calls:
            tool = tool['function']
            tool_callback = self.tools_callback_map[tool['name']]
            function_response = tool_callback(tool['arguments'])
            resp_obj = {
                "role": "tool",
                "content": json.dumps(function_response)
            }
            self.message_history.append(resp_obj)
            return self._act()

    def _act(self):
        res = self._gen_response()
        self.message_history.append(res['message'])

        if self._is_tool_called(res):
            return self._handle_tool_call(res['message']['tool_calls'])
        else:
            return res['message']['content']

    def prompt(self, prompt: str) -> str:
        memories_related_to_the_prompt = self.memory.get_relevant_memories(
            query=prompt
        )

        prompt_with_related_memories = json.dumps({
            "prompt": prompt,
            "related_memories": memories_related_to_the_prompt
        })

        user_prompt_obj = {
            "role": "user",
            "content": prompt_with_related_memories,
        }
        self.message_history.append(user_prompt_obj)
        # we don't save the prompt with the related memories
        self._save_chat_obj_parallel({
            "role": "user",
            "content": prompt
        })

        response_content = self._act()

        # saving the response generated from the chat history in
        # response object gets appended to the history so its easier to just get it from the history
        self._save_chat_obj_parallel(self.message_history[-1])
        return response_content  # type: ignore

    def _save_chat_obj_parallel(self, chat_obj):
        save_obj_t = threading.Thread(
            target=self.memory.save_chat_obj,
            args=(chat_obj,)
        )
        save_obj_t.start()
        save_obj_t.join()
