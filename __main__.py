from llm_chat import ChatBot
from tools.exec_cmd import exec_cmd_tool


def main():
    ai = ChatBot()
    ai.add_tool(exec_cmd_tool)
    while True:
        res = ai.prompt(input(' > '))
        print(res)


if __name__ == "__main__":
    main()
