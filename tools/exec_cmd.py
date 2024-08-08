import subprocess
from llm_chat import ToolParameter, Tool


def exec_cmd_fn(args):
    cmd: str = args['cmd']
    print(f"** (CMD) exec '{cmd}'")
    stdout, stderr = subprocess.Popen(
        ['/bin/bash'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        shell=False
    ).communicate(cmd.encode())

    return stderr.decode('utf-8') if stderr else (stdout.decode('utf-8') if stdout else None)


exec_cmd_tool = Tool(
    name='exec_cmd',
    description="Executes the command in the user's computer shell. stdout or stderr gets returned. The output of the command is not visible to the user.",
    parameters=[
        ToolParameter(
            name='cmd',
            type='string',
            description='The command to execute',
        ),
    ],
    required=['cmd'],
    callback_fn=exec_cmd_fn
)
