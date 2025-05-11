
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!! INITIALIZE MCP SERVER BEFORE RUNNING THIS SCRIPT
# !!! Using the command-line entry point:
# mcp-flight-search --connection_type http
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Here 4 key imports together form the backbone of how
# we bridge Gemini’s LLM interaction with real-world
# APIs exposed via MCP tools.
#
# GenerateContentConfig:
#   Allows us to configure how the model responds
#   (e.g., temperature, tool support, etc.).
# StdioServerParameters:
#   stdio allows the server to be language-neutral
#   and easily embedded in different environments.
#
# The stdio_client, an asynchronous context manager
# used to establish a connection with an MCP server
# over standard I/O. It ensures that the server
# is correctly launched, and the client is
# ready to send/receive structured requests.
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client

# Other imports
import os
import asyncio
import json

# genai.Client() is the primary interface used to
# interact with Google’s generative models
# (e.g., Gemini 2.5 Pro, Gemini 2 Flash)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Configure MCP Tool Server

# mcp-flight-search:
#    This is the CLI entry point to run
#    local MCP server,or could be a Python module
#    in our case that implements the MCP protocol.
#
# stdio:
#   This tells the server to use standard
#   input/output (stdio) as its communication channel.
#   Stdio is simple, language-agnostic,
#   and great for running tool servers
#   locally or in subprocesses.
#
# SERP_API_KEY:
#   This passes an environment variable (SERP_API_KEY)
#   to the subprocess running the tool. In our case,
#   the tool needs it to authenticate with SerpAPI,
#   which fetches real-time flight data
server_params = StdioServerParameters(
    command="mcp-flight-search",
    args=["--connection_type", "stdio"],
    env={"SERP_API_KEY": os.getenv("SERP_API_KEY")},
)


async def run():

    # stdio_client is an asynchronous context manager that handles:
    #     - Launching the MCP server as a subprocess
    #     - Managing the input/output streams for message exchange
    #
    # read and write objects are asynchronous streams
    #     - read: reads responses or tool registration from the server
    #     - write: sends requests or tool invocations to the server

    # collega il client MCP via SSE al server HTTP in esecuzione
    async with sse_client(url="http://localhost:3001/sse") as (read, write):
        async with ClientSession(read, write) as session:
            # qui avviene l'handshake iniziale sul canale SSE
            await session.initialize()

            # server registers its available tools
            # (in our case: a flight search tool).
            # Each tool in mcp_tools.tools contains:
            # - name
            # - description
            # - input schema (i.e., what parameters it accepts, in JSON Schema format)
            mcp_tools = await session.list_tools()

            # This step converts the MCP tool definitions
            # into Gemini’s function_declarations format.
            tools = [
                types.Tool(
                    function_declarations=[
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": {
                                k: v
                                for k, v in tool.inputSchema.items()
                                if k not in ["additionalProperties", "$schema"]
                            },
                        }
                    ]
                )
                for tool in mcp_tools.tools
            ]

            # If Gemini recognizes the prompt as matching
            # a function’s schema, it returns a function_call
            # object that includes the tool name and the
            # auto-filled parameters.
            prompt = f"Find Flights from Atlanta to Las Vegas 2025-05-05"
            response = client.models.generate_content(
                model="gemini-2.5-pro-exp-03-25",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    tools=tools,
                ),
            )

            # Remove raw response print
            if response.candidates[0].content.parts[0].function_call:
                function_call = response.candidates[0].content.parts[0].function_call

                result = await session.call_tool(
                    function_call.name, arguments=dict(function_call.args)
                )

                # Parse and print formatted JSON result
                print("--- Formatted Result ---")  # Add header for clarity
                try:
                    flight_data = json.loads(result.content[0].text)
                    print(json.dumps(flight_data, indent=2))
                except json.JSONDecodeError:
                    print("MCP server returned non-JSON response:")
                    print(result.content[0].text)
                except (IndexError, AttributeError):
                    print("Unexpected result structure from MCP server:")
                    print(result)
            else:
                print("No function call was generated by the model.")
                if response.text:
                    print("Model response:")
                    print(response.text)


asyncio.run(run())
