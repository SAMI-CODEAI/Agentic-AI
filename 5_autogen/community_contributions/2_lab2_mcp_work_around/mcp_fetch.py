import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools

async def main():
    # Connect to already-running mcp-server-fetch via stdio (capture stderr to log)
    with open("mcp_fetch_error.log", "w") as errlog:
        fetch_mcp_server = StdioServerParams(command="uvx", args=["mcp-server-fetch"], errlog=errlog)
        try:
            fetcher = await mcp_server_tools(fetch_mcp_server)
        except FileNotFoundError as e:
            errlog.flush()
            try:
                err_contents = open("mcp_fetch_error.log").read()
            except Exception:
                err_contents = "<no stderr captured>"
            hint = (
                "On Windows 'uvx' is often unavailable. Try running this inside WSL (recommended),\n"
                "or ensure 'uvx' is installed and on your PATH.\n"
            )
            raise RuntimeError(
                f"Failed to start mcp-server-fetch with 'uvx'. {hint}\nCaptured stderr:\n{err_contents}"
            ) from e

    # Create assistant agent
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    agent = AssistantAgent(name="fetcher", model_client=model_client, tools=fetcher, reflect_on_tool_use=True)

    # Use the tool
    result = await agent.run(task="Review edwarddonner.com and summarize what you learn. Reply in Markdown.")
    print(result.messages[-1].content)

if __name__ == "__main__":
    asyncio.run(main())
