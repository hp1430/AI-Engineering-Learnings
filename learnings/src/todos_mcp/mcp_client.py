import asyncio

from fastmcp import Client
from mcp_server import mcp

SERVER_TARGET = mcp

async def main():
    async with Client(SERVER_TARGET) as client:
        print("\n--- Available Tools ---")
        tools = await client.list_tools()
        for tool in tools:
            print(f"{tool.name}: {tool.description}")

        print("\n--- Creating a Todo ---")
        created_todo = await client.call_tool(
            "create_todo",
            {
                "title": "Buy groceries",
                "description": "Buy groceries from the store",
                "status": "pending",
            }
        )
        print(f"Created todo result: {created_todo}")

        print("\n--- Listing Todos ---")
        todos = await client.call_tool("list_todos")
        if hasattr(todos, "data") and isinstance(todos.data, list):
            for todo in todos.data:
                print(todo)
        else:
            print(f"Todos response: {todos}")

if __name__ == "__main__":
    asyncio.run(main())