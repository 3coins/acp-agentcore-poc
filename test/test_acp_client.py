"""Test client for connecting to ACP agent server.

This test client connects to the ACP agent server via WebSocket and allows
interactive testing of the agent. It's based on the http_client.py example
from the acp-python-sdk.

Requirements:
    - Server must be running (python src/acp_agent_main.py or agentcore dev)
    - All dependencies installed (uv sync)

Usage:
    # Start server in one terminal:
    python src/acp_agent_main.py

    # Run test client in another terminal:
    python test/test_acp_client.py

    # Or with custom URL:
    python test/test_acp_client.py ws://localhost:8080/ws
"""

import asyncio
import logging
import sys
import os
from typing import Any

from acp import (
    PROTOCOL_VERSION,
    Client,
    RequestError,
    text_block,
)
from acp.http import connect_http_agent
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AudioContentBlock,
    AvailableCommandsUpdate,
    ClientCapabilities,
    CreateTerminalResponse,
    CurrentModeUpdate,
    EmbeddedResourceContentBlock,
    EnvVariable,
    ImageContentBlock,
    Implementation,
    KillTerminalCommandResponse,
    PermissionOption,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    ResourceContentBlock,
    TerminalOutputResponse,
    TextContentBlock,
    ToolCall,
    ToolCallProgress,
    ToolCallStart,
    UserMessageChunk,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestACPClient(Client):
    """Test client implementation for ACP agent."""

    async def request_permission(
        self, options: list[PermissionOption], session_id: str, tool_call: ToolCall, **kwargs: Any
    ) -> RequestPermissionResponse:
        """Handle permission requests from agent."""
        print(f"\n🔐 Permission Request: {tool_call.get('title', 'Unknown')}")
        print("Options:")
        for i, option in enumerate(options):
            print(f"  {i+1}. {option.name} ({option.kind})")

        # For testing, auto-approve by default
        # In production, you'd want to prompt the user
        selected_option = options[0]  # Always select first option (usually "approve")
        print(f"✓ Auto-selected: {selected_option.name}\n")

        from acp.schema import PermissionOutcome
        return RequestPermissionResponse(
            outcome=PermissionOutcome(
                outcome="selected",
                option_id=selected_option.option_id,
            )
        )

    async def write_text_file(
        self, content: str, path: str, session_id: str, **kwargs: Any
    ) -> WriteTextFileResponse | None:
        """Not implemented for test client."""
        raise RequestError.method_not_found("fs/write_text_file")

    async def read_text_file(
        self, path: str, session_id: str, limit: int | None = None, line: int | None = None, **kwargs: Any
    ) -> ReadTextFileResponse:
        """Not implemented for test client."""
        raise RequestError.method_not_found("fs/read_text_file")

    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[EnvVariable] | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> CreateTerminalResponse:
        """Not implemented for test client."""
        raise RequestError.method_not_found("terminal/create")

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> TerminalOutputResponse:
        """Not implemented for test client."""
        raise RequestError.method_not_found("terminal/output")

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> ReleaseTerminalResponse | None:
        """Not implemented for test client."""
        raise RequestError.method_not_found("terminal/release")

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> WaitForTerminalExitResponse:
        """Not implemented for test client."""
        raise RequestError.method_not_found("terminal/wait_for_exit")

    async def kill_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> KillTerminalCommandResponse | None:
        """Not implemented for test client."""
        raise RequestError.method_not_found("terminal/kill")

    async def session_update(
        self,
        session_id: str,
        update: UserMessageChunk
        | AgentMessageChunk
        | AgentThoughtChunk
        | ToolCallStart
        | ToolCallProgress
        | AgentPlanUpdate
        | AvailableCommandsUpdate
        | CurrentModeUpdate,
        **kwargs: Any,
    ) -> None:
        """Handle session updates from the agent."""
        # Handle agent messages
        if isinstance(update, AgentMessageChunk):
            content = update.content
            text: str
            if isinstance(content, TextContentBlock):
                text = content.text
            elif isinstance(content, ImageContentBlock):
                text = "<image>"
            elif isinstance(content, AudioContentBlock):
                text = "<audio>"
            elif isinstance(content, ResourceContentBlock):
                text = content.uri or "<resource>"
            elif isinstance(content, EmbeddedResourceContentBlock):
                text = "<resource>"
            else:
                text = "<content>"

            print(f"{text}", end="", flush=True)

        # Handle tool calls
        elif isinstance(update, ToolCallStart):
            title = update.get("title", "Unknown tool")
            status = update.get("status", "pending")
            print(f"\n🔧 Tool Call: {title} ({status})")

        elif isinstance(update, ToolCallProgress):
            status = update.get("status", "in_progress")
            print(f"  → Status: {status}")

        # Handle plan updates
        elif isinstance(update, AgentPlanUpdate):
            entries = update.get("entries", [])
            if entries:
                print("\n📋 Plan Update:")
                for i, entry in enumerate(entries, 1):
                    content = entry.get("content", "")
                    status = entry.get("status", "pending")
                    print(f"  {i}. [{status}] {content}")
            else:
                print("\n📋 Plan cleared")

    async def ext_method(self, method: str, params: dict) -> dict:
        """Handle extension methods."""
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict) -> None:
        """Handle extension notifications."""
        raise RequestError.method_not_found(method)


async def read_console(prompt: str) -> str:
    """Read input from console asynchronously."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


async def interactive_loop(conn, session_id: str) -> None:
    """Run interactive prompt loop."""
    print("\n" + "="*80)
    print("Connected to ACP Agent. Type messages (Ctrl+C or Ctrl+D to exit)")
    print("="*80 + "\n")

    while True:
        try:
            line = await read_console("\n> ")
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n", file=sys.stderr)
            break

        if not line:
            continue

        try:
            print()  # Add newline before agent response
            await conn.prompt(
                session_id=session_id,
                prompt=[text_block(line)],
            )
            print()  # Add newline after agent response
        except Exception as exc:
            logger.error("Prompt failed: %s", exc)


async def main(argv: list[str]) -> int:
    """Main entry point."""
    # Default to localhost
    url = argv[1] if len(argv) >= 2 else "ws://localhost:8080/ws"

    logger.info("Connecting to %s", url)

    try:
        # Connect to agent via WebSocket
        async with connect_http_agent(TestACPClient(), url) as conn:
            logger.info("✓ Connected! Initializing protocol...")

            # Initialize protocol
            await conn.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_capabilities=ClientCapabilities(),
                client_info=Implementation(
                    name="test-acp-client",
                    title="Test ACP Client",
                    version="0.1.0"
                ),
            )
            logger.info("✓ Protocol initialized")

            # Create new session
            cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            session = await conn.new_session(mcp_servers=[], cwd=cwd)
            logger.info("✓ Session created: %s", session.session_id)

            # Run interactive loop
            await interactive_loop(conn, session.session_id)

    except ConnectionRefusedError:
        logger.error("❌ Connection refused. Is the server running?")
        logger.error("   Start server with: python src/acp_agent_main.py")
        return 1
    except Exception as e:
        logger.error("❌ Connection failed: %s", e)
        return 1

    logger.info("Connection closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv)))
