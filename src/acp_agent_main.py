#!/usr/bin/env python3
"""
DeepAgents ACP Integration with AWS Bedrock AgentCore

This module integrates DeepAgents Agent Client Protocol (ACP) implementation
with AWS AgentCore Runtime's WebSocket support.

Architecture:
- AgentCore provides managed WebSocket infrastructure at /ws:8080
- ACPDeepAgent implements ACP protocol with LangGraph-based agent
- This module bridges the two with Starlette WebSocket adapters

Usage:
    Local: python src/acp_agent_main.py
    AgentCore: agentcore dev

Environment Variables:
    WORKSPACE_DIR: Root directory for agent file operations (default: /tmp/workspace)
    AGENT_MODE: Agent mode - "ask_before_edits" or "auto" (default: ask_before_edits)
    AWS_REGION: AWS region for Bedrock (default: us-east-1)
    MODEL_ID: Bedrock model ID (default: claude-sonnet-4-5)
"""

import os
import sys
from typing import Dict, Optional
from pathlib import Path

from bedrock_agentcore import BedrockAgentCoreApp

# Import DeepAgents and ACP components
try:
    from deepagents_acp.agent import ACPDeepAgent
    from deepagents import create_deep_agent
    from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
    from acp.agent.connection import AgentSideConnection
    from acp.http import StarletteWebSocketWrapper, WebSocketStreamAdapter
    from acp.schema import AgentCapabilities, PromptCapabilities
    DEEPAGENTS_AVAILABLE = True
except ImportError as e:
    DEEPAGENTS_AVAILABLE = False
    print(f"Warning: DeepAgents ACP not available: {e}", file=sys.stderr)
    print("Install with: uv sync", file=sys.stderr)

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langchain_aws import ChatBedrockConverse
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"Warning: LangChain not available: {e}", file=sys.stderr)


# Configuration from environment
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/tmp/workspace")
AGENT_MODE = os.getenv("AGENT_MODE", "ask_before_edits")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = os.getenv("MODEL_ID", "global.anthropic.claude-haiku-4-5-20251001-v1:0")

# Create AgentCore app
app = BedrockAgentCoreApp()

# Custom ACPDeepAgent subclass that accepts a Bedrock model
class ACPDeepAgentBedrock(ACPDeepAgent):
    """ACPDeepAgent with support for custom Bedrock models.

    Extends ACPDeepAgent to allow passing a specific ChatBedrockConverse model
    instance, enabling use of different Bedrock models and configurations.
    """

    def __init__(
        self,
        root_dir: str,
        mode: str,
        checkpointer,
        model=None,
    ):
        """Initialize with Bedrock model.

        Args:
            root_dir: Root directory for file operations
            mode: Agent mode ("ask_before_edits" or "auto")
            checkpointer: LangGraph checkpointer for state persistence
            model: ChatBedrockConverse model instance (optional)
        """
        self._model = model
        # Call parent __init__ which will call _create_deepagent
        super().__init__(root_dir=root_dir, mode=mode, checkpointer=checkpointer)

    def _create_deepagent(self, mode: str):
        """Create a DeepAgent with Bedrock model and mode configuration."""
        interrupt_config = self._get_interrupt_config(mode)

        def create_backend(tr):
            ephemeral_backend = StateBackend(tr)
            return CompositeBackend(
                default=FilesystemBackend(root_dir=self._root_dir, virtual_mode=True),
                routes={
                    "/memories/": ephemeral_backend,
                    "/conversation_history/": ephemeral_backend,
                },
            )

        return create_deep_agent(
            model=self._model,  # Pass the Bedrock model
            checkpointer=self._checkpointer,
            backend=create_backend,
            interrupt_on=interrupt_config,
        )


# Global state
checkpointer = InMemorySaver() if LANGCHAIN_AVAILABLE else None
agents: Dict[str, ACPDeepAgentBedrock] = {}  # Connection ID -> Agent instance


def create_acp_agent(workspace_dir: str = None, mode: str = None) -> Optional[ACPDeepAgentBedrock]:
    """
    Create an ACPDeepAgentBedrock instance with Bedrock model.

    Args:
        workspace_dir: Root directory for file operations
        mode: Agent mode ("ask_before_edits" or "auto")

    Returns:
        ACPDeepAgentBedrock instance or None if dependencies missing
    """
    if not DEEPAGENTS_AVAILABLE or not LANGCHAIN_AVAILABLE:
        return None

    workspace_dir = workspace_dir or WORKSPACE_DIR
    mode = mode or AGENT_MODE

    # Ensure workspace exists
    Path(workspace_dir).mkdir(parents=True, exist_ok=True)

    # Create Bedrock model
    model = ChatBedrockConverse(
        model=MODEL_ID,
        region_name=AWS_REGION,
    )

    # Create agent with Bedrock model
    agent = ACPDeepAgentBedrock(
        root_dir=workspace_dir,
        mode=mode,
        checkpointer=checkpointer,
        model=model
    )

    return agent


@app.websocket
async def websocket_handler(websocket, context):
    """
    AgentCore WebSocket endpoint implementing ACP protocol.

    This endpoint is automatically mapped to /ws:8080 by AgentCore Runtime.
    It uses the acp.http WebSocketStreamAdapter to bridge Starlette WebSocket
    with asyncio streams, then uses AgentSideConnection to handle the full
    ACP protocol (JSON-RPC routing, method calls, response handling).

    The implementation follows the pattern from acp examples/http_echo_agent.py:
    1. Wrap Starlette WebSocket with StarletteWebSocketWrapper
    2. Create WebSocketStreamAdapter to bridge with asyncio streams
    3. Start background tasks to pump messages through queues
    4. Create AgentSideConnection and listen for messages
    5. Clean up when connection closes

    Args:
        websocket: Starlette WebSocket connection
        context: AgentCore context (contains session info, auth, etc.)
    """
    if not DEEPAGENTS_AVAILABLE:
        await websocket.close(code=1011, reason="DeepAgents ACP not installed")
        return

    await websocket.accept()

    # Get connection identifier
    conn_id = id(websocket)
    print(f"🔌 ACP client connected (conn_id={conn_id})", file=sys.stderr)

    # Extract configuration from query params or context
    query_params = dict(websocket.query_params)
    workspace_dir = query_params.get("workspace_dir", WORKSPACE_DIR)
    mode = query_params.get("mode", AGENT_MODE)

    # Create agent for this connection
    agent = create_acp_agent(workspace_dir=workspace_dir, mode=mode)
    if not agent:
        await websocket.close(code=1011, reason="Failed to create agent")
        return

    agents[conn_id] = agent

    # Wrap Starlette WebSocket to match WebSocketLike protocol
    wrapped_ws = StarletteWebSocketWrapper(websocket)

    # Create adapter to bridge WebSocket with asyncio streams
    adapter = WebSocketStreamAdapter(wrapped_ws)

    agent_conn = None
    try:
        print(f"📝 Agent initialized (mode={mode}, workspace={workspace_dir})", file=sys.stderr)

        # Start background tasks for bidirectional message pumping
        await adapter.start()

        # Create AgentSideConnection
        # From agent's perspective:
        # - input_stream (writer) = for sending to client
        # - output_stream (reader) = for receiving from client
        agent_conn = AgentSideConnection(
            to_agent=agent,
            input_stream=adapter.writer,  # Agent sends via this
            output_stream=adapter.reader,  # Agent receives via this
            listening=False,  # Don't auto-start receive loop
        )

        print(f"🔄 Starting ACP message loop (conn_id={conn_id})", file=sys.stderr)

        # Run main message loop (blocks until connection closes)
        await agent_conn.listen()

        print(f"✓ ACP connection closed normally (conn_id={conn_id})", file=sys.stderr)

    except Exception as e:
        print(f"❌ WebSocket error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
    finally:
        # Cleanup in correct order: connection first, then adapter
        if conn_id in agents:
            del agents[conn_id]
        if agent_conn:
            try:
                await agent_conn.close()
            except:
                pass
        try:
            await adapter.close()
        except:
            pass
        print(f"👋 ACP client disconnected (conn_id={conn_id})", file=sys.stderr)

if __name__ == "__main__":
    app.run(log_level="info")
