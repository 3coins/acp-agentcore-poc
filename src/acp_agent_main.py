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

import asyncio
import json
import os
import sys
from typing import Any, Dict, Optional
from pathlib import Path

from bedrock_agentcore import BedrockAgentCoreApp

# Import DeepAgents and ACP components
try:
    from deepagents_acp_bedrock import ACPDeepAgentBedrock, create_bedrock_model
    from acp.schema import (
        InitializeResponse,
        NewSessionResponse,
        PromptResponse,
        SetSessionModeResponse,
        TextContentBlock,
    )
    DEEPAGENTS_AVAILABLE = True
except ImportError as e:
    DEEPAGENTS_AVAILABLE = False
    print(f"Warning: DeepAgents ACP not available: {e}", file=sys.stderr)
    print("Install with: uv add deepagents agent-client-protocol langchain-aws", file=sys.stderr)

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langchain_aws import ChatBedrockConverse
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not available", file=sys.stderr)


# Configuration from environment
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/tmp/workspace")
AGENT_MODE = os.getenv("AGENT_MODE", "ask_before_edits")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = os.getenv("MODEL_ID", "global.anthropic.claude-haiku-4-5-20251001-v1:0")

# Create AgentCore app
app = BedrockAgentCoreApp()

# Global state
checkpointer = InMemorySaver() if LANGCHAIN_AVAILABLE else None
agents: Dict[str, "ACPDeepAgentBedrock"] = {}  # Connection ID -> Agent instance


def create_acp_agent(workspace_dir: str = None, mode: str = None) -> Optional["ACPDeepAgentBedrock"]:
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
    model = create_bedrock_model(
        model_id=MODEL_ID,
        region=AWS_REGION
    )

    # Create agent with Bedrock model
    agent = ACPDeepAgentBedrock(
        root_dir=workspace_dir,
        mode=mode,
        checkpointer=checkpointer,
        model=model
    )

    return agent


class StarletteWebSocketAdapter:
    """
    Adapter to make Starlette WebSocket compatible with ACP protocol.

    ACP protocol expects a stream-based interface for reading/writing
    JSON-RPC messages. This adapter provides that interface for
    Starlette's WebSocket object.
    """

    def __init__(self, websocket):
        self.websocket = websocket
        self._closed = False

    async def read_message(self) -> Optional[Dict]:
        """Read a JSON message from the WebSocket"""
        if self._closed:
            return None
        try:
            text = await self.websocket.receive_text()
            return json.loads(text)
        except Exception as e:
            print(f"Error reading message: {e}", file=sys.stderr)
            self._closed = True
            return None

    async def write_message(self, message: Dict):
        """Write a JSON message to the WebSocket"""
        if self._closed:
            return
        try:
            text = json.dumps(message)
            await self.websocket.send_text(text)
        except Exception as e:
            print(f"Error writing message: {e}", file=sys.stderr)
            self._closed = True

    async def close(self):
        """Close the WebSocket connection"""
        if not self._closed:
            try:
                await self.websocket.close()
            except:
                pass
            self._closed = True


@app.websocket
async def websocket_handler(websocket, context):
    """
    AgentCore WebSocket endpoint implementing ACP protocol.

    This endpoint is automatically mapped to /ws:8080 by AgentCore Runtime.
    It implements the Agent Client Protocol, allowing ACP-compatible clients
    (like Zed, VSCode with ACP extension, etc.) to connect.

    Protocol Flow:
        1. Client connects
        2. Client sends "initialize" message
        3. Client sends "new_session" message
        4. Client sends "prompt" messages (main interaction)
        5. Client can send "set_session_mode" to change mode
        6. Client can send "cancel" to interrupt execution
        7. Connection closes

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

    # Create WebSocket adapter
    ws_adapter = StarletteWebSocketAdapter(websocket)

    # Session state
    session_id = None

    try:
        print(f"📝 Agent initialized (mode={mode}, workspace={workspace_dir})", file=sys.stderr)

        # Message loop
        while True:
            # Read message
            message = await ws_adapter.read_message()
            if message is None:
                break

            method = message.get("method")
            params = message.get("params", {})
            msg_id = message.get("id")

            print(f"📨 Received: {method}", file=sys.stderr)

            try:
                # Route to appropriate ACP method
                if method == "initialize":
                    result = await agent.initialize(
                        protocol_version=params.get("protocol_version", 1),
                        client_capabilities=params.get("client_capabilities"),
                        client_info=params.get("client_info")
                    )

                elif method == "new_session":
                    result = await agent.new_session(
                        cwd=params.get("cwd", workspace_dir),
                        mcp_servers=params.get("mcp_servers", [])
                    )
                    if isinstance(result, NewSessionResponse):
                        session_id = result.session_id

                elif method == "prompt":
                    # Convert params to proper format if needed
                    prompt_messages = params.get("prompt", [])
                    if not isinstance(prompt_messages, list):
                        prompt_messages = [{"type": "text", "text": str(prompt_messages)}]

                    # Convert to TextContentBlock objects
                    content_blocks = []
                    for msg in prompt_messages:
                        if isinstance(msg, dict):
                            if msg.get("type") == "text":
                                content_blocks.append(TextContentBlock(type="text", text=msg.get("text", "")))
                        elif isinstance(msg, str):
                            content_blocks.append(TextContentBlock(type="text", text=msg))

                    result = await agent.prompt(
                        prompt=content_blocks,
                        session_id=params.get("session_id") or session_id
                    )

                elif method == "set_session_mode":
                    result = await agent.set_session_mode(
                        mode_id=params.get("mode_id"),
                        session_id=params.get("session_id") or session_id
                    )

                elif method == "cancel":
                    await agent.cancel(session_id=params.get("session_id") or session_id)
                    result = {}

                else:
                    raise ValueError(f"Unknown method: {method}")

                # Send response
                response = {
                    "jsonrpc": "2.0",
                    "result": result.dict() if hasattr(result, 'dict') else result,
                    "id": msg_id
                }
                await ws_adapter.write_message(response)

                print(f"📤 Sent response for: {method}", file=sys.stderr)

            except Exception as e:
                print(f"❌ Error handling {method}: {e}", file=sys.stderr)
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": str(e),
                        "data": {"method": method}
                    },
                    "id": msg_id
                }
                await ws_adapter.write_message(error_response)

    except Exception as e:
        print(f"❌ WebSocket error: {e}", file=sys.stderr)
    finally:
        # Cleanup
        if conn_id in agents:
            del agents[conn_id]
        await ws_adapter.close()
        print(f"👋 ACP client disconnected (conn_id={conn_id})", file=sys.stderr)


# @app.get("/health")
# async def health_check():
#     """
#     Health check endpoint for monitoring.

#     Returns agent status and configuration information.
#     """
#     return {
#         "status": "healthy",
#         "deepagents_acp_available": DEEPAGENTS_AVAILABLE,
#         "langchain_available": LANGCHAIN_AVAILABLE,
#         "active_connections": len(agents),
#         "configuration": {
#             "workspace_dir": WORKSPACE_DIR,
#             "agent_mode": AGENT_MODE,
#             "model_id": MODEL_ID,
#             "aws_region": AWS_REGION
#         },
#         "endpoints": {
#             "websocket": "/ws (ACP protocol)",
#             "health": "/health"
#         },
#         "protocol": {
#             "name": "Agent Client Protocol (ACP)",
#             "version": "1.0",
#             "methods": ["initialize", "new_session", "prompt", "set_session_mode", "cancel"]
#         }
#     }

# @app.get("/info")
# async def info():
#     """
#     Information endpoint about the ACP agent.

#     Provides details on how to connect and use the agent.
#     """
#     return {
#         "name": "DeepAgents ACP on AgentCore",
#         "version": "1.0.0",
#         "description": "LangGraph-based agent with ACP protocol support, deployed on AWS Bedrock AgentCore",
#         "connection": {
#             "protocol": "WebSocket (wss://)",
#             "path": "/ws",
#             "authentication": "AWS SigV4 or OAuth 2.0",
#             "message_format": "JSON-RPC 2.0"
#         },
#         "capabilities": {
#             "file_operations": ["read", "edit", "write", "search"],
#             "planning": "Interactive todo/plan management",
#             "approval_workflows": "Human-in-the-loop for edits",
#             "modes": [
#                 {"id": "ask_before_edits", "description": "Request approval for file operations"},
#                 {"id": "auto", "description": "Auto-approve file operations"}
#             ],
#             "multimodal": ["text", "images"]
#         },
#         "model": {
#             "provider": "AWS Bedrock",
#             "model_id": MODEL_ID,
#             "region": AWS_REGION
#         },
#         "compatible_clients": [
#             "Zed Editor",
#             "VSCode with ACP extension",
#             "Claude Desktop",
#             "Any ACP-compatible client"
#         ],
#         "example_connection": {
#             "python": """
# from bedrock_agentcore.runtime import AgentCoreRuntimeClient
# import websockets
# import json

# client = AgentCoreRuntimeClient(region='us-west-2')
# ws_url, headers = client.generate_ws_connection(runtime_arn='your-arn')

# async with websockets.connect(ws_url, additional_headers=headers) as ws:
#     await ws.send(json.dumps({
#         "jsonrpc": "2.0",
#         "method": "initialize",
#         "params": {"protocol_version": 1},
#         "id": 1
#     }))
#     response = await ws.recv()
#     print(response)
# """
#         }
#     }
    


def main():
    """
    Main entry point for local development.

    For AgentCore deployment, the app.run() is called automatically.
    """
    print("=" * 80)
    print("DeepAgents ACP on AWS Bedrock AgentCore")
    print("=" * 80)
    print(f"Workspace: {WORKSPACE_DIR}")
    print(f"Mode: {AGENT_MODE}")
    print(f"Model: {MODEL_ID}")
    print(f"Region: {AWS_REGION}")
    print("=" * 80)
    print("\nDependencies:")
    print(f"  DeepAgents ACP: {'✓' if DEEPAGENTS_AVAILABLE else '✗'}")
    print(f"  LangChain: {'✓' if LANGCHAIN_AVAILABLE else '✗'}")
    print("=" * 80)
    print("\nEndpoints:")
    print("  WebSocket: ws://localhost:8080/ws (ACP protocol)")
    #print("  Health: http://localhost:8080/health")
    #print("  Info: http://localhost:8080/info")
    print("\nPress Ctrl+C to stop")
    print("=" * 80)

    if not DEEPAGENTS_AVAILABLE:
        print("\n⚠️  WARNING: DeepAgents ACP not available!")
        print("Install with: uv add deepagents agent-client-protocol")
        print("=" * 80)

    app.run(log_level="info")


if __name__ == "__main__":
    main()
