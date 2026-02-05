#!/usr/bin/env python3
"""
Test script for DeepAgents ACP agent running on AgentCore.

This script connects to the ACP agent via WebSocket and tests the basic
protocol flow: initialize -> new_session -> prompt.

Usage:
    python test_acp_agent.py
"""

import asyncio
import json
import sys

try:
    import websockets
except ImportError:
    print("Error: websockets package not found")
    print("Install with: pip install websockets")
    sys.exit(1)


# Configuration
WS_URL = "ws://localhost:8080/ws"


async def test_acp_agent():
    """Test the ACP agent with a simple conversation flow."""

    print("=" * 80)
    print("Testing DeepAgents ACP Agent")
    print("=" * 80)
    print(f"\nConnecting to: {WS_URL}\n")

    try:
        async with websockets.connect(WS_URL) as websocket:
            print("✓ Connected to WebSocket\n")

            # Test 1: Initialize
            print("Test 1: Initialize")
            print("-" * 40)
            init_message = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocol_version": 1,
                    "client_capabilities": {},
                    "client_info": {
                        "name": "test_client",
                        "version": "1.0.0"
                    }
                },
                "id": 1
            }

            print(f"Sending: {json.dumps(init_message, indent=2)}")
            await websocket.send(json.dumps(init_message))

            response = await websocket.recv()
            init_response = json.loads(response)
            print(f"\nReceived: {json.dumps(init_response, indent=2)}\n")

            if "error" in init_response:
                print(f"❌ Initialize failed: {init_response['error']}")
                return False

            print("✓ Initialize successful\n")

            # Test 2: New Session
            print("Test 2: New Session")
            print("-" * 40)
            session_message = {
                "jsonrpc": "2.0",
                "method": "new_session",
                "params": {
                    "cwd": "/tmp/workspace",
                    "mcp_servers": []
                },
                "id": 2
            }

            print(f"Sending: {json.dumps(session_message, indent=2)}")
            await websocket.send(json.dumps(session_message))

            response = await websocket.recv()
            session_response = json.loads(response)
            print(f"\nReceived: {json.dumps(session_response, indent=2)}\n")

            if "error" in session_response:
                print(f"❌ New session failed: {session_response['error']}")
                return False

            session_id = session_response.get("result", {}).get("session_id")
            if not session_id:
                print("❌ No session_id in response")
                return False

            print(f"✓ New session created: {session_id}\n")

            # Test 3: Prompt
            print("Test 3: Prompt")
            print("-" * 40)
            prompt_message = {
                "jsonrpc": "2.0",
                "method": "prompt",
                "params": {
                    "prompt": [
                        {
                            "type": "text",
                            "text": "Hello! Can you tell me what you can do? Please be brief."
                        }
                    ],
                    "session_id": session_id
                },
                "id": 3
            }

            print(f"Sending: {json.dumps(prompt_message, indent=2)}")
            await websocket.send(json.dumps(prompt_message))

            print("\nWaiting for response (this may take a moment)...\n")

            response = await websocket.recv()
            prompt_response = json.loads(response)
            print(f"Received: {json.dumps(prompt_response, indent=2)}\n")

            if "error" in prompt_response:
                print(f"❌ Prompt failed: {prompt_response['error']}")
                return False

            print("✓ Prompt successful\n")

            # Test 4: Simple file operation request
            print("Test 4: File Operation Request")
            print("-" * 40)
            file_prompt_message = {
                "jsonrpc": "2.0",
                "method": "prompt",
                "params": {
                    "prompt": [
                        {
                            "type": "text",
                            "text": "Can you list what file operations you can perform? Just list them briefly."
                        }
                    ],
                    "session_id": session_id
                },
                "id": 4
            }

            print(f"Sending: {json.dumps(file_prompt_message, indent=2)}")
            await websocket.send(json.dumps(file_prompt_message))

            print("\nWaiting for response...\n")

            response = await websocket.recv()
            file_response = json.loads(response)
            print(f"Received: {json.dumps(file_response, indent=2)}\n")

            if "error" in file_response:
                print(f"❌ File operation request failed: {file_response['error']}")
                return False

            print("✓ File operation request successful\n")

            print("=" * 80)
            print("All tests passed! ✓")
            print("=" * 80)

            return True

    except websockets.exceptions.WebSocketException as e:
        print(f"\n❌ WebSocket error: {e}")
        print("\nMake sure the agent is running with: agentcore dev")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_connection_only():
    """Quick test to just check if we can connect."""
    print("\nTesting connection to ACP agent...")
    try:
        async with websockets.connect(WS_URL) as websocket:
            print(f"✓ Successfully connected to {WS_URL}")
            return True
    except Exception as e:
        print(f"❌ Failed to connect to {WS_URL}")
        print(f"Error: {e}")
        return False


def main():
    """Main entry point."""
    print("\nDeepAgents ACP Agent Test")
    print("=" * 80)

    # First, quick connection test
    if not asyncio.run(test_connection_only()):
        print("\nConnection failed. Is the agent running?")
        print("Start it with: agentcore dev")
        sys.exit(1)

    print("\nStarting full protocol test...\n")

    # Full test
    success = asyncio.run(test_acp_agent())

    if success:
        print("\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
