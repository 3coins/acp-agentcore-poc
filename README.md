# ACP AgentCore POC

Proof of concept integrating [DeepAgents](https://github.com/langchain-ai/deepagents) with AWS Bedrock AgentCore using the http via websockets transport for [Agent Client Protocol (ACP)](https://github.com/anthropics/agent-client-protocol).

## Overview

This project demonstrates how to deploy a DeepAgents-powered agent as an AWS Bedrock AgentCore service using the Agent Client Protocol (ACP) for client-agent communication.

## Architecture

- **Agent Framework**: [DeepAgents](https://github.com/langchain-ai/deepagents) with LangGraph-based agent orchestration
- **Protocol**: [Agent Client Protocol (ACP)](https://github.com/anthropics/agent-client-protocol) for standardized client-agent communication (using a [fork](https://github.com/3coins/python-sdk/tree/http-ws-transport) with HTTP transport)
- **Transport**: Custom HTTP transport via WebSockets that enables remote ACP agents to be connected with ACP clients.
- **Integration**: [deepagents-acp](https://github.com/langchain-ai/deepagents/tree/main/libs/acp) library bridging DeepAgents with ACP protocol

This project uses a fork of the [agent-client-protocol](https://github.com/3coins/python-sdk/tree/http-ws-transport) that implements an HTTP transport layer via WebSockets. This transport implementation enables remote ACP agents to be connected seamlessly with ACP clients, allowing the agent to run on AWS infrastructure while clients connect over standard WebSocket connections.

The agent uses the `deepagents_acp` project to expose a DeepAgents agent through the ACP protocol, allowing clients to interact with the agent using standardized ACP messages over WebSocket connections.

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- AWS credentials configured (for Bedrock access)
- AWS Bedrock AgentCore CLI (for deployment)

## Installation

Install dependencies using uv:

```bash
uv sync
```

## Running the Agent

### Local Development Server

Start the AgentCore development server:

```bash
agentcore dev
```

This will:
- Start the agent on `http://localhost:8080`
- Expose WebSocket endpoint at `ws://localhost:8080/ws`
- Enable hot-reloading for development

### Testing the Agent

In a separate terminal, run the test client:

```bash
uv run test/test_acp_client.py
```

The test client will:
- Connect to the agent via WebSocket
- Initialize the ACP protocol
- Create a session
- Open an interactive prompt for sending messages to the agent

### Demo Video

https://github.com/user-attachments/assets/ce34b7eb-63ec-4424-9c68-948b525c1ca0


## Configuration

Environment variables can be configured in the agent:

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKSPACE_DIR` | `/tmp/workspace` | Root directory for agent file operations |
| `AGENT_MODE` | `ask_before_edits` | Agent mode: `ask_before_edits` or `auto` |
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock |
| `MODEL_ID` | `global.anthropic.claude-haiku-4-5-20251001-v1:0` | Bedrock model ID |

## Agent Modes

- **`ask_before_edits`**: Agent requests permission before making file changes
- **`auto`**: Agent automatically executes file operations without asking

## Project Structure

```
.
├── src/
│   └── acp_agent_main.py     # Main agent implementation
├── test/
│   └── test_acp_client.py    # Test client for connecting to agent
├── .bedrock_agentcore.yaml   # AgentCore configuration
├── pyproject.toml            # Project dependencies
└── README.md
```

## Key Dependencies

- **[agent-client-protocol](https://github.com/anthropics/agent-client-protocol)**: Protocol and SDK for agent-client communication
- **[deepagents](https://github.com/langchain-ai/deepagents)**: LangGraph-based agent framework
- **[deepagents-acp](https://github.com/langchain-ai/deepagents/tree/main/libs/acp)**: Integration layer between DeepAgents and ACP
- **bedrock-agentcore**: AWS Bedrock AgentCore SDK
- **langchain-aws**: AWS Bedrock integration for LangChain

## How It Works

1. **AgentCore Runtime** provides managed WebSocket infrastructure at `/ws:8080`
2. **ACPDeepAgentBedrock** implements the ACP protocol with a LangGraph-based DeepAgents agent
3. **WebSocketStreamAdapter** bridges Starlette WebSocket with asyncio streams
4. **AgentSideConnection** handles ACP protocol (JSON-RPC routing, method calls, responses)
5. Client connects via WebSocket and communicates using ACP messages

See `src/acp_agent_main.py:3-21` for detailed architecture documentation.

## Deployment

Deploy to AWS using AgentCore CLI:

```bash
agentcore deploy
```

This will:
- Package the agent code
- Deploy to AWS Bedrock AgentCore
- Configure WebSocket endpoints
- Set up IAM roles and permissions

## Related Projects

- [Agent Client Protocol](https://github.com/anthropics/agent-client-protocol) - Protocol specification and Python SDK
- [DeepAgents](https://github.com/langchain-ai/deepagents) - LangGraph-based agent framework
- [DeepAgents ACP](https://github.com/langchain-ai/deepagents/tree/main/libs/acp) - ACP integration for DeepAgents
