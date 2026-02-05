"""
ACPDeepAgent extension with Bedrock support.

Simple subclass that adds model parameter support to ACPDeepAgent,
enabling use of ChatBedrockConverse instead of the default ChatAnthropic.
"""

import os
from typing import Optional

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.graph import Checkpointer
from deepagents_acp.agent import ACPDeepAgent
from langchain_aws import ChatBedrockConverse
from langchain_core.language_models import BaseChatModel


def create_bedrock_model(
    model_id: Optional[str] = None,
    region: Optional[str] = None,
) -> ChatBedrockConverse:
    """
    Create a ChatBedrockConverse model.

    Args:
        model_id: Bedrock model ID (default: Claude Sonnet 4.5)
        region: AWS region (default: from AWS_REGION env var or us-east-1)

    Returns:
        Configured ChatBedrockConverse instance
    """
    model_id = model_id or os.getenv(
        "MODEL_ID",
        "global.anthropic.claude-haiku-4-5-20251001-v1:0"
    )
    region = region or os.getenv("AWS_REGION", "us-east-1")

    return ChatBedrockConverse(
        model=model_id,
        region_name=region,
    )


class ACPDeepAgentBedrock(ACPDeepAgent):
    """
    ACPDeepAgent with Bedrock model support.

    This subclass adds a model parameter to ACPDeepAgent, allowing you
    to pass in a ChatBedrockConverse (or any LangChain model) instead
    of using the default ChatAnthropic.

    Example:
        from langchain_aws import ChatBedrockConverse

        model = ChatBedrockConverse(
            model="us.anthropic.claude-sonnet-4-5-v2:0",
            region_name="us-east-1"
        )

        agent = ACPDeepAgentBedrock(
            root_dir="/workspace",
            mode="ask_before_edits",
            checkpointer=checkpointer,
            model=model
        )
    """

    def __init__(
        self,
        root_dir: str,
        checkpointer: Checkpointer,
        mode: str,
        model: Optional[BaseChatModel] = None,
    ):
        """
        Initialize ACPDeepAgent with Bedrock support.

        Args:
            root_dir: Root directory for file operations
            checkpointer: LangGraph checkpointer for state persistence
            mode: Agent mode ("ask_before_edits" or "auto")
            model: Optional LangChain chat model (defaults to ChatBedrockConverse)
        """
        self._root_dir = root_dir
        self._checkpointer = checkpointer
        self._mode = mode

        # Store model (create default if not provided)
        self._model = model or create_bedrock_model()

        # Create the deep agent (this will use our overridden method)
        self._deepagent = self._create_deepagent(mode)
        self._cancelled = False
        self._session_plans = {}

        # Call Agent base class __init__ (not ACPDeepAgent)
        # to avoid double initialization
        from acp import Agent
        Agent.__init__(self)

    def _create_deepagent(self, mode: str):
        """
        Create a DeepAgent with Bedrock model.

        Overrides the parent method to pass the model parameter.
        """
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

        # The key change: pass model parameter
        return create_deep_agent(
            model=self._model,  # ← Pass Bedrock model
            checkpointer=self._checkpointer,
            backend=create_backend,
            interrupt_on=interrupt_config,
        )
