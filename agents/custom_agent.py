'''
Experimental agent, currently incomplete
'''

import os
from dotenv import load_dotenv
from typing import Optional, Union, Sequence, List

from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from utils.werewolf_utils import _convert_to_str

import agentscope
from agentscope.agents import AgentBase
from agentscope.agents import UserAgent
from agentscope.parsers import ParserBase
from agentscope.message import Msg


'''def format_prompt(
        *args: Union[Msg, Sequence[Msg]],
    ) -> List[dict]:
        """A common format strategy for chat models, which will format the
        input messages into a system message (if provided) and a user message.

        Note this strategy maybe not suitable for all scenarios,
        and developers are encouraged to implement their own prompt
        engineering strategies.

        The following is an example:

        .. code-block:: python

            prompt1 = model.format(
                Msg("system", "You're a helpful assistant", role="system"),
                Msg("Bob", "Hi, how can I help you?", role="assistant"),
                Msg("user", "What's the date today?", role="user")
            )

            prompt2 = model.format(
                Msg("Bob", "Hi, how can I help you?", role="assistant"),
                Msg("user", "What's the date today?", role="user")
            )

        The prompt will be as follows:

        .. code-block:: python

            # prompt1
            [
                {
                    "role": "system",
                    "content": "You're a helpful assistant"
                },
                {
                    "role": "user",
                    "content": (
                        "## Conversation History\\n"
                        "Bob: Hi, how can I help you?\\n"
                        "user: What's the date today?"
                    )
                }
            ]

            # prompt2
            [
                {
                    "role": "user",
                    "content": (
                        "## Conversation History\\n"
                        "Bob: Hi, how can I help you?\\n"
                        "user: What's the date today?"
                    )
                }
            ]


        Args:
            args (`Union[Msg, Sequence[Msg]]`):
                The input arguments to be formatted, where each argument
                should be a `Msg` object, or a list of `Msg` objects.
                In distribution, placeholder is also allowed.

        Returns:
            `List[dict]`:
                The formatted messages.
        """
        if len(args) == 0:
            raise ValueError(
                "At least one message should be provided. An empty message "
                "list is not allowed.",
            )

        # Parse all information into a list of messages
        input_msgs = []
        for _ in args:
            if _ is None:
                continue
            if isinstance(_, Msg):
                input_msgs.append(_)
            elif isinstance(_, list) and all(isinstance(__, Msg) for __ in _):
                input_msgs.extend(_)
            else:
                raise TypeError(
                    f"The input should be a Msg object or a list "
                    f"of Msg objects, got {type(_)}.",
                )

        # record dialog history as a list of strings
        dialogue = []
        sys_prompt = None
        for i, unit in enumerate(input_msgs):
            if i == 0 and unit.role == "system":
                # if system prompt is available, place it at the beginning
                sys_prompt = _convert_to_str(unit.content)
            else:
                # Merge all messages into a conversation history prompt
                dialogue.append(
                    f"{unit.name}: {_convert_to_str(unit.content)}",
                )

        content_components = []

        # The conversation history is added to the user message if not empty
        if len(dialogue) > 0:
            content_components.extend(["## Conversation History"] + dialogue)

        messages = [
            {
                "role": "user",
                "content": "\n".join(content_components),
            },
        ]

        # Add system prompt at the beginning if provided
        if sys_prompt is not None:
            messages = [{"role": "system", "content": sys_prompt}] + messages

        return messages'''


class CustomAgent(AgentBase):
    """An agent that implemented by langchain."""

    def __init__(self, 
                 name: str, 
                 sys_prompt: str, 
                 #use_memory: bool = True,
                 max_retries: Optional[int] = 3,
        ) -> None:
        """Initialize the agent."""

        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            #use_memory=use_memory,
        )
        self.parser = None
        self.max_retries = max_retries

        # [START] BY LANGCHAIN
        # Create a memory in langchain
        memory=ConversationBufferMemory(
            memory_key="chat_history",
            input_key="human_input"
        )

       # Prepare prompt
        template = """
                {sys_prompt}

                Past conversation:
                {chat_history}

                Current input:
                {human_input}
                {format_instruction}
                """

        prompt = PromptTemplate(
            input_variables=["sys_prompt", "chat_history", "human_input", "format_instruction"],
            template=template,
        )

        load_dotenv()
        llm = OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"])

        # Prepare a chain and manage the memory by LLMChain in langchain
        self.llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=False,
            memory=memory,
        )
        # [END] BY LANGCHAIN

    def set_parser(self, parser: ParserBase) -> None:
        """Set response parser, which will provide 1) format instruction; 2)
        response parsing; 3) filtering fields when returning message, storing
        message in memory. So developers only need to change the
        parser, and the agent will work as expected.
        """
        self.parser = parser

    def reply(self, x: Optional[Union[Msg, Sequence[Msg]]] = None) -> Msg:
        # [START] BY LANGCHAIN

        # Generate response
        if x:
            raw_response = self.llm_chain.predict(sys_prompt=self.sys_prompt,
                                                human_input=x.content,
                                                format_instruction=self.parser.format_instruction,)
        else:
            raw_response = self.llm_chain.predict(sys_prompt=self.sys_prompt,
                                                human_input="",
                                                format_instruction=self.parser.format_instruction,)

        # [END] BY LANGCHAIN

        # Ensure compatibility with MarkdownJsonDictParser
        class Response:
            def __init__(self, content):
                self.text = content

        raw_response = Response(raw_response)

        self.speak(raw_response.text)

        # Parsing the raw response
        res = self.parser.parse(raw_response)

        # Wrap the response in a message object in AgentScope
        msg = Msg(
            self.name,
            content=self.parser.to_content(res.parsed),
            role="assistant",
            metadata=self.parser.to_metadata(res.parsed),
        )

        return msg
