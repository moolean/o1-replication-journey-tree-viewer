import typing
import functools

import typer
from pydantic import BaseModel


class Message(BaseModel):
    """A message in a chat-oriented LLM."""

    role: typing.Literal["user", "system", "assistant"]
    content: str
    name: str | None = None
    weight: int = 1

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Create an assistant message."""
        return cls(role="assistant", content=content)

    def to_dict(self):
        """Convert the message to a Python dictionary."""
        return self.model_dump(exclude_defaults=True)

    def to_conversation(self) -> "Conversation":
        """Convert the message to a Conversation."""
        return Conversation(messages=[self])


class ScoreMessage(Message, BaseModel):
    """A step with an extracted score"""

    @classmethod
    def from_message(cls, message: Message) -> "ScoreMessage":
        """Create a ScoreMessage from a Message."""
        return cls(**message.model_dump())

    @functools.cached_property
    def score(self) -> float:
        """Score of the step."""
        try:
            score = float(
                [line for line in self.content.split("\n") if "Total Score:" in line][
                    -1
                ]
                .split("Total Score:")[1]
                .strip()
            )
        except (ValueError, IndexError) as e:
            typer.echo(f"Error parsing score ({e}) from:\n{self.content}", err=True)
            score = -1.0
        return score

    def __lt__(self, other: "ScoreMessage"):
        """For priority queue ordering (higher scores first!)."""
        return self.score > other.score


class TerminalCheckMessage(Message):
    """A final message check. It ends in TERMINAL: `[YES/NO]`."""

    @classmethod
    def from_message(cls, message: Message) -> "TerminalCheckMessage":
        """Create a TerminalCheckMessage from a Message."""
        return cls(**message.model_dump())

    @functools.cached_property
    def is_final(self) -> bool:
        """Check if the message has `TERMINAL: YES` or `TERMINAL: NO.`"""
        try:
            if "TERMINAL: YES" in self.content.upper():
                return True
            elif "TERMINAL: NO" in self.content.upper():
                return False
        except Exception as e:
            typer.echo(
                f"Error parsing terminal check ({e}) from:\n{self.content}", err=True
            )
            return False


class Conversation(BaseModel):
    """A conversation is a list of messages."""

    messages: list[Message]

    @classmethod
    def from_mmd(cls, message_dicts):
        """Validate a list of message dictionaries into a Conversation."""
        return cls.model_validate(dict(messages=message_dicts))

    def to_mmd(self) -> list[dict]:
        """Convert the conversation to a list of message dictionaries."""
        return [
            Message(
                **{
                    k: v
                    for k, v in message.model_dump(exclude_defaults=True).items()
                    if k in Message.model_fields
                }
            ).to_dict()
            if type(message) is not Message
            else message.to_dict()
            for message in self.messages
        ]

    def to_markdown(self) -> str:
        """Convert the conversation to a Markdown string."""
        return "\n\n".join(
            f"**{message.role.title()}:**\n{message.content}"
            for message in self.messages
        )

    def __add__(self, other: "Conversation | Message") -> "Conversation":
        """Add two conversations together."""
        if isinstance(other, Message):
            return Conversation(messages=self.messages + [other])
        return Conversation(messages=self.messages + other.messages)

    def __radd__(self, other: Message) -> "Conversation":
        """Add a message to the beginning of a conversation."""
        return Conversation(messages=[other] + self.messages)
