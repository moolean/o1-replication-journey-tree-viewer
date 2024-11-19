from pydantic import BaseModel, ConfigDict, Field

from o1_replication_journey.conversation import Conversation, Message, ScoreMessage, TerminalCheckMessage

TERMINATION_MARKER = "Done."
STEPPER_MESSAGE = Message.user("next")


class StepTrajectory(BaseModel):
    """A trajectory of steps."""

    base_conversation: Conversation
    step_messages: list[Message]


class Step(BaseModel):
    """A base step."""

    model_config = ConfigDict(frozen=False)

    child_steps: list["ReasoningStep"] = Field(default_factory=list)

    @property
    def score(self) -> float:
        """Score of the step."""
        raise NotImplementedError

    def __lt__(self, other):
        """For priority queue ordering (higher scores first!)."""
        return self.score > other.score

    def to_stepped_conversation(self, with_next_stepper: bool) -> Conversation:
        """Convert the step messages to a Conversation object (without system prompt)
        with STEPPER_MESSAGE between each step.

        Returns:
            A Conversation object.
        """
        raise NotImplementedError

    def to_step_trajectory(self) -> StepTrajectory:
        """Convert the step messages to a list of messages.

        Returns:
            A StepTrajectory object.
        """
        raise NotImplementedError


class Root(Step, BaseModel):
    """The root step of the tree."""

    base_conversation: Conversation

    @property
    def score(self) -> float:
        return 0.0

    def to_stepped_conversation(self, with_next_stepper: bool) -> Conversation:
        return self.base_conversation

    def to_step_trajectory(self) -> StepTrajectory:
        return StepTrajectory(
            base_conversation=self.base_conversation, step_messages=[]
        )


class ReasoningStep(Step, BaseModel):
    """A step with its score and verification."""

    parent_step: Step
    step_message: Message
    score_message: ScoreMessage
    verification_message: Message
    terminal_message_check: TerminalCheckMessage | None = None
    improved_step: "ReasoningStep | None" = None
    aborted: bool = True
    
    # TODO(blackhc): Log conversations for debugging somewhere so we can see
    # how a message was generated.

    @property
    def score(self) -> float:
        """Score of the step."""
        return self.score_message.score
    
    @property
    def is_terminal_step(self) -> bool | None:
        """Check if the step is terminal."""
        if self.terminal_message_check is None:
            return None
        return self.terminal_message_check.is_final

    @property
    def is_terminal_marker(self) -> bool:
        """Check if the step is the terminal "Done." message."""
        return TERMINATION_MARKER == self.step_message.content

    def to_stepped_conversation(self, with_next_stepper: bool) -> Conversation:
        reversed_messages = []

        node = self
        while not isinstance(node, Root):
            reversed_messages.append(STEPPER_MESSAGE)
            reversed_messages.append(node.step_message)
            node = node.parent_step

        if not with_next_stepper:
            reversed_messages = reversed_messages[1:]

        # Reverse the reversed_messages
        conversation = node.base_conversation + Conversation(
            messages=reversed_messages[::-1]
        )
        return conversation

    def to_step_trajectory(self) -> StepTrajectory:
        reversed_messages = []
        node = self
        while not isinstance(node, Root):
            reversed_messages.append(node.step_message)
            node = node.parent_step
        return StepTrajectory(
            base_conversation=node.base_conversation,
            step_messages=reversed_messages[::-1],
        )
