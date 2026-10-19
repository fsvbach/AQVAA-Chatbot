"""Single source of truth for the interview phase progression, keyed on turn count:

    turns 1-2  -> education
    turn  3    -> stance_elicitation, warn buttons are coming
    turn  4    -> stance_elicitation, buttons appear (announce them)
    turns 5+   -> stance_elicitation, buttons already shown
"""
from dataclasses import dataclass
from typing import Sequence

from models.schemas import ChatHistoryMessage, InterviewStage

# The user turn on which the voting buttons first become available.
SHOW_BUTTONS_AFTER_USER_TURNS = 4
NR_OF_EDUCATION_TURNS = 2


def count_meaningful_user_turns(
    chat_history: Sequence[ChatHistoryMessage],
    current_turn_violated: bool = False,
) -> int:
    """
    Count user turns that actually advanced the interview. 
    Messages caught by guardrails are excluded.
    """
    meaningful = sum(
        1 for m in chat_history if m.role == "user" and not m.guardrail_violated
    )
    if current_turn_violated:
        meaningful -= 1
    return max(meaningful, 0)


@dataclass(frozen=True)
class DialogueState:
    """Immutable snapshot of where a conversation sits in the interview flow.

    `ask_follow_up` is suppressed only on the button-announce turn, so the
    announcement is not buried behind another question.
    """

    stage: InterviewStage
    before_button: bool
    show_buttons: bool
    buttons_first_shown: bool
    ask_follow_up: bool

    @classmethod
    def from_user_turns(cls, user_turns: int) -> "DialogueState":
        """Derive state from the user-turn count (incl. the current turn; pass the
        full history, before any context-window truncation)."""
        show_buttons = user_turns >= SHOW_BUTTONS_AFTER_USER_TURNS
        buttons_first_shown = user_turns == SHOW_BUTTONS_AFTER_USER_TURNS
        return cls(
            stage="education" if user_turns <= NR_OF_EDUCATION_TURNS else "stance_elicitation",
            before_button=user_turns == SHOW_BUTTONS_AFTER_USER_TURNS - 1,
            show_buttons=show_buttons,
            buttons_first_shown=buttons_first_shown,
            ask_follow_up=not buttons_first_shown,
        )
