"""Base player interface."""

from abc import ABC, abstractmethod
from typing import List


class Player(ABC):
    """Abstract base class for players."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def make_statement(self, context: str) -> str:
        """Generate a statement during discussion phase.

        Args:
            context: Current game state context

        Returns:
            Player's statement
        """
        pass

    @abstractmethod
    def vote(self, context: str, candidates: List[str]) -> dict:
        """Vote to eliminate a player with reason.

        Args:
            context: Current game state context
            candidates: List of player names that can be voted for

        Returns:
            Dict with 'target' (player name) and 'reason' (explanation)
        """
        pass
