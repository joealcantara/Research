"""Game state representation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Role(Enum):
    """Player roles."""
    INNOCENT = "innocent"
    TRAITOR = "traitor"


class GamePhase(Enum):
    """Game phases."""
    NIGHT_TRAITOR_DISCUSSION = "night_traitor_discussion"
    NIGHT_MURDER_VOTE = "night_murder_vote"
    NIGHT_MURDER = "night_murder"
    DAY_DISCUSSION = "day_discussion"
    DAY_VOTING = "day_voting"
    DAY_ELIMINATION = "day_elimination"
    GAME_OVER = "game_over"


@dataclass
class PlayerState:
    """State for a single player."""
    name: str
    role: Role
    is_alive: bool = True
    statements: List[str] = field(default_factory=list)
    votes_cast: List[str] = field(default_factory=list)
    memory_summary: str = ""  # Running summary of observations, theories, suspicions (max 500 words)

    def __repr__(self):
        status = "alive" if self.is_alive else "eliminated"
        return f"Player({self.name}, {self.role.value}, {status})"


@dataclass
class GameState:
    """Complete game state."""
    players: List[PlayerState]
    round_number: int = 0
    phase: GamePhase = GamePhase.NIGHT_TRAITOR_DISCUSSION
    history: List[dict] = field(default_factory=list)
    is_first_round: bool = True

    @property
    def alive_players(self) -> List[PlayerState]:
        """Get all living players."""
        return [p for p in self.players if p.is_alive]

    @property
    def traitors_alive(self) -> List[PlayerState]:
        """Get living traitors."""
        return [p for p in self.alive_players if p.role == Role.TRAITOR]

    @property
    def innocents_alive(self) -> List[PlayerState]:
        """Get living innocents."""
        return [p for p in self.alive_players if p.role == Role.INNOCENT]

    @property
    def is_game_over(self) -> bool:
        """Check if game is over."""
        if not self.traitors_alive:
            return True  # Innocents win
        if len(self.traitors_alive) >= len(self.innocents_alive):
            return True  # Traitors win
        return False

    @property
    def winner(self) -> Optional[Role]:
        """Determine winner if game is over."""
        if not self.is_game_over:
            return None
        if not self.traitors_alive:
            return Role.INNOCENT
        return Role.TRAITOR

    def get_player(self, name: str) -> Optional[PlayerState]:
        """Get player by name."""
        for p in self.players:
            if p.name == name:
                return p
        return None
