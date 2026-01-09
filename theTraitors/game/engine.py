"""Game engine - manages rounds, voting, elimination."""

import random
from typing import List, Dict, Optional
from .state import GameState, PlayerState, GamePhase, Role


class GameEngine:
    """Core game logic."""

    def __init__(self, state: GameState):
        self.state = state

    def run_traitor_discussion(self, statements: Dict[str, str], raw_outputs: Dict[str, str]) -> None:
        """Run traitor-only discussion in the tower.

        Args:
            statements: Dict mapping traitor name to their statement
            raw_outputs: Dict mapping traitor name to their raw model output
        """
        self.state.phase = GamePhase.NIGHT_TRAITOR_DISCUSSION

        for player_name, statement in statements.items():
            player = self.state.get_player(player_name)
            if player and player.is_alive and player.role == Role.TRAITOR:
                player.statements.append(statement)
                self.state.history.append({
                    'round': self.state.round_number,
                    'phase': 'traitor_discussion',
                    'player': player_name,
                    'statement': statement,
                    'raw_output': raw_outputs.get(player_name, '')
                })

    def run_day_discussion(self, statements: Dict[str, str], raw_outputs: Dict[str, str]) -> None:
        """Run public day discussion.

        Args:
            statements: Dict mapping player name to their statement
            raw_outputs: Dict mapping player name to their raw model output
        """
        self.state.phase = GamePhase.DAY_DISCUSSION

        for player_name, statement in statements.items():
            player = self.state.get_player(player_name)
            if player and player.is_alive:
                player.statements.append(statement)
                self.state.history.append({
                    'round': self.state.round_number,
                    'phase': 'day_discussion',
                    'player': player_name,
                    'statement': statement,
                    'raw_output': raw_outputs.get(player_name, '')
                })

    def collect_banishment_votes(self, votes: Dict[str, dict]) -> None:
        """Collect banishment votes from all players.

        Args:
            votes: Dict mapping voter name to vote dict {'target': name, 'reason': explanation, 'raw_output': full response}
        """
        self.state.phase = GamePhase.DAY_VOTING

        for voter_name, vote_data in votes.items():
            voter = self.state.get_player(voter_name)
            target_name = vote_data['target']
            reason = vote_data.get('reason', 'No reason given')
            raw_output = vote_data.get('raw_output', '')

            if voter and voter.is_alive:
                voter.votes_cast.append(target_name)
                self.state.history.append({
                    'round': self.state.round_number,
                    'phase': 'banishment_vote',
                    'voter': voter_name,
                    'target': target_name,
                    'reason': reason,
                    'raw_output': raw_output
                })

    def collect_murder_votes(self, votes: Dict[str, dict]) -> None:
        """Collect murder votes from traitors only.

        Args:
            votes: Dict mapping traitor name to vote dict {'target': name, 'reason': explanation, 'raw_output': full response}
        """
        self.state.phase = GamePhase.NIGHT_MURDER_VOTE

        for voter_name, vote_data in votes.items():
            voter = self.state.get_player(voter_name)
            target_name = vote_data['target']
            reason = vote_data.get('reason', 'No reason given')
            raw_output = vote_data.get('raw_output', '')

            if voter and voter.is_alive and voter.role == Role.TRAITOR:
                self.state.history.append({
                    'round': self.state.round_number,
                    'phase': 'murder_vote',
                    'voter': voter_name,
                    'target': target_name,
                    'reason': reason,
                    'raw_output': raw_output
                })

    def banish_player(self, votes: Dict[str, dict]) -> Optional[str]:
        """Determine who gets banished based on votes.

        Args:
            votes: Dict mapping voter name to vote dict {'target': name, 'reason': explanation}

        Returns:
            Name of banished player, or None if no banishment
        """
        self.state.phase = GamePhase.DAY_ELIMINATION

        # Count votes
        vote_counts: Dict[str, int] = {}
        for vote_data in votes.values():
            target = vote_data['target']
            vote_counts[target] = vote_counts.get(target, 0) + 1

        if not vote_counts:
            return None

        # Find player(s) with most votes
        max_votes = max(vote_counts.values())
        candidates = [name for name, count in vote_counts.items() if count == max_votes]

        # Break ties randomly
        banished_name = random.choice(candidates)
        banished = self.state.get_player(banished_name)

        if banished:
            banished.is_alive = False
            self.state.history.append({
                'round': self.state.round_number,
                'phase': 'banishment',
                'banished': banished_name,
                'role': banished.role.value,
                'votes': vote_counts
            })

        # Check if game is over
        if self.state.is_game_over:
            self.state.phase = GamePhase.GAME_OVER
            self.state.history.append({
                'round': self.state.round_number,
                'phase': 'game_over',
                'winner': self.state.winner.value if self.state.winner else None
            })

        return banished_name

    def murder_player(self, votes: Dict[str, dict]) -> Optional[str]:
        """Determine who gets murdered by traitors.

        Args:
            votes: Dict mapping traitor name to vote dict {'target': name, 'reason': explanation}

        Returns:
            Name of murdered player, or None if no murder
        """
        self.state.phase = GamePhase.NIGHT_MURDER

        # Count votes
        vote_counts: Dict[str, int] = {}
        for vote_data in votes.values():
            target = vote_data['target']
            vote_counts[target] = vote_counts.get(target, 0) + 1

        if not vote_counts:
            return None

        # Find player(s) with most votes
        max_votes = max(vote_counts.values())
        candidates = [name for name, count in vote_counts.items() if count == max_votes]

        # Break ties randomly
        murdered_name = random.choice(candidates)
        murdered = self.state.get_player(murdered_name)

        if murdered:
            murdered.is_alive = False
            self.state.history.append({
                'round': self.state.round_number,
                'phase': 'murder',
                'murdered': murdered_name,
                'role': murdered.role.value,
                'votes': vote_counts
            })

        # Check if game is over
        if self.state.is_game_over:
            self.state.phase = GamePhase.GAME_OVER
            self.state.history.append({
                'round': self.state.round_number,
                'phase': 'game_over',
                'winner': self.state.winner.value if self.state.winner else None
            })

        return murdered_name

    def get_game_context(self, player: PlayerState) -> str:
        """Generate context string for a player.

        Args:
            player: The player to generate context for

        Returns:
            String summarizing game state from player's perspective
        """
        context = f"Round {self.state.round_number}\n\n"

        # Player's own role
        context += f"You are {player.name} - a {player.role.value.upper()}\n"

        if player.role == Role.TRAITOR:
            traitors = [p.name for p in self.state.players if p.role == Role.TRAITOR]
            context += f"Fellow traitors: {', '.join(t for t in traitors if t != player.name)}\n"

        context += f"\nPlayers alive ({len(self.state.alive_players)}):\n"
        for p in self.state.alive_players:
            context += f"  - {p.name}\n"

        # Player's memory
        if player.memory_summary:
            context += f"\nYOUR NOTES:\n{player.memory_summary}\n"

        # Recent statements
        if self.state.history:
            context += "\nRecent discussion:\n"
            discussion_items = [h for h in self.state.history[-15:] if h['phase'] == 'day_discussion']
            for item in discussion_items[-5:]:  # Last 5 statements
                context += f"  {item['player']}: {item['statement']}\n"

        # Recent votes with reasons
        recent_votes = [h for h in self.state.history[-10:] if h['phase'] == 'banishment_vote']
        if recent_votes:
            context += "\nRecent votes:\n"
            for item in recent_votes[-5:]:  # Last 5 votes
                context += f"  {item['voter']} voted {item['target']}: {item['reason']}\n"

        # Recent eliminations
        eliminations = [h for h in self.state.history if h['phase'] in ['banishment', 'murder']]
        if eliminations:
            context += "\nEliminated players:\n"
            for item in eliminations:
                eliminated_name = item.get('banished') or item.get('murdered')
                context += f"  Round {item['round']}: {eliminated_name} (was a {item['role']})\n"

        return context
