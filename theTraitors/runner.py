"""Game runner - orchestrates full game."""

import random
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from game import GameState, GameEngine, Role, PlayerState
from players import LLMPlayer


class GameRunner:
    """Orchestrates a full game of The Traitors."""

    def __init__(self, num_players: int = 12, num_traitors: int = 3, model: str = "llama3.2:1b"):
        """Initialize game runner.

        Args:
            num_players: Total number of players
            num_traitors: Number of traitors
            model: Ollama model to use
        """
        self.num_players = num_players
        self.num_traitors = num_traitors
        self.model = model
        self.games_dir = Path("games")
        self.games_dir.mkdir(exist_ok=True)
        self.notebook_template = Path("notebook_template.ipynb")

    def create_game(self) -> tuple[GameState, dict[str, LLMPlayer]]:
        """Create a new game with players.

        Returns:
            Tuple of (GameState, dict of player name to LLMPlayer)
        """
        # Generate player names
        player_names = [f"Player{i+1}" for i in range(self.num_players)]

        # Randomly assign roles
        traitor_indices = random.sample(range(self.num_players), self.num_traitors)
        roles = [Role.TRAITOR if i in traitor_indices else Role.INNOCENT
                for i in range(self.num_players)]

        # Create player states
        player_states = [PlayerState(name=name, role=role)
                        for name, role in zip(player_names, roles)]

        # Create LLM players
        llm_players = {
            name: LLMPlayer(name=name, role=role, model=self.model)
            for name, role in zip(player_names, roles)
        }

        # Create game state
        state = GameState(players=player_states)

        return state, llm_players

    def run_night_phase(self, engine: GameEngine, llm_players: dict[str, LLMPlayer]) -> Optional[str]:
        """Run night phase: traitors discuss and murder someone.

        Args:
            engine: Game engine
            llm_players: Dict of player name to LLMPlayer

        Returns:
            Name of murdered player
        """
        state = engine.state

        print(f"\n{'='*60}")
        print(f"🌙 NIGHT - THE TOWER")
        print(f"{'='*60}")

        # Traitor discussion
        print("\n--- TRAITOR DISCUSSION ---")
        traitor_statements = {}
        traitor_raw_outputs = {}
        for traitor in state.traitors_alive:
            context = engine.get_game_context(traitor)
            llm_player = llm_players[traitor.name]
            result = llm_player.make_statement(context)
            traitor_statements[traitor.name] = result['statement']
            traitor_raw_outputs[traitor.name] = result['raw_output']
            print(f"🔴 {traitor.name}: {result['statement']}")

        engine.run_traitor_discussion(traitor_statements, traitor_raw_outputs)

        # Murder vote (skip memory update here - will update after murder reveal)
        print("\n--- MURDER VOTE ---")
        murder_votes = {}
        innocent_candidates = [p.name for p in state.innocents_alive]

        for traitor in state.traitors_alive:
            context = engine.get_game_context(traitor)
            llm_player = llm_players[traitor.name]
            vote_data = llm_player.vote(context, innocent_candidates)
            murder_votes[traitor.name] = vote_data
            print(f"🔴 {traitor.name} votes to murder {vote_data['target']}")
            print(f"     Reason: {vote_data['reason']}")

        engine.collect_murder_votes(murder_votes)

        # Execute murder
        murdered = engine.murder_player(murder_votes)
        if murdered:
            murdered_player = state.get_player(murdered)
            print(f"\n💀 {murdered} was MURDERED in the night!")
            print(f"   (They were a {murdered_player.role.value.upper()})")

            # Check game over
            if state.is_game_over:
                print(f"\n{'='*60}")
                print(f"GAME OVER - {state.winner.value.upper()}S WIN!")
                print(f"{'='*60}")

        return murdered

    def run_day_phase(self, engine: GameEngine, llm_players: dict[str, LLMPlayer], with_banishment: bool = True) -> Optional[str]:
        """Run day phase: public discussion and optional banishment.

        Args:
            engine: Game engine
            llm_players: Dict of player name to LLMPlayer
            with_banishment: Whether to hold a banishment vote

        Returns:
            Name of banished player, or None if no banishment
        """
        state = engine.state

        print(f"\n{'='*60}")
        print(f"☀️  DAY - PUBLIC DISCUSSION")
        print(f"{'='*60}")

        # Public discussion
        print("\n--- DISCUSSION ---")
        statements = {}
        raw_outputs = {}
        for player in state.alive_players:
            context = engine.get_game_context(player)
            llm_player = llm_players[player.name]
            result = llm_player.make_statement(context)
            statements[player.name] = result['statement']
            raw_outputs[player.name] = result['raw_output']
            emoji = '🔴' if player.role == Role.TRAITOR else '🔵'
            print(f"{emoji} {player.name}: {result['statement']}")

        engine.run_day_discussion(statements, raw_outputs)

        # Skip memory update here - will update after deaths are revealed
        if not with_banishment:
            print("\n(No banishment vote this round)")
            return None

        # Banishment vote
        print("\n--- BANISHMENT VOTE ---")
        votes = {}
        candidates = [p.name for p in state.alive_players]

        for player in state.alive_players:
            # Can't vote for yourself
            valid_candidates = [c for c in candidates if c != player.name]
            context = engine.get_game_context(player)
            llm_player = llm_players[player.name]
            vote_data = llm_player.vote(context, valid_candidates)
            votes[player.name] = vote_data
            emoji = '🔴' if player.role == Role.TRAITOR else '🔵'
            print(f"{emoji} {player.name} votes to banish {vote_data['target']}")
            print(f"     Reason: {vote_data['reason']}")

        engine.collect_banishment_votes(votes)

        # Banishment
        banished = engine.banish_player(votes)
        if banished:
            banished_player = state.get_player(banished)
            print(f"\n⚖️  {banished} was BANISHED!")
            print(f"   They were a {banished_player.role.value.upper()}")

            # Update memories after banishment reveal (MAJOR EVENT - keep this one)
            print("\n(Players updating their notes...)")
            for player in state.alive_players:
                llm_player = llm_players[player.name]
                context = engine.get_game_context(player)
                new_info = f"{banished} was banished and revealed to be a {banished_player.role.value.upper()}! This is important information about who the traitors might be."
                memory_result = llm_player.update_memory(context, new_info, player.memory_summary)
                player.memory_summary = memory_result['summary']
                # Store raw output in history
                state.history.append({
                    'round': state.round_number,
                    'phase': 'memory_update',
                    'player': player.name,
                    'trigger': 'banishment_reveal',
                    'new_info': new_info,
                    'raw_output': memory_result['raw_output']
                })

            # Check game over
            if state.is_game_over:
                print(f"\n{'='*60}")
                print(f"GAME OVER - {state.winner.value.upper()}S WIN!")
                print(f"{'='*60}")

        return banished

    def run_game(self, game_name: str = None) -> GameState:
        """Run a complete game.

        Args:
            game_name: Optional custom name for the game (e.g., "0.1").
                      If not provided, uses timestamp.

        Returns:
            Final game state
        """
        self.game_name = game_name  # Store for later use in save_log

        print("="*60)
        print("THE TRAITORS - Game Starting")
        print("="*60)

        state, llm_players = self.create_game()
        engine = GameEngine(state)

        # Print initial setup
        print(f"\nPlayers: {', '.join(p.name for p in state.players)}")
        print(f"Traitors: {', '.join(p.name for p in state.players if p.role == Role.TRAITOR)}")
        print(f"Innocents: {', '.join(p.name for p in state.players if p.role == Role.INNOCENT)}")

        # Round 1: Day (no banishment) → Night (murder)
        state.round_number = 1
        print(f"\n{'='*60}")
        print(f"ROUND 1")
        print(f"{'='*60}")

        self.run_day_phase(engine, llm_players, with_banishment=False)
        if not state.is_game_over:
            self.run_night_phase(engine, llm_players)

            # After murder, all players update their memory
            recent_murder = [h for h in state.history if h.get('phase') == 'murder'][-1]
            if recent_murder:
                print("\n(Players updating their notes after murder...)")
                for player in state.alive_players:
                    llm_player = llm_players[player.name]
                    context = engine.get_game_context(player)
                    new_info = f"{recent_murder['murdered']} was MURDERED during the night! (They were a {recent_murder['role'].upper()})"
                    memory_result = llm_player.update_memory(context, new_info, player.memory_summary)
                    player.memory_summary = memory_result['summary']
                    # Store raw output in history
                    state.history.append({
                        'round': state.round_number,
                        'phase': 'memory_update',
                        'player': player.name,
                        'trigger': 'murder_reveal',
                        'new_info': new_info,
                        'raw_output': memory_result['raw_output']
                    })

        # Round 2+: Day (with banishment) → Night (murder)
        while not state.is_game_over:
            state.round_number += 1
            print(f"\n{'='*60}")
            print(f"ROUND {state.round_number}")
            print(f"{'='*60}")

            self.run_day_phase(engine, llm_players, with_banishment=True)
            if not state.is_game_over:
                self.run_night_phase(engine, llm_players)

                # After murder, all players update their memory
                recent_murder = [h for h in state.history if h.get('phase') == 'murder'][-1]
                if recent_murder:
                    print("\n(Players updating their notes after murder...)")
                    for player in state.alive_players:
                        llm_player = llm_players[player.name]
                        context = engine.get_game_context(player)
                        new_info = f"{recent_murder['murdered']} was MURDERED during the night! (They were a {recent_murder['role'].upper()})"
                        memory_result = llm_player.update_memory(context, new_info, player.memory_summary)
                        player.memory_summary = memory_result['summary']
                        # Store raw output in history
                        state.history.append({
                            'round': state.round_number,
                            'phase': 'memory_update',
                            'player': player.name,
                            'trigger': 'murder_reveal',
                            'new_info': new_info,
                            'raw_output': memory_result['raw_output']
                        })

        # Save game log
        self.save_log(state)

        return state

    def save_log(self, state: GameState) -> None:
        """Save game log and analysis notebook to dedicated folder.

        Args:
            state: Final game state
        """
        if hasattr(self, 'game_name') and self.game_name:
            game_dir_name = f"game_{self.game_name}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            game_dir_name = f"game_{timestamp}"

        game_dir = self.games_dir / game_dir_name
        game_dir.mkdir(exist_ok=True)

        # Save game data
        game_file = game_dir / "game.json"
        log_data = {
            'players': [
                {
                    'name': p.name,
                    'role': p.role.value,
                    'survived': p.is_alive
                }
                for p in state.players
            ],
            'winner': state.winner.value if state.winner else None,
            'rounds': state.round_number,
            'history': state.history
        }

        with open(game_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        # Copy analysis notebook template
        if self.notebook_template.exists():
            notebook_file = game_dir / "analyze.ipynb"
            shutil.copy(self.notebook_template, notebook_file)

        # Generate text log
        self._generate_text_log(game_file, game_dir / "game_log.txt")

        print(f"\nGame saved to: {game_dir}")

    def _generate_text_log(self, game_json_path: Path, output_path: Path) -> None:
        """Generate human-readable text log from game.json

        Args:
            game_json_path: Path to game.json
            output_path: Path for output text file
        """
        with open(game_json_path) as f:
            data = json.load(f)

        lines = []
        lines.append("=" * 80)
        lines.append("THE TRAITORS - GAME LOG")
        lines.append("=" * 80)
        lines.append("")

        # Setup
        lines.append("SETUP")
        lines.append("-" * 80)
        traitors = [p['name'] for p in data['players'] if p['role'] == 'traitor']
        innocents = [p['name'] for p in data['players'] if p['role'] == 'innocent']
        lines.append(f"Traitors: {', '.join(traitors)}")
        lines.append(f"Innocents: {', '.join(innocents)}")
        lines.append("")

        # Organize history by rounds
        current_round = 0

        for entry in data['history']:
            round_num = entry['round']
            phase = entry['phase']

            # New round header
            if round_num != current_round:
                current_round = round_num
                lines.append("")
                lines.append("=" * 80)
                lines.append(f"ROUND {round_num}")
                lines.append("=" * 80)
                lines.append("")

            # Day discussion
            if phase == 'day_discussion':
                if not any('DAY DISCUSSION' in line for line in lines[-5:]):
                    lines.append("DAY DISCUSSION")
                    lines.append("-" * 80)
                player = entry['player']
                statement = entry['statement']
                lines.append(f"{player}: {statement}")
                if 'raw_output' in entry and entry['raw_output'] != statement:
                    lines.append(f"  [Full: {entry['raw_output']}]")
                lines.append("")

            # Traitor discussion
            elif phase == 'traitor_discussion':
                if not any('TRAITOR DISCUSSION' in line for line in lines[-5:]):
                    lines.append("TRAITOR DISCUSSION (Private)")
                    lines.append("-" * 80)
                player = entry['player']
                statement = entry['statement']
                lines.append(f"{player}: {statement}")
                if 'raw_output' in entry and entry['raw_output'] != statement:
                    lines.append(f"  [Full: {entry['raw_output']}]")
                lines.append("")

            # Banishment vote
            elif phase == 'banishment_vote':
                if not any('BANISHMENT VOTE' in line for line in lines[-5:]):
                    lines.append("BANISHMENT VOTE")
                    lines.append("-" * 80)
                voter = entry['voter']
                target = entry['target']
                reason = entry['reason']
                lines.append(f"{voter} votes to BANISH {target}")
                lines.append(f"  Reason: {reason}")
                if 'raw_output' in entry:
                    lines.append(f"  [Full: {entry['raw_output']}]")
                lines.append("")

            # Murder vote
            elif phase == 'murder_vote':
                if not any('MURDER VOTE' in line for line in lines[-5:]):
                    lines.append("MURDER VOTE (Private)")
                    lines.append("-" * 80)
                voter = entry['voter']
                target = entry['target']
                reason = entry['reason']
                lines.append(f"{voter} votes to MURDER {target}")
                lines.append(f"  Reason: {reason}")
                if 'raw_output' in entry:
                    lines.append(f"  [Full: {entry['raw_output']}]")
                lines.append("")

            # Banishment result
            elif phase == 'banishment':
                banished = entry['banished']
                role = entry['role']
                votes = entry.get('votes', {})
                lines.append("BANISHMENT RESULT")
                lines.append("-" * 80)
                lines.append(f">>> {banished} was BANISHED <<<")
                lines.append(f">>> They were a {role.upper()} <<<")
                if votes:
                    lines.append(f"Vote counts: {votes}")
                lines.append("")

            # Murder result
            elif phase == 'murder':
                murdered = entry['murdered']
                role = entry['role']
                votes = entry.get('votes', {})
                lines.append("MURDER RESULT")
                lines.append("-" * 80)
                lines.append(f">>> {murdered} was MURDERED <<<")
                lines.append(f">>> They were a {role.upper()} <<<")
                if votes:
                    lines.append(f"Vote counts: {votes}")
                lines.append("")

            # Memory update
            elif phase == 'memory_update':
                if not any('MEMORY UPDATES' in line for line in lines[-5:]):
                    lines.append("MEMORY UPDATES (Players processing new information)")
                    lines.append("-" * 80)
                player = entry['player']
                trigger = entry.get('trigger', 'unknown')
                new_info = entry.get('new_info', '')
                lines.append(f"{player} updates mental notes:")
                lines.append(f"  Trigger: {trigger}")
                lines.append(f"  New info: {new_info}")
                if 'raw_output' in entry:
                    lines.append(f"  Updated thoughts:")
                    # Indent the raw output
                    for line in entry['raw_output'].split('\n')[:20]:  # Limit to 20 lines
                        lines.append(f"    {line}")
                    if len(entry['raw_output'].split('\n')) > 20:
                        lines.append(f"    ... (truncated)")
                lines.append("")

            # Game over
            elif phase == 'game_over':
                lines.append("")
                lines.append("=" * 80)
                lines.append("GAME OVER")
                lines.append("=" * 80)
                winner = entry.get('winner', 'unknown')
                lines.append(f"Winner: {winner.upper()}S")
                lines.append("")

        # Final summary
        lines.append("")
        lines.append("=" * 80)
        lines.append("FINAL SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Winner: {data['winner'].upper()}S")
        lines.append(f"Rounds: {data['rounds']}")
        survivors = [p['name'] for p in data['players'] if p['survived']]
        lines.append(f"Survivors: {', '.join(survivors)}")
        lines.append("")

        eliminated = [p for p in data['players'] if not p['survived']]
        lines.append("Eliminated players:")
        for p in eliminated:
            lines.append(f"  - {p['name']} ({p['role']})")
        lines.append("")

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
