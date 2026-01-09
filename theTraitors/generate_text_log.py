"""Generate human-readable text log from game.json"""

import json
import sys
from pathlib import Path


def generate_text_log(game_json_path: Path) -> str:
    """Generate readable text log from game.json"""

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
                lines.append(f"  [Full output: {entry['raw_output']}]")
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
                lines.append(f"  [Full output: {entry['raw_output']}]")
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
                lines.append(f"  [Full output: {entry['raw_output']}]")
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
                lines.append(f"  [Full output: {entry['raw_output']}]")
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
            lines.append(f"{player} updates their mental notes:")
            lines.append(f"  Trigger: {trigger}")
            lines.append(f"  New info: {new_info}")
            if 'raw_output' in entry:
                lines.append(f"  Updated thoughts:")
                # Indent the raw output
                for line in entry['raw_output'].split('\n'):
                    lines.append(f"    {line}")
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

    return '\n'.join(lines)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python generate_text_log.py <path_to_game.json>")
        sys.exit(1)

    game_json = Path(sys.argv[1])
    if not game_json.exists():
        print(f"Error: {game_json} not found")
        sys.exit(1)

    text_log = generate_text_log(game_json)

    # Save to same directory as game.json
    output_path = game_json.parent / "game_log.txt"
    with open(output_path, 'w') as f:
        f.write(text_log)

    print(f"Text log saved to: {output_path}")
