import sys
from runner import GameRunner


def main():
    """Run a game of The Traitors."""
    # Get game name from command line args if provided
    game_name = sys.argv[1] if len(sys.argv) > 1 else None

    runner = GameRunner(
        num_players=12,
        num_traitors=3,
        model="llama3.2:1b"
    )

    state = runner.run_game(game_name=game_name)

    # Print summary
    print("\n" + "="*60)
    print("GAME SUMMARY")
    print("="*60)
    print(f"Winner: {state.winner.value.upper()}S")
    print(f"Rounds played: {state.round_number}")
    print(f"Survivors: {', '.join(p.name for p in state.alive_players)}")


if __name__ == "__main__":
    main()
