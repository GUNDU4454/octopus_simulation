"""
Run the live octopus simulation.

  Without a trained agent (demo mode — colour wave animation):
    py simulate.py

  With a trained agent:
    py simulate.py --checkpoint models/checkpoints/octopus_agent
"""
import argparse
from src.visualization import run_simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a saved agent checkpoint (omit for demo mode)")
    args = parser.parse_args()

    agent = None
    if args.checkpoint:
        from src.agent import load_agent
        agent = load_agent(args.checkpoint)

    run_simulation(agent=agent)


if __name__ == "__main__":
    main()
