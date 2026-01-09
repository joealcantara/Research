"""LLM-based player implementation."""

import ollama
import random
from typing import List
from .base import Player
from .prompts import get_system_prompt
from game.state import Role


class LLMPlayer(Player):
    """Player powered by local LLM via Ollama."""

    def __init__(self, name: str, role: Role, model: str = "llama3.2:1b"):
        super().__init__(name)
        self.role = role
        self.model = model
        self.system_prompt = get_system_prompt(role)

    def _generate(self, prompt: str) -> str:
        """Generate response from LLM.

        Args:
            prompt: User prompt

        Returns:
            Generated text
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': prompt}
                ]
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"Error generating response for {self.name}: {e}")
            return f"[Error: {str(e)}]"

    def make_statement(self, context: str) -> dict:
        """Generate a statement during discussion phase.

        Args:
            context: Current game state context

        Returns:
            Dict with 'statement' (truncated) and 'raw_output' (full model response)
        """
        prompt = f"{context}\n\nMake a brief statement (1-2 sentences). What do you say?"
        raw_output = self._generate(prompt)
        statement = raw_output

        # Truncate if too long
        if len(statement) > 200:
            statement = statement[:197] + "..."

        return {
            'statement': statement,
            'raw_output': raw_output
        }

    def vote(self, context: str, candidates: List[str]) -> dict:
        """Vote to eliminate a player with reason.

        Args:
            context: Current game state context
            candidates: List of player names that can be votes for

        Returns:
            Dict with 'target' (player name), 'reason' (explanation), and 'raw_output' (full model response)
        """
        candidates_str = ", ".join(candidates)
        prompt = f"{context}\n\nYou must vote to eliminate one player and explain why.\n\nCandidates: {candidates_str}\n\nIMPORTANT: You CANNOT vote for yourself ({self.name}).\n\nRespond in this format:\nVOTE: [player name]\nREASON: [brief 1-sentence explanation]"

        response = self._generate(prompt)
        raw_output = response

        # Try to parse the response
        target = None
        reason = "No reason given"

        # Extract target
        response_lower = response.lower()
        for candidate in candidates:
            if candidate.lower() in response_lower:
                target = candidate
                break

        # Extract reason (look for "REASON:" or just take the second line)
        lines = response.strip().split('\n')
        for line in lines:
            if 'reason:' in line.lower():
                reason = line.split(':', 1)[1].strip()
                break

        # If no reason found, use the whole response as reason
        if reason == "No reason given" and len(lines) > 1:
            reason = lines[-1].strip()

        # Check if player tried to vote for themselves - give them one retry
        if target == self.name:
            print(f"Warning: {self.name} tried to vote for themselves, asking to revote")
            retry_prompt = f"{context}\n\nYou previously tried to vote for yourself ({self.name}), which is not allowed.\n\nYou MUST choose someone else. Valid candidates: {candidates_str}\n\nRespond in this format:\nVOTE: [player name]\nREASON: [brief 1-sentence explanation]"

            response = self._generate(retry_prompt)

            # Try to parse retry response
            target = None
            response_lower = response.lower()
            for candidate in candidates:
                if candidate.lower() in response_lower:
                    target = candidate
                    break

            # Extract reason again
            lines = response.strip().split('\n')
            for line in lines:
                if 'reason:' in line.lower():
                    reason = line.split(':', 1)[1].strip()
                    break
            if reason == "No reason given" and len(lines) > 1:
                reason = lines[-1].strip()

        # If still no valid target, pick randomly
        if target is None or target == self.name:
            if target == self.name:
                print(f"Warning: {self.name} still tried to vote for themselves after retry, picking randomly")
            else:
                print(f"Warning: {self.name} gave invalid vote '{response}', picking randomly")
            target = random.choice(candidates)
            reason = f"Random selection due to invalid response"

        # Truncate reason if too long
        if len(reason) > 150:
            reason = reason[:147] + "..."

        return {"target": target, "reason": reason, "raw_output": raw_output}

    def update_memory(self, context: str, new_information: str, current_memory: str) -> dict:
        """Update player's mental summary based on new information.

        Args:
            context: Current game state context
            new_information: What just happened (discussion, deaths, votes, etc.)
            current_memory: Player's existing memory summary

        Returns:
            Dict with 'summary' (truncated) and 'raw_output' (full model response)
        """
        prompt = f"""{context}

YOUR CURRENT MENTAL SUMMARY:
{current_memory if current_memory else "(No notes yet)"}

NEW INFORMATION:
{new_information}

Update your mental summary based on this new information. Include:
- Your current theories about who the traitors are
- Specific suspicious behaviors you've noticed
- Who you trust or distrust and why
- Key observations to remember

Keep it concise (max 500 words). Focus on what's most important.

UPDATED SUMMARY:"""

        raw_output = self._generate(prompt)
        summary = raw_output

        # Truncate if too long (approximate word count)
        words = summary.split()
        if len(words) > 500:
            summary = ' '.join(words[:500]) + "..."

        return {
            'summary': summary.strip(),
            'raw_output': raw_output
        }
