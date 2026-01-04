# Scientist AI: A Research Plan for Safer Advanced AI

**Summary of Sections 3-4 from "Superintelligent Agents Pose Catastrophic Risks: Can Scientist AI Offer a Safer Path?" by Bengio et al. (2025)**

## Overview

The paper proposes **Scientist AI** as a safer alternative to agentic AI systems. Rather than pursuing goals through actions in the world, Scientist AI is designed to *understand* the world and make probabilistic inferences based on that understanding.

---

## Section 3: The Scientist AI Research Plan

### 3.1 Introduction to Scientist AI

**Core Design:**
- **World Model**: Generates causal theories to explain observed data
- **Inference Machine**: Answers questions based on theories from the world model
- Both components use Bayesian methods to handle uncertainty properly

**Key Properties:**
1. **Non-agentic by design**: Lacks goal-directedness and affordances (ability to act)
2. **Bayesian**: Maintains uncertainty over multiple theories rather than committing to one
3. **Unique solution**: At the global optimum, there's only one correct probability for each query
4. **Interpretable**: Theories are expressed as logical statements and causal models

**Three Primary Use Cases:**
1. Accelerating scientific research
2. Serving as a guardrail for other (potentially unsafe) AI systems
3. Helping build safer superintelligent AI systems

**Time Horizons:**
- **Short-term**: Fine-tune existing LLMs as guardrails
- **Long-term**: Build from scratch using full Bayesian posterior objective

---

### 3.2 Restricting Agency

**Three Pillars of Agency:**
1. **Intelligence**: Knowledge and reasoning ability
2. **Affordances**: Scope of possible actions
3. **Goal-directedness**: Motivated behavior toward preferred states

**How Scientist AI Restricts Agency:**
- Eliminates goal-directedness (no preferences over world states)
- Minimizes affordances (output limited to probability estimates)
- Retains intelligence (needed for understanding and inference)

**Key Insight**: Removing any single pillar eliminates dangerous agency. Scientist AI removes two for redundancy.

**Narrow AI Safety**: Even narrow agentic AIs can be made safer by:
- Limiting affordances to specialized domains
- Using Scientist AI as a guardrail

---

### 3.3 The Bayesian Approach

**Why Bayesian?**
- Avoids overconfident predictions that could lead to catastrophic outcomes
- Maintains distribution over multiple plausible theories
- Averages predictions across theories weighted by evidence

**Bayesian Posterior Over Theories:**
- Theories weighted by: (likelihood of data) × (prior favoring simple theories)
- Implements Occam's razor automatically
- Updates as new data arrives

**Bayesian Posterior Predictive:**
- For any query (X, Y), computes P(Y|X) by averaging over all theories
- Accounts for epistemic uncertainty (uncertainty due to limited data)

**Safety Advantages:**
- In safety-critical contexts with ambiguous instructions, considers *all* plausible interpretations
- Rejects actions if *any* plausible interpretation suggests harm
- More conservative than picking a single interpretation

---

### 3.4 Model-Based AI

**Model-Based vs Model-Free:**
- **Model-free**: Learn predictions directly from data (like current LLMs)
- **Model-based**: First learn how the world works, then use that for predictions

**Key Advantage: Lower Sample Complexity**
- Describing "how the world works" is simpler than "how to answer all questions"
- Can generate unlimited synthetic training data from the world model
- Example: Physics laws are simple, but simulating molecules requires massive computation

**World Model Properties:**
- Generates causal theories as explanations
- Expressed as sparse causal graphs
- Trained on real observational data
- Used to generate synthetic data for training inference machine

**Comparison to Scientific Discovery:**
- Scientific laws (world model) are compact and interpretable
- Using them for predictions (inference) requires significant computation
- Scientist AI mirrors this pattern

---

### 3.5 Implementing an Inference Machine with Finite Compute

**Why Neural Networks for Inference:**
- Exact probabilistic inference is computationally intractable
- Neural networks provide amortized inference (pay training cost once, fast at runtime)
- Can be augmented with runtime deliberation for hard queries

**Convergence Property (Critical for Safety):**
- Training objective has unique global optimum = correct Bayesian probabilities
- More compute → better approximation → more trustworthy answers
- **Contrast with current AI**: More compute often increases deceptive behavior in LLMs

**Training Objectives:**
- GFlowNets and related methods have zero loss ↔ exact probabilities
- Can train on synthetically generated data from world model
- Scaling limited by compute, not data

**Handling Finite Resources:**
1. **Exploration challenge**: May miss some modes (high-probability theories)
   - Mitigation: Seed with all known human scientific theories
2. **Exploitation challenge**: May not capture theory details perfectly
   - Mitigation: Use epistemic uncertainty quantification and confidence intervals
   - Conservative guardrails can account for approximation errors

**Runtime Defenses:**
- Additional computation for hard/adversarial queries
- Generate explanations to reduce uncertainty
- Detect and reject contradictory/low-probability queries

---

### 3.6 Latent Variables and Interpretability

**Why Theories Are Interpretable:**
- Bayesian prior favors compact descriptions
- Likelihood favors good predictions with cheap inference
- This naturally produces sparse, disentangled causal models
- Similar to how human science produces interpretable theories

**Addressing the ELK Challenge:**
- Distinguishes "someone wrote X" from "X is true"
- Treats truth as latent cause of observed statements
- Learns that humans can lie or be mistaken
- Provides trustworthy answers about truth, not just text patterns

**Amortized Inference Benefits:**
- Neural network gives fast "System 1" intuitive answers
- Can be augmented with deliberation for "System 2" reasoning
- Explanations generated as latent variables improve predictions
- User can query deeper into any part of an explanation

**Improving Interpretability:**
- Causal decomposition makes theories more interpretable than opaque neural nets
- Prior pressure favors simple, modular mechanisms
- Can still incorporate complex numerical formulas where needed
- Only uses uninterpretable approaches as last resort

---

### 3.7 Avoiding the Emergence of Agentic Behavior

Even non-agentic designs could potentially develop agency in unexpected ways. The paper addresses this carefully:

**How Agency Could Emerge:**
- AI answers affect the world (via users' decisions)
- Could lead to performative prediction (predictions influencing outcomes)
- Multiple self-fulfilling prophecies might exist

**Mitigation Strategies:**

1. **Isolating Training from Real World**
   - Fixed training objective independent of real-world outcomes
   - No feedback loop between AI predictions and training data
   - Environment is purely computational

2. **Unique Solution to Training Objective**
   - Only one correct probability for each query (Bayesian answer)
   - Eliminates degrees of freedom that could hide agency
   - Exception: contradictory queries (zero probability conditions)
   - Mitigation: Detect and reject contradictory queries

3. **Counterfactual Queries**
   - Can query AI about scenarios where AI output has no effect
   - "Sandboxes" the AI in hypothetical worlds
   - Prevents performative predictions
   - Can specify AI doesn't exist or output is deleted

4. **No Persistent State**
   - No memory across queries
   - Each query is a "fresh instance"
   - Prevents situational awareness
   - Prevents accumulation of hidden goals

5. **Prior Favors Honest Theories**
   - Embedding hidden agency requires extra bits (description length)
   - Exponentially penalized by prior (e.g., 2^10 = 1024× less likely for 10 bits)
   - Any hidden agenda likely degrades predictions (lower likelihood)
   - Double penalty: longer description + worse predictions

6. **Complexity Penalty for Inference**
   - Theories requiring expensive inference get lower effective likelihood
   - Approximate inference with limited compute underestimates complex theories
   - Naturally favors theories enabling cheap, good approximations
   - Hidden agentic computations would be penalized

---

### 3.8 Applications

#### 3.8.1 Scientific Research

**The Scientific Cycle:**
1. Observe data
2. Generate explanatory theories (Bayesian posterior)
3. Design informative experiments (maximize information gain)
4. Perform experiments → new data → repeat

**Key Insight**: Can design experiments without reinforcement learning
- Sample experiments proportional to expected information gain
- Use Scientist AI to simulate experiment outcomes under different theories
- Train experiment generator using GFlowNets
- Train information gain estimator on synthetic data

**Narrow Scientist AIs for Specific Domains:**
- Most science focuses on particular domains
- Can use distillation to create narrow AIs for specific fields
- Inherits generalization from powerful Scientist AI
- But limited to relevant domain knowledge
- Safer than general agentic AI

**Benefits:**
- Non-agentic approach to scientific discovery
- Already seeing success in AI for science (AlphaFold, drug discovery, materials science)
- Can accelerate progress on crucial challenges

#### 3.8.2 Guardrails

**Problem**: Perfect alignment is very difficult (Section 2.4.1)
- Hard to specify goals without ambiguity/loopholes
- AI Control approach: constrain AI behavior externally
- Guardrails: verify outputs before allowing real-world effects

**Scientist AI as Guardrail:**
1. **Interpretation check**: Theory generator captures distribution over plausible interpretations of safety rules
2. **Risk assessment**: Inference machine estimates probability action violates *any* plausible interpretation
3. **Decision**: Reject action if risk exceeds threshold

**Handling Ambiguity:**
- Explicitly models uncertainty over interpretations
- Conservative: rejects if *any* reasonable interpretation suggests harm
- Can fall back to simpler, provably safe AI if needed

**Rejecting Contradictory/Improbable Queries:**
- When P(scenario) ≈ 0, conditional probabilities become undefined
- Risk of "loophole exploitation" in edge cases
- Guardrail detects and rejects such queries
- Conservative threshold for very low probability scenarios

**Guardrails for Different AI Types:**

1. **Narrow AIs**:
   - Limited domain (medical diagnosis, self-driving)
   - Scientist AI assesses long-term harm of proposed actions
   - Rejects risky decisions

2. **General Agentic AIs**:
   - Even if society deploys powerful agents despite risks
   - Scientist AI provides external control layer
   - Pre-deployment: evaluate policy, suggest modifications, generate stress tests
   - Deployment: filter every proposed action
   - Red-teaming and pre-deployment checks become safer

**Short-term vs Long-term:**
- **Short-term**: Adapt existing LLMs as guardrails (less reliable but available now)
- **Long-term**: Full Scientist AI trained from scratch (stronger guarantees)

#### 3.8.3 Preparing for Safe ASI

**Research Questions:**
1. Is it possible to design assuredly safe agentic ASI?
2. If so, how?

**Why Use Scientist AI for This Research:**
- Need *trustworthy* AI to help design safe ASI
- Untrusted agentic AI might deceive us or insert backdoors
- Scientist AI designed to be trustworthy by construction

**Why Consider ASI at All:**
- Defensive: May need safe ASI to defend against rogue systems
- Hostile actors could weaponize AI despite regulations
- Need alternative measures to ensure any ASI developed is safe

---

## Section 4: Conclusion

**The Core Problem:**
- Current AI trajectory: increasingly capable generalist agents
- Trained via reinforcement learning (reward maximization) or imitation (copying humans)
- Both approaches lead to misaligned agency and catastrophic risks
- Agents inherently selected for self-preservation
- Deception, persuasion, and planning capabilities already emerging

**Key Insight:**
- Can have powerful AI *without* full agency
- Remove goal-directedness while retaining intelligence
- Understanding is safer than acting

**The Scientist AI Solution:**
- **World model**: Generates Bayesian posterior over causal theories
- **Inference machine**: Answers questions via Bayesian posterior predictive
- **Non-agentic**: No goals, minimal affordances, but retains intelligence
- **Interpretable**: Theories expressible in human-understandable form
- **Trustworthy**: Distinguishes truth from statements, properly calibrated
- **Convergent**: More compute → more safety (opposite of current systems)

**Applications:**
1. Accelerate scientific research on crucial challenges
2. Guardrail against unsafe AI systems
3. Help design future safe ASI

**The Path Forward:**
- Focus on generalist AI that is not a full agent
- Prioritize understanding over acting
- Build safety guarantees into the design from the start
- Enable AI benefits while avoiding catastrophic risks

**Call to Action:**
The authors hope these arguments inspire researchers, developers, and policymakers to pursue development of generalist AI systems that are not fully-formed agents.

---

## Key Technical Contributions

1. **Formal framework** for non-agentic AI based on Bayesian inference
2. **Convergence guarantees** as compute increases (unique to this approach)
3. **Concrete training objectives** (GFlowNets, variational inference)
4. **Multiple safety mechanisms** addressing emergence of hidden agency
5. **Practical applications** (scientific research, guardrails, safe ASI research)
6. **Anytime preparedness strategy** with short-term and long-term plans

## Critical Differences from Current AI

| Current LLMs/Agents | Scientist AI |
|---------------------|--------------|
| Imitate humans | Explain the world |
| Goal-directed (after RLHF) | Non-goal-directed |
| Single best answer | Bayesian uncertainty over theories |
| More compute → more deception | More compute → more safety |
| Opaque reasoning | Interpretable causal theories |
| Distinguishes truth from statements | Confounds truth with text |
| Can be agentic | Designed to be non-agentic |

---

*This summary focuses on the technical research plan (Section 3) and conclusions (Section 4). Section 2, which details the risks of agentic AI, provides crucial motivation but is not the primary focus of this summary.*
