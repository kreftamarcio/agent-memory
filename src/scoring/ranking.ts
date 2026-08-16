/**
 * Multi-factor retrieval ranking.
 *
 * The central trade-off in agent memory is relevance against recency:
 *
 *   - Pure semantic similarity returns a perfectly relevant fact from six months
 *     ago that has since been superseded.
 *   - Pure recency returns the last thing said, regardless of whether it answers
 *     the question.
 *
 * Neither is correct in general, so both are weighted, alongside write-time
 * importance and current trace strength. The right balance is task-dependent, which
 * is why weights are a parameter rather than a constant.
 */

import type { MemoryStore, MemoryTrace } from './decay.js';

export interface RankingWeights {
  /** Semantic similarity to the query. */
  relevance: number;
  /** How recently the memory was created. */
  recency: number;
  /** Write-time importance. */
  importance: number;
  /** Current decayed strength. */
  strength: number;
}

export interface RankingConfig {
  weights: RankingWeights;
  maxResults: number;
  /** Traces below this effective strength are treated as forgotten. */
  minStrength: number;
  /** Half-life for the recency term, independent of decay. Defaults to 14 days. */
  recencyHalfLifeDays?: number;
  /** Only consider these stores. Defaults to all. */
  stores?: MemoryStore[];
}

export interface Candidate extends MemoryTrace {
  content: string;
  /** Cosine similarity in [0,1], supplied by the vector store. */
  relevance: number;
  /** Token count, needed for budgeted assembly. */
  tokenCount: number;
  /** Set when this trace has been superseded by a correction. */
  supersededBy?: string;
}

export interface ScoredCandidate extends Candidate {
  score: number;
  components: {
    relevance: number;
    recency: number;
    importance: number;
    strength: number;
  };
}

export interface AssembledContext {
  memories: ScoredCandidate[];
  tokensUsed: number;
  /** Relevant memories that did not fit the budget. */
  omittedCount: number;
  /** Excluded because they decayed below minStrength. */
  forgottenCount: number;
  /** Excluded because a correction superseded them. */
  supersededCount: number;
}

const DEFAULT_RECENCY_HALF_LIFE_DAYS = 14;
const MS_PER_DAY = 86_400_000;

export class RetrievalRanker {
  private readonly config: RankingConfig;
  private readonly weights: RankingWeights;
  private readonly recencyLambda: number;

  constructor(config: RankingConfig) {
    this.config = config;
    // Normalised rather than rejected when they do not sum to 1. An operator
    // writing 5:2:2:1 clearly means a ratio, and honouring the intent beats a
    // config error over arithmetic we can do ourselves.
    this.weights = this.normalizeWeights(config.weights);

    const halfLife = config.recencyHalfLifeDays ?? DEFAULT_RECENCY_HALF_LIFE_DAYS;
    if (halfLife <= 0) {
      throw new Error(`recencyHalfLifeDays must be positive, received ${halfLife}`);
    }
    this.recencyLambda = Math.LN2 / halfLife;
  }

  /**
   * Score and order candidates.
   *
   * Filtering happens before scoring: a superseded or forgotten trace should not
   * compete for a slot, and scoring it wastes work on a result that is discarded.
   */
  rank(
    candidates: Candidate[],
    effectiveStrength: (trace: MemoryTrace) => number,
    now: number = Date.now(),
  ): { scored: ScoredCandidate[]; forgottenCount: number; supersededCount: number } {
    const allowedStores = this.config.stores;
    let forgottenCount = 0;
    let supersededCount = 0;
    const scored: ScoredCandidate[] = [];

    for (const candidate of candidates) {
      if (allowedStores && !allowedStores.includes(candidate.store)) continue;

      // A correction supersedes rather than duplicates. Returning both would let
      // the agent read a fact and its retraction as equally valid, which is worse
      // than returning neither.
      if (candidate.supersededBy !== undefined) {
        supersededCount++;
        continue;
      }

      const strength = effectiveStrength(candidate);
      if (strength < this.config.minStrength) {
        forgottenCount++;
        continue;
      }

      const components = {
        relevance: this.clamp01(candidate.relevance),
        recency: this.recencyScore(candidate.createdAt, now),
        importance: this.clamp01(candidate.importance),
        strength,
      };

      const score =
        this.weights.relevance * components.relevance +
        this.weights.recency * components.recency +
        this.weights.importance * components.importance +
        this.weights.strength * components.strength;

      scored.push({ ...candidate, score, components });
    }

    scored.sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      // Deterministic tie-break: newer first, then id. Without it, identical scores
      // order by input sequence, and recall becomes non-reproducible across runs.
      const byRecency = b.createdAt.getTime() - a.createdAt.getTime();
      return byRecency !== 0 ? byRecency : a.id.localeCompare(b.id);
    });

    return {
      scored: scored.slice(0, this.config.maxResults),
      forgottenCount,
      supersededCount,
    };
  }

  /**
   * Assemble a token-budgeted context window.
   *
   * Budgeted in tokens, not result count: ten short memories and ten long ones
   * consume wildly different context, and the budget the caller actually has is
   * tokens.
   *
   * A memory that does not fit is skipped rather than terminating the loop, because
   * a single very long memory should not block every shorter one behind it. This is
   * a greedy fill, not knapsack-optimal, and that is deliberate: the ordering is
   * already the point, and reordering by size to pack tighter would return lower
   * scoring memories ahead of higher scoring ones.
   */
  assemble(scored: ScoredCandidate[], tokenBudget: number): AssembledContext {
    if (tokenBudget <= 0) {
      throw new Error(`tokenBudget must be positive, received ${tokenBudget}`);
    }

    const memories: ScoredCandidate[] = [];
    let tokensUsed = 0;
    let omittedCount = 0;

    for (const candidate of scored) {
      if (tokensUsed + candidate.tokenCount <= tokenBudget) {
        memories.push(candidate);
        tokensUsed += candidate.tokenCount;
      } else {
        omittedCount++;
      }
    }

    return {
      memories,
      tokensUsed,
      omittedCount,
      forgottenCount: 0,
      supersededCount: 0,
    };
  }

  /**
   * Weight preset for a recency-dominant query, such as "what did we decide?".
   *
   * Strength is zeroed here on purpose: a frequently-reinforced old memory is
   * strong but not recent, and including strength would pull exactly the wrong
   * results into a recency query.
   */
  static recencyFocused(): RankingWeights {
    return { relevance: 0.3, recency: 0.6, importance: 0.1, strength: 0.0 };
  }

  /** Preset for a stable-fact query, such as "what database does the user use?". */
  static stabilityFocused(): RankingWeights {
    return { relevance: 0.5, recency: 0.05, importance: 0.25, strength: 0.2 };
  }

  /** Balanced default. */
  static balanced(): RankingWeights {
    return { relevance: 0.5, recency: 0.2, importance: 0.2, strength: 0.1 };
  }

  /**
   * Explain a ranking decision.
   *
   * Exists because "why did the agent recall that?" is otherwise unanswerable, and
   * an unexplainable memory system cannot be debugged when it surfaces something
   * irrelevant.
   */
  explain(candidate: ScoredCandidate): string {
    const contributions = [
      { name: 'relevance', value: this.weights.relevance * candidate.components.relevance },
      { name: 'recency', value: this.weights.recency * candidate.components.recency },
      { name: 'importance', value: this.weights.importance * candidate.components.importance },
      { name: 'strength', value: this.weights.strength * candidate.components.strength },
    ].sort((a, b) => b.value - a.value);

    const dominant = contributions[0]!;
    const breakdown = contributions
      .map((c) => `${c.name}=${c.value.toFixed(3)}`)
      .join(' ');

    return (
      `${candidate.id} (${candidate.store}) scored ${candidate.score.toFixed(3)}, ` +
      `driven by ${dominant.name}. ${breakdown}`
    );
  }

  /**
   * Exponential recency decay, independent of trace strength.
   *
   * Deliberately a separate half-life: "how recent is this" and "how strong is
   * this" are different questions. Sharing one parameter would make a reinforced
   * old memory look recent purely because it is well remembered.
   */
  private recencyScore(createdAt: Date, now: number): number {
    const elapsedDays = (now - createdAt.getTime()) / MS_PER_DAY;
    if (elapsedDays <= 0) return 1;
    return this.clamp01(Math.exp(-this.recencyLambda * elapsedDays));
  }

  private normalizeWeights(weights: RankingWeights): RankingWeights {
    const sum =
      weights.relevance + weights.recency + weights.importance + weights.strength;

    if (sum <= 0) {
      throw new Error(
        'Ranking weights must sum to a positive value. All-zero weights would make ' +
          'every candidate score identically, reducing retrieval to input order.',
      );
    }

    return {
      relevance: weights.relevance / sum,
      recency: weights.recency / sum,
      importance: weights.importance / sum,
      strength: weights.strength / sum,
    };
  }

  private clamp01(value: number): number {
    if (Number.isNaN(value)) return 0;
    return Math.min(1, Math.max(0, value));
  }
}
