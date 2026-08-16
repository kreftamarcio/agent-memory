/**
 * agent-memory: long-term memory for AI agents.
 *
 * Three stores, one decay model:
 *   episodic   -> what happened   (events, timestamped, consolidates over time)
 *   semantic   -> what is true    (subject-predicate-object facts, confidence-scored)
 *   procedural -> how to do it    (versioned procedures with success tracking)
 *
 * The stores are independent and usable on their own. MemoryManager exists because
 * most agents need all three at once and should not have to fan out three queries and
 * merge the results by hand on every turn.
 *
 * Relative imports carry explicit .js extensions because moduleResolution is NodeNext,
 * which requires them. Omitting them is a compile error, not a style choice.
 */

import { EpisodicStore } from './stores/episodic.store.js';
import { SemanticStore } from './stores/semantic.store.js';
import { ProceduralStore } from './stores/procedural.store.js';
import { DecayEngine } from './scoring/decay.js';
import { RetrievalRanker } from './scoring/ranking.js';

import type {
  Episode,
  EpisodeContext,
  EpisodicQuery,
  EpisodicStoreConfig,
} from './stores/episodic.store.js';
import type {
  Fact,
  FactSource,
  SemanticQuery,
  SemanticStoreConfig,
  ContradictionReport,
} from './stores/semantic.store.js';
import type {
  Procedure,
  ProcedureStep,
  ProcedurePerformance,
  ProceduralQuery,
  ProceduralStoreConfig,
} from './stores/procedural.store.js';
import type {
  DecayProfile,
  DecayConfig,
  MemoryTrace,
  MemoryStore as MemoryStoreKind,
} from './scoring/decay.js';
import type {
  RankingWeights,
  RankingConfig,
  Candidate,
  ScoredCandidate,
  AssembledContext,
} from './scoring/ranking.js';

export { EpisodicStore, SemanticStore, ProceduralStore, DecayEngine, RetrievalRanker };

export type {
  Episode,
  EpisodeContext,
  EpisodicQuery,
  EpisodicStoreConfig,
  Fact,
  FactSource,
  SemanticQuery,
  SemanticStoreConfig,
  ContradictionReport,
  Procedure,
  ProcedureStep,
  ProcedurePerformance,
  ProceduralQuery,
  ProceduralStoreConfig,
  DecayProfile,
  DecayConfig,
  MemoryTrace,
  MemoryStoreKind,
  RankingWeights,
  RankingConfig,
  Candidate,
  ScoredCandidate,
  AssembledContext,
};

/**
 * Defaults chosen so each store outlives the one below it: episodes fade in days,
 * facts in months, procedures in about a year.
 *
 * The ordering is not cosmetic. DecayEngine rejects a semantic half-life shorter than
 * the episodic one, because a fact must outlive the event that produced it.
 */
export const DEFAULT_DECAY_PROFILE: DecayProfile = {
  episodic: { halfLifeDays: 3 },
  semantic: { halfLifeDays: 90 },
  procedural: { halfLifeDays: 365, strengthenOnUse: true },
  reinforcementAlpha: 0.3,
};

export interface MemoryManagerConfig {
  /**
   * Shared embedding function, passed to all three stores so a single provider change
   * does not require touching each one.
   */
  embed?: (text: string) => Promise<number[]>;
  /** Used by episodic consolidation. Without it, episodes accumulate unsummarized. */
  summarize?: (episodes: Episode[]) => Promise<string>;
  decay?: DecayProfile;
  episodic?: Omit<EpisodicStoreConfig, 'embed' | 'summarize'>;
  semantic?: Omit<SemanticStoreConfig, 'embed'>;
  procedural?: Omit<ProceduralStoreConfig, 'embed'>;
}

export interface RecallOptions {
  /** Natural-language description of what the agent is trying to do. */
  goal: string;
  /** Precomputed vector for `goal`. Supply it to avoid embedding twice. */
  goalEmbedding?: number[];
  episodeLimit?: number;
  factLimit?: number;
  procedureLimit?: number;
  /** Only return procedures with a proven track record. */
  minSuccessRate?: number;
}

export interface RecallResult {
  episodes: Episode[];
  facts: Fact[];
  procedures: Procedure[];
}

export class MemoryManager {
  readonly episodic: EpisodicStore;
  readonly semantic: SemanticStore;
  readonly procedural: ProceduralStore;
  readonly decay: DecayEngine;

  constructor(config: MemoryManagerConfig = {}) {
    const { embed, summarize } = config;

    this.episodic = new EpisodicStore({ ...config.episodic, embed, summarize });
    this.semantic = new SemanticStore({ ...config.semantic, embed });
    this.procedural = new ProceduralStore({ ...config.procedural, embed });
    this.decay = new DecayEngine(config.decay ?? DEFAULT_DECAY_PROFILE);
  }

  /**
   * Query all three stores for a single goal.
   *
   * Runs in parallel because the stores are independent: sequential fan-out would add
   * the embedding round-trips together for no reason. If one store has no embedding
   * function configured it degrades to its structural or success-rate ordering rather
   * than failing the whole recall.
   */
  async recall(options: RecallOptions): Promise<RecallResult> {
    const {
      goal,
      goalEmbedding,
      episodeLimit = 5,
      factLimit = 10,
      procedureLimit = 3,
      minSuccessRate,
    } = options;

    const [episodes, facts, procedures] = await Promise.all([
      this.episodic.retrieve({
        query: goal,
        queryEmbedding: goalEmbedding,
        limit: episodeLimit,
      }),
      this.semantic.query({
        query: goal,
        queryEmbedding: goalEmbedding,
        limit: factLimit,
      }),
      this.procedural.findForGoal({
        goal,
        goalEmbedding,
        limit: procedureLimit,
        minSuccessRate,
      }),
    ]);

    return { episodes, facts, procedures };
  }

  /**
   * Promote an episode into a durable fact.
   *
   * This is the bridge between "the user told me X once" and "X is true". Kept
   * explicit rather than automatic: silently inferring facts from single events is how
   * agents end up confidently repeating noise.
   *
   * Contradictions are RETURNED rather than resolved. Silently overwriting is how an
   * agent becomes confidently wrong, and a disagreement between sources is information
   * the caller usually wants to act on.
   */
  async promoteToFact(
    episodeId: string,
    triple: { subject: string; predicate: string; object: string },
    confidence: number,
  ): Promise<{ fact: Fact; contradictions: ContradictionReport[] }> {
    if (confidence < 0 || confidence > 1) {
      throw new RangeError(`confidence must be within [0,1], got ${confidence}`);
    }

    const result = await this.semantic.add({
      ...triple,
      confidence,
      source: { type: 'observation', episodeId },
    });

    // The source episode is reinforced, because an episode that justified a durable
    // fact has demonstrated its value and should not decay at the same rate as one
    // that was never used again. Previously this method validated its input and then
    // ignored the decay engine entirely.
    this.episodic.touch?.(episodeId);

    return result;
  }

  /** Aggregate counters from every store. Useful for dashboards and evals. */
  stats(): {
    episodic: ReturnType<EpisodicStore['getStats']>;
    semantic: ReturnType<SemanticStore['getStats']>;
    procedural: ReturnType<ProceduralStore['getStats']>;
  } {
    return {
      episodic: this.episodic.getStats(),
      semantic: this.semantic.getStats(),
      procedural: this.procedural.getStats(),
    };
  }
}

/**
 * Weight presets for the retrieval ranker.
 *
 * Exported as named presets because the correct balance is task-dependent, and a
 * single default silently produces the wrong answer for half of all queries.
 */
export const RANKING_PRESETS = {
  /** "What did we decide last week?" Strength is zero on purpose: a frequently
   *  reinforced old memory is strong but not recent, and including it would pull
   *  exactly the wrong results into a recency query. */
  recencyFocused: RetrievalRanker.recencyFocused,
  /** "What database does the user use?" Favours relevance and stability. */
  stabilityFocused: RetrievalRanker.stabilityFocused,
  balanced: RetrievalRanker.balanced,
} as const;
