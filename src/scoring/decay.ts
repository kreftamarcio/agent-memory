/**
 * Temporal decay and access reinforcement for memory traces.
 *
 * Model:
 *   strength(t) = strength0 * exp(-lambda * dt)
 *   lambda      = ln(2) / halfLife
 *
 * Half-life parameterization is used instead of a raw decay constant because
 * it is directly interpretable: halfLifeDays = 7 means a trace at strength 1.0
 * sits at 0.5 after one untouched week. A raw lambda of 0.099 conveys nothing
 * to whoever is configuring the system.
 */

export type MemoryStore = 'episodic' | 'semantic' | 'procedural';

export interface DecayConfig {
  halfLifeDays: number;
  /** Procedural memories can strengthen with use rather than only decay. */
  strengthenOnUse?: boolean;
}

export interface DecayProfile {
  episodic: DecayConfig;
  semantic: DecayConfig;
  procedural: DecayConfig;
  /**
   * Reinforcement coefficient in (0,1). Higher means retrieval strengthens
   * a trace more per access. Default 0.3.
   */
  reinforcementAlpha?: number;
}

export interface MemoryTrace {
  id: string;
  store: MemoryStore;
  /** Current strength in [0,1] as of `lastAccessedAt`. */
  strength: number;
  /** Write-time importance in [0,1]. Acts as a decay floor. */
  importance: number;
  createdAt: Date;
  lastAccessedAt: Date;
  accessCount: number;
}

const MS_PER_DAY = 86_400_000;
const DEFAULT_ALPHA = 0.3;

export class DecayEngine {
  private readonly profile: DecayProfile;
  private readonly alpha: number;
  private readonly lambdaByStore: Record<MemoryStore, number>;

  constructor(profile: DecayProfile) {
    this.validate(profile);
    this.profile = profile;
    this.alpha = profile.reinforcementAlpha ?? DEFAULT_ALPHA;

    // Precompute lambda per store. These never change at runtime, and decay
    // is evaluated on every retrieval over every candidate.
    this.lambdaByStore = {
      episodic: Math.LN2 / profile.episodic.halfLifeDays,
      semantic: Math.LN2 / profile.semantic.halfLifeDays,
      procedural: Math.LN2 / profile.procedural.halfLifeDays,
    };
  }

  /**
   * Effective strength of a trace at a given moment.
   *
   * Importance acts as a floor: a trace written with importance 0.9 never
   * decays below 0.9 * floorFactor, so a critical fact does not silently
   * vanish because nobody happened to query it for a month.
   */
  currentStrength(trace: MemoryTrace, now: Date = new Date()): number {
    const elapsedDays = (now.getTime() - trace.lastAccessedAt.getTime()) / MS_PER_DAY;

    if (elapsedDays <= 0) {
      return this.clamp(trace.strength);
    }

    const lambda = this.lambdaByStore[trace.store];
    const decayed = trace.strength * Math.exp(-lambda * elapsedDays);

    // Importance-derived floor. The 0.5 factor means even maximum importance
    // does not fully pin a memory, it only slows its disappearance.
    const floor = trace.importance * 0.5;

    return this.clamp(Math.max(decayed, floor));
  }

  /**
   * Apply access reinforcement.
   *
   *   strength' = min(1, strength + alpha * (1 - strength))
   *
   * Deliberately sub-linear. Linear reinforcement lets a trace retrieved fifty
   * times become permanently pinned at 1.0, which turns the store into a cache
   * that never evicts its oldest entries. The (1 - strength) factor makes each
   * access contribute less than the last, approaching 1 asymptotically.
   *
   * Pattern follows the retrieval-practice effect from spaced-repetition
   * research: recall strengthens the trace, and the effect saturates.
   */
  reinforce(trace: MemoryTrace, now: Date = new Date()): MemoryTrace {
    const current = this.currentStrength(trace, now);

    const isProcedural = trace.store === 'procedural';
    const strengthensOnUse = this.profile[trace.store].strengthenOnUse ?? isProcedural;

    // Procedural memories that strengthen on use get a larger coefficient:
    // a procedure that keeps working should become the default path.
    const effectiveAlpha = strengthensOnUse ? this.alpha * 1.5 : this.alpha;

    const reinforced = current + effectiveAlpha * (1 - current);

    return {
      ...trace,
      strength: this.clamp(reinforced),
      lastAccessedAt: now,
      accessCount: trace.accessCount + 1,
    };
  }

  /**
   * Has this trace decayed below the retention threshold?
   * Callers treat `true` as "forgotten" and exclude it from recall.
   */
  isForgotten(trace: MemoryTrace, minStrength: number, now: Date = new Date()): boolean {
    return this.currentStrength(trace, now) < minStrength;
  }

  /**
   * Days until a trace decays to `targetStrength`, or null if it never will
   * because its importance floor sits at or above the target.
   *
   * Solving strength0 * exp(-lambda * t) = target for t:
   *   t = ln(strength0 / target) / lambda
   */
  daysUntilStrength(trace: MemoryTrace, targetStrength: number): number | null {
    if (targetStrength <= 0 || targetStrength >= 1) {
      throw new Error('targetStrength must be strictly between 0 and 1');
    }

    const floor = trace.importance * 0.5;
    if (floor >= targetStrength) {
      return null; // Protected by its importance floor
    }

    if (trace.strength <= targetStrength) {
      return 0; // Already there
    }

    const lambda = this.lambdaByStore[trace.store];
    return Math.log(trace.strength / targetStrength) / lambda;
  }

  /**
   * Batch evaluation for retrieval. Avoids constructing a Date per candidate,
   * which matters when ranking thousands of traces per query.
   */
  evaluateBatch(traces: MemoryTrace[], now: Date = new Date()): Array<MemoryTrace & { effectiveStrength: number }> {
    return traces.map(trace => ({
      ...trace,
      effectiveStrength: this.currentStrength(trace, now),
    }));
  }

  /**
   * Recency component for ranking, normalized to [0,1].
   *
   * Uses the same exponential form as decay but with an independent half-life,
   * because "how recent is this" and "how strong is this" are different
   * questions. A frequently-reinforced old memory is strong but not recent.
   */
  recencyScore(trace: MemoryTrace, recencyHalfLifeDays = 14, now: Date = new Date()): number {
    const elapsedDays = (now.getTime() - trace.createdAt.getTime()) / MS_PER_DAY;
    if (elapsedDays <= 0) return 1;

    const lambda = Math.LN2 / recencyHalfLifeDays;
    return this.clamp(Math.exp(-lambda * elapsedDays));
  }

  private clamp(value: number): number {
    if (Number.isNaN(value)) return 0;
    return Math.min(1, Math.max(0, value));
  }

  private validate(profile: DecayProfile): void {
    for (const store of ['episodic', 'semantic', 'procedural'] as const) {
      const halfLife = profile[store].halfLifeDays;
      if (!Number.isFinite(halfLife) || halfLife <= 0) {
        throw new Error(
          `decay.${store}.halfLifeDays must be a positive finite number, got ${halfLife}`,
        );
      }
    }

    const alpha = profile.reinforcementAlpha;
    if (alpha !== undefined && (alpha <= 0 || alpha >= 1)) {
      throw new Error(
        `reinforcementAlpha must be strictly between 0 and 1, got ${alpha}. ` +
        `Values at or above 1 would saturate a trace on a single access.`,
      );
    }

    // A semantic half-life shorter than episodic is almost certainly a mistake:
    // it would make stable facts evaporate faster than one-off events.
    if (profile.semantic.halfLifeDays < profile.episodic.halfLifeDays) {
      throw new Error(
        `semantic halfLifeDays (${profile.semantic.halfLifeDays}) is shorter than ` +
        `episodic (${profile.episodic.halfLifeDays}). Facts should outlive the events ` +
        `that produced them. If this is intentional, use a custom store instead.`,
      );
    }
  }
}
