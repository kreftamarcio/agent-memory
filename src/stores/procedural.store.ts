/**
 * Procedural Memory Store: learned skills, workflows, and action patterns.
 *
 * Stores HOW to do things, not WHAT happened (episodic) or WHAT is true (semantic).
 * When an agent successfully completes a task, the procedure is stored here
 * for future reuse and refinement.
 *
 * Analogy: A chef's muscle memory for knife technique vs. recipe knowledge.
 *
 * Features:
 *   - Procedure versioning (skills improve over time)
 *   - Success/failure tracking per procedure
 *   - Context-dependent retrieval (match procedures to current situation)
 *   - Composition: combine simple procedures into complex workflows
 */

export interface Procedure {
  id: string;
  name: string;
  description: string;
  /** Ordered steps to execute */
  steps: ProcedureStep[];
  /** When this procedure applies */
  triggerConditions: string[];
  /** Track record */
  performance: ProcedurePerformance;
  /** Version history */
  version: number;
  previousVersionId?: string;
  /** Tags for categorization */
  tags: string[];
  createdAt: number;
  updatedAt: number;
  embedding?: number[];
}

export interface ProcedureStep {
  order: number;
  action: string;
  parameters?: Record<string, unknown>;
  expectedOutcome?: string;
  fallbackAction?: string;
  /** Estimated duration in ms */
  estimatedDuration?: number;
}

export interface ProcedurePerformance {
  totalExecutions: number;
  successCount: number;
  failureCount: number;
  avgDuration: number;
  lastExecuted: number;
  /** Exponential moving average of success rate */
  successRate: number;
}

export interface ProceduralQuery {
  /** Describe what you want to accomplish */
  goal?: string;
  goalEmbedding?: number[];
  /** Filter by tags */
  tags?: string[];
  /** Minimum success rate */
  minSuccessRate?: number;
  /** Only procedures executed at least N times */
  minExecutions?: number;
  limit?: number;
}

export interface ProceduralStoreConfig {
  embed?: (text: string) => Promise<number[]>;
  /** Minimum executions before success rate is trusted */
  maturityThreshold?: number;
}

export class ProceduralStore {
  private procedures: Procedure[] = [];
  private readonly maturityThreshold: number;

  constructor(private readonly config: ProceduralStoreConfig = {}) {
    this.maturityThreshold = config.maturityThreshold ?? 5;
  }

  /**
   * Store a new procedure or update an existing one.
   */
  async add(
    procedure: Omit<Procedure, 'id' | 'version' | 'createdAt' | 'updatedAt' | 'performance'>,
  ): Promise<Procedure> {
    const newProcedure: Procedure = {
      ...procedure,
      id: crypto.randomUUID(),
      version: 1,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      performance: {
        totalExecutions: 0,
        successCount: 0,
        failureCount: 0,
        avgDuration: 0,
        lastExecuted: 0,
        successRate: 0.5, // Prior: 50% before any data
      },
    };

    if (!newProcedure.embedding && this.config.embed) {
      const text = `${procedure.name}: ${procedure.description}`;
      newProcedure.embedding = await this.config.embed(text);
    }

    this.procedures.push(newProcedure);
    return newProcedure;
  }

  /**
   * Find procedures relevant to a goal.
   */
  async findForGoal(query: ProceduralQuery): Promise<Procedure[]> {
    let candidates = [...this.procedures];

    // Apply filters
    if (query.tags?.length) {
      candidates = candidates.filter(p => query.tags!.some(t => p.tags.includes(t)));
    }
    if (query.minSuccessRate !== undefined) {
      candidates = candidates.filter(p =>
        p.performance.totalExecutions >= this.maturityThreshold &&
        p.performance.successRate >= query.minSuccessRate!,
      );
    }
    if (query.minExecutions !== undefined) {
      candidates = candidates.filter(p => p.performance.totalExecutions >= query.minExecutions!);
    }

    // Semantic ranking
    let goalEmbedding = query.goalEmbedding;
    if (!goalEmbedding && query.goal && this.config.embed) {
      goalEmbedding = await this.config.embed(query.goal);
    }

    if (goalEmbedding) {
      const scored = candidates
        .filter(p => p.embedding)
        .map(p => ({
          procedure: p,
          relevance: this.cosineSimilarity(goalEmbedding!, p.embedding!),
        }))
        .sort((a, b) => b.relevance - a.relevance);

      candidates = scored.map(s => s.procedure);
    } else {
      // Sort by success rate (most reliable first)
      candidates.sort((a, b) => b.performance.successRate - a.performance.successRate);
    }

    return candidates.slice(0, query.limit ?? 5);
  }

  /**
   * Record execution outcome for a procedure.
   * Updates success rate using exponential moving average.
   */
  recordOutcome(
    procedureId: string,
    outcome: { success: boolean; duration: number },
  ): Procedure | null {
    const procedure = this.procedures.find(p => p.id === procedureId);
    if (!procedure) return null;

    const perf = procedure.performance;
    perf.totalExecutions++;
    perf.lastExecuted = Date.now();

    if (outcome.success) {
      perf.successCount++;
    } else {
      perf.failureCount++;
    }

    // Update average duration (running average)
    perf.avgDuration = perf.avgDuration + (outcome.duration - perf.avgDuration) / perf.totalExecutions;

    // Exponential moving average for success rate (alpha = 0.2)
    const alpha = 0.2;
    const successValue = outcome.success ? 1 : 0;
    perf.successRate = alpha * successValue + (1 - alpha) * perf.successRate;

    procedure.updatedAt = Date.now();
    return procedure;
  }

  /**
   * Create a new version of a procedure (skill improvement).
   */
  async evolve(
    procedureId: string,
    updates: Partial<Pick<Procedure, 'steps' | 'description' | 'triggerConditions'>>,
  ): Promise<Procedure | null> {
    const existing = this.procedures.find(p => p.id === procedureId);
    if (!existing) return null;

    const evolved: Procedure = {
      ...existing,
      ...updates,
      id: crypto.randomUUID(),
      version: existing.version + 1,
      previousVersionId: existing.id,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      performance: {
        totalExecutions: 0,
        successCount: 0,
        failureCount: 0,
        avgDuration: 0,
        lastExecuted: 0,
        successRate: existing.performance.successRate, // Inherit prior
      },
    };

    if (this.config.embed) {
      const text = `${evolved.name}: ${evolved.description}`;
      evolved.embedding = await this.config.embed(text);
    }

    this.procedures.push(evolved);
    return evolved;
  }

  /**
   * Compose multiple procedures into a workflow.
   */
  async compose(procedureIds: string[], name: string, description: string): Promise<Procedure | null> {
    const procedures = procedureIds
      .map(id => this.procedures.find(p => p.id === id))
      .filter((p): p is Procedure => p !== undefined);

    if (procedures.length === 0) return null;

    // Flatten steps from all procedures
    let order = 0;
    const composedSteps: ProcedureStep[] = procedures.flatMap(p =>
      p.steps.map(step => ({ ...step, order: order++ })),
    );

    return this.add({
      name,
      description,
      steps: composedSteps,
      triggerConditions: procedures.flatMap(p => p.triggerConditions),
      tags: [...new Set(procedures.flatMap(p => p.tags)), 'composed'],
    });
  }

  getStats(): { total: number; mature: number; avgSuccessRate: number } {
    const mature = this.procedures.filter(p => p.performance.totalExecutions >= this.maturityThreshold);
    const avgRate = mature.length > 0
      ? mature.reduce((sum, p) => sum + p.performance.successRate, 0) / mature.length
      : 0;

    return {
      total: this.procedures.length,
      mature: mature.length,
      avgSuccessRate: avgRate,
    };
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i]! * b[i]!;
      normA += a[i]! * a[i]!;
      normB += b[i]! * b[i]!;
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom === 0 ? 0 : dot / denom;
  }
}
