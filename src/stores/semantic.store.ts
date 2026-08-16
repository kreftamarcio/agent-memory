/**
 * Semantic Memory Store: facts, concepts, and general knowledge.
 *
 * Unlike episodic memory (events), semantic memory stores declarative facts
 * independent of when/where they were learned. Think of it as the agent's
 * knowledge base that grows over time.
 *
 * Features:
 *   - Entity-relationship storage (subject, predicate, object triples)
 *   - Confidence scoring (facts can be uncertain)
 *   - Source tracking (where was this fact learned)
 *   - Contradiction detection (new facts vs existing beliefs)
 *   - Retrieval by entity, relation, or semantic similarity
 */

export interface Fact {
  id: string;
  subject: string;
  predicate: string;
  object: string;
  confidence: number;
  source: FactSource;
  createdAt: number;
  updatedAt: number;
  accessCount: number;
  embedding?: number[];
  metadata?: Record<string, unknown>;
}

export interface FactSource {
  type: 'observation' | 'inference' | 'user_stated' | 'external';
  episodeId?: string;
  reference?: string;
}

export interface SemanticQuery {
  /** Search by subject entity */
  subject?: string;
  /** Search by predicate/relation type */
  predicate?: string;
  /** Search by object entity */
  object?: string;
  /** Semantic similarity search */
  query?: string;
  queryEmbedding?: number[];
  /** Minimum confidence threshold */
  minConfidence?: number;
  /** Result limit */
  limit?: number;
}

export interface ContradictionReport {
  existingFact: Fact;
  newFact: Omit<Fact, 'id' | 'createdAt' | 'updatedAt' | 'accessCount'>;
  type: 'direct_contradiction' | 'confidence_conflict' | 'outdated';
  resolution: 'keep_existing' | 'replace' | 'keep_both';
}

export interface SemanticStoreConfig {
  embed?: (text: string) => Promise<number[]>;
  /** Strategy for handling contradictions */
  contradictionStrategy?: 'newest_wins' | 'highest_confidence' | 'keep_both' | 'ask_user';
}

export class SemanticStore {
  private facts: Fact[] = [];
  private readonly contradictionStrategy: string;

  constructor(private readonly config: SemanticStoreConfig = {}) {
    this.contradictionStrategy = config.contradictionStrategy ?? 'highest_confidence';
  }

  /**
   * Store a new fact. Checks for contradictions with existing knowledge.
   */
  async add(
    fact: Omit<Fact, 'id' | 'createdAt' | 'updatedAt' | 'accessCount'>,
  ): Promise<{ fact: Fact; contradictions: ContradictionReport[] }> {
    // Check for contradictions
    const contradictions = this.detectContradictions(fact);

    // Resolve contradictions
    for (const contradiction of contradictions) {
      contradiction.resolution = this.resolveContradiction(contradiction);

      if (contradiction.resolution === 'replace') {
        this.facts = this.facts.filter(f => f.id !== contradiction.existingFact.id);
      }
    }

    // If any contradiction says keep_existing and strategy isn't keep_both, skip
    const blocked = contradictions.some(
      c => c.resolution === 'keep_existing',
    );

    const newFact: Fact = {
      ...fact,
      id: crypto.randomUUID(),
      createdAt: Date.now(),
      updatedAt: Date.now(),
      accessCount: 0,
    };

    if (!blocked) {
      // Generate embedding
      if (!newFact.embedding && this.config.embed) {
        const factText = `${fact.subject} ${fact.predicate} ${fact.object}`;
        newFact.embedding = await this.config.embed(factText);
      }
      this.facts.push(newFact);
    }

    return { fact: newFact, contradictions };
  }

  /**
   * Query facts by structured or semantic search.
   */
  async query(params: SemanticQuery): Promise<Fact[]> {
    let candidates = [...this.facts];

    // Structural filters
    if (params.subject) {
      candidates = candidates.filter(f =>
        f.subject.toLowerCase().includes(params.subject!.toLowerCase()),
      );
    }
    if (params.predicate) {
      candidates = candidates.filter(f =>
        f.predicate.toLowerCase().includes(params.predicate!.toLowerCase()),
      );
    }
    if (params.object) {
      candidates = candidates.filter(f =>
        f.object.toLowerCase().includes(params.object!.toLowerCase()),
      );
    }
    if (params.minConfidence) {
      candidates = candidates.filter(f => f.confidence >= params.minConfidence!);
    }

    // Semantic search
    let queryEmbedding = params.queryEmbedding;
    if (!queryEmbedding && params.query && this.config.embed) {
      queryEmbedding = await this.config.embed(params.query);
    }

    if (queryEmbedding) {
      const scored = candidates
        .filter(f => f.embedding)
        .map(f => ({
          fact: f,
          score: this.cosineSimilarity(queryEmbedding!, f.embedding!),
        }))
        .sort((a, b) => b.score - a.score);

      candidates = scored.map(s => s.fact);
    }

    // Update access counts
    const limit = params.limit ?? 10;
    const results = candidates.slice(0, limit);
    for (const fact of results) {
      fact.accessCount++;
    }

    return results;
  }

  /**
   * Get all facts about a specific entity.
   */
  getEntityFacts(entity: string): Fact[] {
    const lower = entity.toLowerCase();
    return this.facts.filter(
      f => f.subject.toLowerCase() === lower || f.object.toLowerCase() === lower,
    );
  }

  /**
   * Update confidence of an existing fact (reinforcement or weakening).
   */
  updateConfidence(factId: string, delta: number): Fact | null {
    const fact = this.facts.find(f => f.id === factId);
    if (!fact) return null;

    fact.confidence = Math.max(0, Math.min(1, fact.confidence + delta));
    fact.updatedAt = Date.now();
    return fact;
  }

  getStats(): { totalFacts: number; avgConfidence: number; entities: number } {
    const entities = new Set<string>();
    let totalConfidence = 0;

    for (const fact of this.facts) {
      entities.add(fact.subject.toLowerCase());
      entities.add(fact.object.toLowerCase());
      totalConfidence += fact.confidence;
    }

    return {
      totalFacts: this.facts.length,
      avgConfidence: this.facts.length > 0 ? totalConfidence / this.facts.length : 0,
      entities: entities.size,
    };
  }

  private detectContradictions(
    newFact: Omit<Fact, 'id' | 'createdAt' | 'updatedAt' | 'accessCount'>,
  ): ContradictionReport[] {
    const contradictions: ContradictionReport[] = [];

    for (const existing of this.facts) {
      // Same subject and predicate but different object = potential contradiction
      if (
        existing.subject.toLowerCase() === newFact.subject.toLowerCase() &&
        existing.predicate.toLowerCase() === newFact.predicate.toLowerCase() &&
        existing.object.toLowerCase() !== newFact.object.toLowerCase()
      ) {
        contradictions.push({
          existingFact: existing,
          newFact,
          type: 'direct_contradiction',
          resolution: 'keep_both', // Will be resolved
        });
      }
    }

    return contradictions;
  }

  private resolveContradiction(contradiction: ContradictionReport): ContradictionReport['resolution'] {
    switch (this.contradictionStrategy) {
      case 'newest_wins':
        return 'replace';
      case 'highest_confidence':
        return contradiction.newFact.confidence > contradiction.existingFact.confidence
          ? 'replace'
          : 'keep_existing';
      case 'keep_both':
        return 'keep_both';
      default:
        return 'keep_both';
    }
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
