/**
 * Episodic Memory Store: records specific events and interactions.
 *
 * Modeled after human episodic memory: stores "what happened, when, in what context."
 * Each episode captures a complete interaction with temporal and contextual metadata.
 *
 * Key behaviors:
 *   - Events are immutable once stored (append-only)
 *   - Temporal ordering is preserved
 *   - Retrieval supports time-range queries and similarity search
 *   - Consolidation: old episodes are compressed/summarized over time
 *   - Forgetting curve: access recency affects retrieval ranking
 */

export interface Episode {
  id: string;
  timestamp: number;
  type: 'interaction' | 'observation' | 'action' | 'outcome';
  content: string;
  context: EpisodeContext;
  embedding?: number[];
  importance: number;
  accessCount: number;
  lastAccessed: number;
  consolidated: boolean;
  tags: string[];
}

export interface EpisodeContext {
  agentId: string;
  userId?: string;
  sessionId?: string;
  location?: string;
  emotionalValence?: number; // -1 to 1
  relatedEpisodes?: string[];
}

export interface EpisodicQuery {
  /** Semantic search query */
  query?: string;
  queryEmbedding?: number[];
  /** Time range filter */
  after?: number;
  before?: number;
  /** Filter by type */
  type?: Episode['type'];
  /** Filter by context */
  userId?: string;
  sessionId?: string;
  tags?: string[];
  /** Result limit */
  limit?: number;
  /** Weighting between recency and relevance (0 = pure relevance, 1 = pure recency) */
  recencyBias?: number;
}

export interface EpisodicStoreConfig {
  /** Maximum episodes before triggering consolidation */
  maxEpisodes?: number;
  /** Episodes older than this (ms) are candidates for consolidation */
  consolidationAge?: number;
  /** Embedding function for semantic search */
  embed?: (text: string) => Promise<number[]>;
  /** Summarization function for consolidation */
  summarize?: (episodes: Episode[]) => Promise<string>;
}

export class EpisodicStore {
  private episodes: Episode[] = [];
  private readonly maxEpisodes: number;
  private readonly consolidationAge: number;

  constructor(private readonly config: EpisodicStoreConfig = {}) {
    this.maxEpisodes = config.maxEpisodes ?? 10_000;
    this.consolidationAge = config.consolidationAge ?? 7 * 24 * 60 * 60 * 1000; // 7 days
  }

  /**
   * Record a new episode.
   */
  async add(episode: Omit<Episode, 'id' | 'accessCount' | 'lastAccessed' | 'consolidated'>): Promise<Episode> {
    const newEpisode: Episode = {
      ...episode,
      id: crypto.randomUUID(),
      accessCount: 0,
      lastAccessed: Date.now(),
      consolidated: false,
    };

    // Generate embedding if function is available and not provided
    if (!newEpisode.embedding && this.config.embed) {
      newEpisode.embedding = await this.config.embed(newEpisode.content);
    }

    this.episodes.push(newEpisode);

    // Trigger consolidation if needed
    if (this.episodes.length > this.maxEpisodes) {
      await this.consolidate();
    }

    return newEpisode;
  }

  /**
   * Retrieve episodes matching a query.
   * Uses combined scoring: relevance + recency + importance + access frequency.
   */
  async retrieve(query: EpisodicQuery): Promise<Episode[]> {
    let candidates = [...this.episodes];

    // Apply filters
    if (query.after) candidates = candidates.filter(e => e.timestamp >= query.after!);
    if (query.before) candidates = candidates.filter(e => e.timestamp <= query.before!);
    if (query.type) candidates = candidates.filter(e => e.type === query.type);
    if (query.userId) candidates = candidates.filter(e => e.context.userId === query.userId);
    if (query.sessionId) candidates = candidates.filter(e => e.context.sessionId === query.sessionId);
    if (query.tags?.length) {
      candidates = candidates.filter(e => query.tags!.some(t => e.tags.includes(t)));
    }

    // Score and rank
    const recencyBias = query.recencyBias ?? 0.3;
    const now = Date.now();

    let queryEmbedding = query.queryEmbedding;
    if (!queryEmbedding && query.query && this.config.embed) {
      queryEmbedding = await this.config.embed(query.query);
    }

    const scored = candidates.map(episode => {
      // Relevance score (embedding similarity)
      let relevanceScore = 0;
      if (queryEmbedding && episode.embedding) {
        relevanceScore = this.cosineSimilarity(queryEmbedding, episode.embedding);
      }

      // Recency score (exponential decay)
      const ageMs = now - episode.timestamp;
      const recencyScore = Math.exp(-ageMs / (this.consolidationAge / 2));

      // Importance score (normalized)
      const importanceScore = episode.importance;

      // Combined score
      const score =
        (1 - recencyBias) * relevanceScore +
        recencyBias * recencyScore +
        0.1 * importanceScore;

      return { episode, score };
    });

    // Sort by score descending
    scored.sort((a, b) => b.score - a.score);

    // Limit results
    const limit = query.limit ?? 10;
    const results = scored.slice(0, limit).map(s => s.episode);

    // Update access metadata
    for (const episode of results) {
      episode.accessCount++;
      episode.lastAccessed = now;
    }

    return results;
  }

  /**
   * Consolidate old episodes: summarize groups and replace with compressed versions.
   * Mimics human memory consolidation during sleep.
   */
  async consolidate(): Promise<{ consolidated: number; remaining: number }> {
    const now = Date.now();
    const oldEpisodes = this.episodes.filter(
      e => !e.consolidated && (now - e.timestamp) > this.consolidationAge,
    );

    if (oldEpisodes.length === 0 || !this.config.summarize) {
      return { consolidated: 0, remaining: this.episodes.length };
    }

    // Group old episodes by day
    const groups = this.groupByDay(oldEpisodes);

    for (const [_day, group] of groups) {
      if (group.length < 3) continue; // Don't consolidate very small groups

      // Summarize the group
      const summary = await this.config.summarize(group);
      const avgImportance = group.reduce((sum, e) => sum + e.importance, 0) / group.length;

      // Create consolidated episode
      const consolidated: Episode = {
        id: crypto.randomUUID(),
        timestamp: group[0]!.timestamp,
        type: 'observation',
        content: summary,
        context: group[0]!.context,
        importance: avgImportance,
        accessCount: 0,
        lastAccessed: now,
        consolidated: true,
        tags: [...new Set(group.flatMap(e => e.tags))],
      };

      if (this.config.embed) {
        consolidated.embedding = await this.config.embed(summary);
      }

      // Replace originals with consolidated version
      const idsToRemove = new Set(group.map(e => e.id));
      this.episodes = this.episodes.filter(e => !idsToRemove.has(e.id));
      this.episodes.push(consolidated);
    }

    return { consolidated: oldEpisodes.length, remaining: this.episodes.length };
  }

  getStats(): { total: number; consolidated: number; oldestTimestamp: number } {
    return {
      total: this.episodes.length,
      consolidated: this.episodes.filter(e => e.consolidated).length,
      oldestTimestamp: this.episodes.length > 0
        ? Math.min(...this.episodes.map(e => e.timestamp))
        : 0,
    };
  }

  private groupByDay(episodes: Episode[]): Map<string, Episode[]> {
    const groups = new Map<string, Episode[]>();
    for (const episode of episodes) {
      const day = new Date(episode.timestamp).toISOString().split('T')[0]!;
      const group = groups.get(day) ?? [];
      group.push(episode);
      groups.set(day, group);
    }
    return groups;
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
