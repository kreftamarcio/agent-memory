import { z } from 'zod';

/**
 * Three memory systems, following the standard cognitive split:
 * - episodic:   "what happened" (timestamped events, decays fast)
 * - semantic:   "what is true" (distilled facts, decays slowly)
 * - procedural: "how to do it" (reusable routines, effectively permanent)
 */
export type MemoryKind = 'episodic' | 'semantic' | 'procedural';

export const MemoryRecordSchema = z.object({
  id: z.string().min(1),
  kind: z.enum(['episodic', 'semantic', 'procedural']),
  content: z.string().min(1),
  /** Optional dense vector. Absent records are retrievable by tag/keyword only. */
  embedding: z.array(z.number()).optional(),
  createdAt: z.number().int().nonnegative(),
  lastAccessedAt: z.number().int().nonnegative(),
  accessCount: z.number().int().nonnegative(),
  /** Author-assigned importance in [0,1]. Survives decay, unlike recency. */
  salience: z.number().min(0).max(1),
  tags: z.array(z.string()).default([]),
  /** Provenance: task id, conversation id, document id. Required for auditability. */
  sourceId: z.string().optional(),
});

export type MemoryRecord = z.infer<typeof MemoryRecordSchema>;

/** Fields the caller supplies. The store owns all bookkeeping fields. */
export type MemoryInput = Omit<
  MemoryRecord,
  'id' | 'createdAt' | 'lastAccessedAt' | 'accessCount'
> & { id?: string };

export interface RetrievalQuery {
  /** Query vector. Omit to rank by importance and recency alone. */
  embedding?: number[];
  kinds?: MemoryKind[];
  tags?: string[];
  limit?: number;
  /** Drop candidates scoring below this threshold. */
  minScore?: number;
  /** 0 = pure relevance, 1 = maximum diversity. See rankWithDiversity. */
  diversity?: number;
}

export interface ScoredMemory {
  record: MemoryRecord;
  score: number;
  breakdown: {
    relevance: number;
    retention: number;
    importance: number;
  };
}

export interface MemoryStore {
  put(input: MemoryInput): Promise<MemoryRecord>;
  get(id: string): Promise<MemoryRecord | undefined>;
  delete(id: string): Promise<boolean>;
  all(): Promise<MemoryRecord[]>;
  /** Marks a read: bumps accessCount and lastAccessedAt. Drives importance. */
  touch(id: string, at?: number): Promise<void>;
  size(): Promise<number>;
}

export class MemoryValidationError extends Error {
  constructor(
    message: string,
    readonly issues: z.ZodIssue[],
  ) {
    super(message);
    this.name = 'MemoryValidationError';
  }
}
