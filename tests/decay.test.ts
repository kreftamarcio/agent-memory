import { describe, it, expect } from 'vitest';
import { DecayEngine } from '../src/scoring/decay';
import type { DecayProfile, MemoryTrace } from '../src/scoring/decay';

const MS_PER_DAY = 86_400_000;

const profile: DecayProfile = {
  episodic: { halfLifeDays: 3 },
  semantic: { halfLifeDays: 90 },
  procedural: { halfLifeDays: 365, strengthenOnUse: true },
  reinforcementAlpha: 0.3,
};

const NOW = new Date('2026-08-16T12:00:00Z');

function trace(overrides: Partial<MemoryTrace> = {}): MemoryTrace {
  return {
    id: 'trace-1',
    store: 'episodic',
    strength: 1,
    importance: 0,
    createdAt: NOW,
    lastAccessedAt: NOW,
    accessCount: 0,
    ...overrides,
  };
}

function daysAgo(days: number): Date {
  return new Date(NOW.getTime() - days * MS_PER_DAY);
}

describe('DecayEngine configuration', () => {
  it('rejects a non-positive half-life', () => {
    expect(
      () => new DecayEngine({ ...profile, episodic: { halfLifeDays: 0 } }),
    ).toThrow(/halfLifeDays must be a positive finite number/);
  });

  it('rejects a reinforcement alpha that would saturate in one access', () => {
    expect(() => new DecayEngine({ ...profile, reinforcementAlpha: 1 })).toThrow(
      /strictly between 0 and 1/,
    );
  });

  it('rejects facts that would decay faster than the events behind them', () => {
    expect(
      () =>
        new DecayEngine({
          ...profile,
          episodic: { halfLifeDays: 30 },
          semantic: { halfLifeDays: 7 },
        }),
    ).toThrow(/shorter than/);
  });

  it('accepts a valid profile', () => {
    expect(() => new DecayEngine(profile)).not.toThrow();
  });
});

describe('currentStrength', () => {
  const engine = new DecayEngine(profile);

  it('halves strength after exactly one half-life', () => {
    const t = trace({ lastAccessedAt: daysAgo(3) });
    expect(engine.currentStrength(t, NOW)).toBeCloseTo(0.5, 6);
  });

  it('quarters strength after two half-lives', () => {
    const t = trace({ lastAccessedAt: daysAgo(6) });
    expect(engine.currentStrength(t, NOW)).toBeCloseTo(0.25, 6);
  });

  it('leaves an untouched trace at full strength', () => {
    expect(engine.currentStrength(trace(), NOW)).toBe(1);
  });

  it('never returns a value outside [0,1]', () => {
    const t = trace({ strength: 5, lastAccessedAt: daysAgo(-10) });
    const value = engine.currentStrength(t, NOW);
    expect(value).toBeLessThanOrEqual(1);
    expect(value).toBeGreaterThanOrEqual(0);
  });

  it('holds an important trace at its floor instead of letting it vanish', () => {
    // importance 0.9 -> floor 0.45. After a year an episodic trace would
    // otherwise be effectively zero.
    const t = trace({ importance: 0.9, lastAccessedAt: daysAgo(365) });
    expect(engine.currentStrength(t, NOW)).toBeCloseTo(0.45, 6);
  });

  it('decays semantic traces far slower than episodic ones', () => {
    const episodic = engine.currentStrength(
      trace({ store: 'episodic', lastAccessedAt: daysAgo(30) }),
      NOW,
    );
    const semantic = engine.currentStrength(
      trace({ store: 'semantic', lastAccessedAt: daysAgo(30) }),
      NOW,
    );
    expect(semantic).toBeGreaterThan(episodic);
  });
});

describe('reinforce', () => {
  const engine = new DecayEngine(profile);

  it('applies alpha to the remaining headroom', () => {
    // current 0.5, alpha 0.3 -> 0.5 + 0.3 * 0.5 = 0.65
    const t = trace({ lastAccessedAt: daysAgo(3) });
    expect(engine.reinforce(t, NOW).strength).toBeCloseTo(0.65, 6);
  });

  it('saturates instead of pinning a trace at 1', () => {
    let t = trace({ strength: 0.5, lastAccessedAt: NOW });
    for (let i = 0; i < 50; i += 1) {
      t = engine.reinforce(t, NOW);
    }
    expect(t.strength).toBeLessThan(1);
    expect(t.strength).toBeGreaterThan(0.99);
  });

  it('yields diminishing returns per access', () => {
    const base = trace({ strength: 0.2, lastAccessedAt: NOW });
    const first = engine.reinforce(base, NOW);
    const second = engine.reinforce(first, NOW);

    const firstGain = first.strength - base.strength;
    const secondGain = second.strength - first.strength;
    expect(secondGain).toBeLessThan(firstGain);
  });

  it('strengthens procedural traces harder than episodic ones', () => {
    const episodic = engine.reinforce(
      trace({ store: 'episodic', strength: 0.4, lastAccessedAt: NOW }),
      NOW,
    );
    const procedural = engine.reinforce(
      trace({ store: 'procedural', strength: 0.4, lastAccessedAt: NOW }),
      NOW,
    );
    expect(procedural.strength).toBeGreaterThan(episodic.strength);
  });

  it('records the access without mutating the input', () => {
    const original = trace({ accessCount: 4 });
    const result = engine.reinforce(original, NOW);

    expect(result.accessCount).toBe(5);
    expect(result.lastAccessedAt).toBe(NOW);
    expect(original.accessCount).toBe(4);
  });
});

describe('isForgotten', () => {
  const engine = new DecayEngine(profile);

  it('flags a trace that fell below the threshold', () => {
    const t = trace({ lastAccessedAt: daysAgo(30) });
    expect(engine.isForgotten(t, 0.1, NOW)).toBe(true);
  });

  it('keeps a recently reinforced trace', () => {
    expect(engine.isForgotten(trace(), 0.1, NOW)).toBe(false);
  });
});

describe('daysUntilStrength', () => {
  const engine = new DecayEngine(profile);

  it('returns the half-life when the target is half the current strength', () => {
    expect(engine.daysUntilStrength(trace(), 0.5)).toBeCloseTo(3, 6);
  });

  it('returns null when the importance floor already protects the target', () => {
    const t = trace({ importance: 0.9 }); // floor 0.45 >= 0.4
    expect(engine.daysUntilStrength(t, 0.4)).toBeNull();
  });

  it('returns 0 for a trace already at or below the target', () => {
    expect(engine.daysUntilStrength(trace({ strength: 0.3 }), 0.5)).toBe(0);
  });

  it('rejects targets outside the open interval (0,1)', () => {
    expect(() => engine.daysUntilStrength(trace(), 0)).toThrow(/strictly between/);
    expect(() => engine.daysUntilStrength(trace(), 1)).toThrow(/strictly between/);
  });
});

describe('recencyScore', () => {
  const engine = new DecayEngine(profile);

  it('halves at the recency half-life regardless of reinforcement', () => {
    const t = trace({ createdAt: daysAgo(14), lastAccessedAt: NOW, accessCount: 20 });
    expect(engine.recencyScore(t, 14, NOW)).toBeCloseTo(0.5, 6);
  });

  it('scores a brand new trace at 1', () => {
    expect(engine.recencyScore(trace(), 14, NOW)).toBe(1);
  });
});

describe('evaluateBatch', () => {
  const engine = new DecayEngine(profile);

  it('annotates every trace with its effective strength', () => {
    const traces = [
      trace({ id: 'a', lastAccessedAt: NOW }),
      trace({ id: 'b', lastAccessedAt: daysAgo(3) }),
    ];

    const evaluated = engine.evaluateBatch(traces, NOW);

    expect(evaluated).toHaveLength(2);
    expect(evaluated[0].effectiveStrength).toBe(1);
    expect(evaluated[1].effectiveStrength).toBeCloseTo(0.5, 6);
  });
});
