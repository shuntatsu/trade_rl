# Execution Evidence and Stage Transaction Hardening Design

## Context

PR #305 bound execution evidence into selection authorization, introduced verified private copies for unsafe deserializers, and added cursor compare-and-swap locking. A post-merge audit found five residual gaps:

1. an execution event artifact can be syntactically canonical without representing a valid order history;
2. the artifact is not bound to action, observation, equity, candidate, evaluation-run, fold, and seed identities;
3. the maintained production path accepts an externally prepared event artifact instead of emitting it from the completed evaluation result;
4. symbol-triplet completion and cursor files are published separately and can diverge after crashes or post-replace durability errors;
5. structured manifests and execution event artifacts still have verify/reopen or non-exclusive write boundaries.

## Chosen approach

Use one content-addressed execution replay root and one generation-based stage state.

### Execution evidence root

`ExecutionEventArtifact` becomes a strict replay artifact rather than a loose JSON envelope. It contains:

- exact candidate configuration digest;
- exact evaluation run digest;
- fold index and seed;
- canonical action, observation, equity, and order-event trace digests;
- strict order events reconstructed through domain constructors;
- terminal book and terminal order-book payloads;
- a `StatefulReplayEvidence` payload whose digest and counts must agree with all embedded traces.

Each event is reconstructed as an `OrderEvent`, sequences must be contiguous from zero, order transitions must follow the maintained order-state machine, cumulative fill identities must hold, and the terminal order book must match the final event state. Promotion evidence is built only from this verified artifact. `complete_order_evidence` remains serialized for compatibility but is derived and must be true; callers may not claim it.

The maintained evaluation workflow writes the replay artifact and promotion evidence together from the completed execution result. Selected-final training consumes the resulting content-addressed artifact and refuses ad-hoc unbound event files.

### Stage state generations

A stage commit writes one immutable generation directory:

```text
<state-root>/generations/<generation-digest>/
  completion.json
  cursor.json
```

After both files are written, fsynced, reloaded, and cross-validated, one `current.json` pointer is atomically replaced. Readers resolve only through the pointer. A crash before pointer replacement leaves an unreachable generation that can be garbage-collected. A post-replace durability error keeps the generation and pointer in place and reports an uncertain-durability error without deleting completion evidence.

Legacy separate cursor/completion paths remain readable for migration, but all maintained writes use the generation store. The migration creates generation zero from the validated legacy state before the next advancement.

### Verified manifest loading and immutable writes

Structured export manifests are loaded from the exact verified byte sequence. A new bytes parser is the canonical parser, while path-based loading opens once with the regular-file boundary and delegates to it.

Execution replay artifacts use exclusive creation, file fsync, and parent-directory fsync. Their filename is derived from the artifact SHA-256 digest when written through the maintained production workflow.

## Alternatives considered

### PostgreSQL-only stage transaction

A database transaction would provide strong atomicity but would make local and filesystem-only workflows depend on PostgreSQL. The generation pointer keeps the current portability and remains crash-recoverable.

### Repairing the two-file commit in place

Adding more cleanup branches cannot make two independent path replacements atomic. It also cannot distinguish process death from ordinary exceptions. A generation pointer provides one commit point and a simpler recovery model.

### Keeping loose event dictionaries with more required keys

A larger required-key set still duplicates the order-domain contract and will drift. Reconstructing domain objects makes the existing invariants the source of truth.

## Error handling

- malformed, truncated, non-canonical, symlinked, non-regular, or non-exclusive replay artifacts fail closed;
- event sequence gaps, invalid transitions, identity mismatches, trace-count mismatches, and terminal-state mismatches fail before promotion evidence is created;
- stale stage writers fail on pointer digest compare-and-swap;
- unreachable generation directories are never treated as committed state;
- pointer replacement durability uncertainty is reported explicitly and is not rolled back destructively;
- legacy state migration fails if cursor and completion do not cross-reference the same digest.

## Testing strategy

Tests must cover:

- hand-written minimal event dictionaries;
- missing, duplicated, reordered, and discontinuous event sequences;
- impossible order transitions and inconsistent fill arithmetic;
- terminal book and order-book mismatch;
- candidate, run, fold, seed, and trace substitution;
- production evaluation artifact emission and selected-final consumption;
- exclusive writer races and process-like interruption before pointer replacement;
- post-pointer-replace directory-fsync failure;
- legacy migration and stale-writer CAS;
- structured manifest swap between verification and parsing;
- symlink, FIFO, partial write, and duplicate artifact publication.

The final merge gate is the repository's complete CI matrix on one unchanged head commit.
