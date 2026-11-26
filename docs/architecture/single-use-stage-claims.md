# Prospective single-use stage reservations

The additive trade_rl.artifacts.stage_claim API is a POSIX-only local reservation primitive for new protocol families. It is not an authorization gate and must never be applied to a frozen v1 ledger.

Provision one existing real absolute directory under a trusted operator-owned location. Persist its observed device/inode and path as part of the separately reviewed prospective policy. Relative paths, Windows literals on POSIX, parent traversal, ambiguous anchors and symlinked directories are refused. Use the same independent root pin across every stage and protocol digest in the family. A different root or machine is not coordinated.

After separate review and authorization, reserve_stage_claim verifies the held directory identity, exclusively creates a comment marker scoped by repository/comment ID and a stage marker scoped by canonical stage name, and fsyncs each file and the directory. A duplicate cannot overwrite either marker. Comment IDs cannot be reused across stages or digest namespaces in that root. Partial failures may consume a marker; they never permit automatic reset or retry. Verify the root again before returning. This requires a trusted stable POSIX filesystem, not arbitrary concurrent directory replacement, network filesystem or Windows equivalence. No actual economic work follows from a successful reservation.

The frozen CLEAN_ALL v1 source, specification, protocol digest and consumed INTERIM_A evidence are not changed. A prospective protocol must independently define its attempt lifecycle, adopt and test this primitive, and receive separate result-blind review and stage authorization before execution.
