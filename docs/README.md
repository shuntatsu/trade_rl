# Documentation status and precedence

This directory contains both current operational specifications and historical research notes.

## Authoritative operational documents

The following documents are normative for deployment and incident response:

1. `DOCUMENTATION_AUDIT_CHECKLIST.md`
2. `deployment_evidence.md`
3. `runbook_incident_response.md`
4. `runbook_model_rollback.md`
5. `runbook_compliance.md`
6. `model_decision_log.md`

## Historical research documents

`ARCHITECTURE.md` and `PROFIT_DESIGN.md` contain measured results, design hypotheses and earlier defaults accumulated during research. They are not proof of live profitability and must not override the Phase 1 frozen configuration, deployment evidence gate, or current runbooks.

Statements such as expected return, Sharpe, persistent alpha, or a feature being a structural edge are hypotheses until confirmed on untouched out-of-sample data and subsequent Shadow/Canary evidence.

## Production status

Code implementation and local tests do not authorize real-money deployment. Production remains NO-GO until every Production blocker in `DOCUMENTATION_AUDIT_CHECKLIST.md` is closed with recorded evidence and an approved `PROD-<digits>` ticket.
