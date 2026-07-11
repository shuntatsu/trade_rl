# Frontend

The previous dashboard expected the removed mixed-responsibility metrics/model-management server. It is not part of the Production Serving Plane.

The supported online service exposes only `/health`, `/ready`, and authenticated `/api/signal/latest`. See the repository [`README.md`](../README.md) and [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).

Any future operator UI must use a separately authenticated Control Plane service or offline artifacts. It must not add training, model deletion, promotion, or rollback routes to the Serving Plane.
