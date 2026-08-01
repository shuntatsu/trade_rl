## Release-owned architecture boundary result

- Ruff/format status: 0
- Focused pytest status: 0
- Import Linter status: 0
- MyPy status: 0

<details><summary>ruff.log tail</summary>

```text
Found 30 errors (30 fixed, 0 remaining).
3 files reformatted, 836 files left unchanged
All checks passed!
839 files already formatted
```
</details>

<details><summary>focused.log tail</summary>

```text
........................................................................ [ 31%]
........................................................................ [ 62%]
........................................................................ [ 94%]
.............                                                            [100%]
229 passed in 11.19s
```
</details>

<details><summary>imports.log tail</summary>

```text

╔══╗─────────▶╔╗ ╔╗      ╔╗◀───┐
╚╣╠╝◀─────┐  ╔╝╚╗║║────▶╔╝╚╗   │
 ║║   ╔══╦══╦╩╗╔╝║║  ╔╦═╩╗╔╝╔═╦══╗
 ║║╔══╣╔╗║╔╗║╔╣║ ║║ ╔╬╣╔╗║║ ║│║╔═╝
╔╣╠╣║║║╚╝║╚╝║║║╚╗║╚═╝║║║║║╚╗║═╣║
╚══╩╩╩╣╔═╩══╩╝╚═╝╚═══╩╩╝╚╩═╩╩═╩╝
  └──▶║║                    ▲ 
      ╚╝────────────────────┘


---------
Contracts
---------

Analyzed 387 files, 2951 dependencies.
--------------------------------------

Trade RL responsibility layers KEPT
Domain remains standard-library only KEPT
Telemetry remains diagnostic and standard-library only KEPT
Serving cannot import training or workflows KEPT
Studio cannot import workflow implementations KEPT
Serving cannot import the RL structured export implementation KEPT
Release remains verification-only and below serving KEPT
Learning contracts cannot depend on SB3 frameworks KEPT
Workflows cannot directly import model frameworks KEPT
Training core remains framework independent KEPT
Runtime and training paths cannot import offline signers KEPT
Catalog contracts remain framework and adapter independent KEPT

Contracts: 12 kept, 0 broken.
```
</details>

<details><summary>mypy.log tail</summary>

```text
Success: no issues found in 330 source files
```
</details>
