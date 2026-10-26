# Evaluating open source components

A component is evaluated before it is adopted and re-evaluated on a schedule. Each
evaluation is a public issue opened with the **OSS evaluation** form and labelled
`oss-evaluation`, so the record and its approval can be found later.

## Steps

The form's fields follow these steps.

1. **Gate.** *Adopt third-party OSS* for a dependency, tool, action, external program,
   dataset or model; *Publish our OSS* for code, data or a model leaving the private
   repository.
2. **Target.** The exact version, version range or commit, and the official URL.
3. **Use.** How the component is used, combined, distributed or published: a runtime
   dependency, an optional extra, a build or CI tool, an external program the user
   installs, something shipped in a container image.
4. **Blocking conditions.** If one fails, the evaluation stops there:
   - the official source and the exact version are identified;
   - the license is identified and its class in the [license policy](license-policy.md)
     allows this use;
   - no unresolved critical vulnerability affects the accepted versions;
   - there is no code, binary or data of unknown origin.
5. **Facts.** License, release and commit activity, known advisories in the accepted
   versions, with the date they were checked.
6. **Score**, out of 100:

   | Area | Points |
   |---|---:|
   | License and rights | 20 |
   | Security and supply chain | 20 |
   | Function and performance | 15 |
   | Maintenance activity (recent releases and commits, not stars) | 15 |
   | Architecture and operation | 10 |
   | Community and governance | 10 |
   | Sustainability and replaceability | 10 |

   The score makes evaluations comparable. The decision rests on the blocking
   conditions and on the conditions attached to it.
7. **Alternatives.** At least one alternative, or why there is none.
8. **Decision.** *Approve*, *Approve with conditions*, *Preview / PoC only*, or *Reject
   or postpone*, with each condition's owner and deadline and what triggers a
   re-evaluation.

The project lead approves the decision with a comment in the issue. The component is
then listed in the [inventory](inventory.md) with a link to the issue. A condition
that needs a code change gets its own issue and pull request, which link back to the
evaluation.

## Re-evaluation

Every quarter (January, April, July and October) a maintainer re-checks each
component in the inventory: the accepted versions and the latest release, the license,
known advisories and maintenance activity. Each result is an issue labelled
`oss-evaluation` and `re-evaluation`, approved like a first evaluation.

A component is also re-evaluated, outside the schedule, when:

- its license changes;
- a high or critical advisory affects the accepted versions;
- the project is archived or stops releasing;
- we move to a new major version;
- the way we use it changes, for example when it starts to ship in a container image.

The first components were evaluated in October 2026. NumPy, the only runtime
dependency before then, was re-evaluated in the same quarter
([#52](https://github.com/Official-Space-AI/OpenPhysGraspGen/issues/52)); the next
quarterly re-evaluation of every component is in January 2027.
