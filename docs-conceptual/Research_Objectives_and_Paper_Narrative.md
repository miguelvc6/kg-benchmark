# Research Objectives and Paper Narrative

WikidataRepairEval studies whether language models can reconstruct historically accepted knowledge-graph repairs when
the information available to them is controlled. It separates where a repair belongs—an A-box claim or a T-box schema
rule—from which visible information condition supports the historical repair.

The paper asks how repair performance changes across logical evidence (IC-L), local graph evidence (IC-G), and cases
where supported logical/local extractors do not identify the target (IC-E-elim); whether context and static few-shot
examples improve repair and repair-locus diagnosis; and whether local models reproduce these effects relative to a
nested external-model calibration sample.

The benchmark is a reasoning floor rather than a retrieval system. `logic_only` exposes sanitized rule context and
`local_graph` adds sanitized local graph context. IC-E-elim receives no external retrieval. Success in that stratum may
reflect latent model knowledge, unsupported guessing, missed extraction, or leakage; it does not prove that external
evidence was available or necessary.

The confirmatory dataset is acquired only after methodology freeze, exhaustively audited, grouped by independent repair
events, and selected deterministically. The 1,200-case population is evaluated by all local models; a nested 600-case
view is evaluated by Azure as a sequential, no-tools calibration condition because batch deployment is unavailable for
the selected external model. Execution mode does not change Azure's calibration-only analytical role. Larger nested
populations are permitted as registered extensions and never overwrite the confirmatory population.
