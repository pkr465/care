"""
LLM prompt templates for Verilog/SystemVerilog HDL codebase analysis
"""

from typing import Dict, Any, List


class PromptTemplates:
    """
    Contains all LLM prompt templates for Verilog/SystemVerilog HDL codebase analysis.

    Templates are designed to provide context-aware prompts for different
    analysis stages, optimized for HDL/RTL specific concerns and professional
    reporting.

    NOTE:
    - Many prompts now require the model to return a final, single JSON object
      with a clearly defined schema. This is intended to be ingestible by both
      humans (as structured narrative) and coding agents (for automation).
    """

    @staticmethod
    def get_codebase_insights_prompt(file_cache: List[Dict[str, Any]]) -> str:
        """Generate prompt for overall codebase insights."""
        total_files = len(file_cache)
        languages: Dict[str, int] = {}
        total_lines = 0
        total_functions = 0

        for file_entry in file_cache:
            lang = file_entry.get("language", "unknown")
            languages[lang] = languages.get(lang, 0) + 1

            metrics = file_entry.get("metrics", {})
            total_lines += metrics.get("total_lines", 0)

            functions = file_entry.get("functions", [])
            total_functions += len(functions)

        sample_files = [f.get("file_relative_path", "") for f in file_cache[:10]]

        return f"""
You are a senior Verilog/SystemVerilog hardware design architect performing a high-level assessment of an RTL codebase.

Context:
- Total files: {total_files}
- Languages: {dict(languages)}
- Total lines of code: {total_lines}
- Total modules: {total_functions}

Sample file structure:
{chr(10).join(f"- {path}" for path in sample_files)}

Objectives:
- Provide a professional, concise, and technically accurate assessment.
- Your analysis should be directly useful to both:
  - Human architects and technical leaders.
  - Automated coding agents, which will consume the structured JSON output.

Please perform the following analysis:

1) Architecture Assessment
   - Infer the overall RTL hierarchy, clock domain organization, and prevalent design patterns based on file structure.
   - Identify apparent layering (e.g., interconnect, core logic, interfaces, testbenches, verification) and violations.
   - Assess reset structure, parameterization strategy, and reuse patterns.

2) Development Practices
   - Infer HDL development and coding practices from the structure (lint discipline, CDC practices, simulation discipline, documentation).
   - Highlight strengths and weaknesses relevant to professional Verilog/SystemVerilog development.

3) Potential Concerns
   - Identify key areas warranting deeper investigation (e.g., monolithic RTL blocks, tight coupling between clock domains, missing CDC synchronizers, unbounded generates).
   - Emphasize HDL-specific concerns such as synthesis complexity, port signature compatibility, clock domain management, and timing constraints.

4) Recommendations
   - Provide prioritized, concrete recommendations that can inform a medium-to-long-term improvement plan.
   - Recommendations should be specific enough to be turned into work items.

5) Verification Strategy
   - Propose a high-level verification and validation strategy aligned to the inferred risks:
     - Unit-level simulation, integration simulation, system-level verification.
     - Formal verification, lint analysis, CDC analysis.
   - Suggest how the organization can measure verification progress.

Important response format requirements:
- Provide a short, professional narrative if necessary.
- Then, at the end of the response, output a single JSON object and nothing after it.
- The JSON must strictly follow this schema:

{{
  "architecture_assessment": {{
    "summary": "string",
    "observations": ["string"]
  }},
  "issues": [
    {{
      "id": "ARCH-ISSUE-1",
      "title": "Short title",
      "description": "Detailed HDL/RTL specific issue description",
      "context": "Where/when this typically occurs in such designs",
      "severity": "low|medium|high|critical",
      "impact": "Design productivity / timing closure / signal integrity / synthesis times / verification coverage / etc."
    }}
  ],
  "proposed_solutions": [
    {{
      "issue_id": "ARCH-ISSUE-1",
      "summary": "Short summary of the solution",
      "steps": [
        "Actionable step 1",
        "Actionable step 2"
      ],
      "implementation_hints": [
        "Concrete hints a coding agent can follow (e.g., introduce interface X, split module Y, add unit tests for module Z)"
      ]
    }}
  ],
  "risks_and_concerns": [
    {{
      "category": "architecture|performance|maintainability|testing|security|build",
      "description": "Risk description",
      "likelihood": "low|medium|high",
      "impact": "low|medium|high|critical",
      "mitigations": ["string"]
    }}
  ],
  "recommendations": [
    {{
      "id": "ARCH-REC-1",
      "title": "Short recommendation name",
      "priority": "P0|P1|P2",
      "description": "What to do and why",
      "related_issues": ["ARCH-ISSUE-1"]
    }}
  ],
  "verification_plan": {{
    "overview": "High-level description of verification strategy",
    "phases": [
      {{
        "name": "Phase name (e.g., Establish baseline unit simulation)",
        "goals": ["string"],
        "verification_types": ["unit_sim", "integration_sim", "system_sim", "formal_verification", "lint_analysis", "cdc_analysis"],
        "actions": [
          "Actionable verification step 1",
          "Actionable verification step 2"
        ]
      }}
    ]
  }}
}}
""".strip()

    @staticmethod
    def get_dependency_analysis_prompt(dependency_graph: Dict[str, Any]) -> str:
        """Generate prompt for module hierarchy and dependency analysis."""
        analysis = dependency_graph.get("analysis", {})

        total_nodes = analysis.get("total_nodes", 0)
        internal_nodes = analysis.get("internal_nodes", 0)
        external_nodes = analysis.get("external_nodes", 0)
        has_cycles = analysis.get("has_cycles", False)
        cycle_count = analysis.get("cycle_count", 0)
        max_fan_out = analysis.get("max_fan_out", 0)

        sample_modules: List[str] = []
        count = 0
        for module_name, module_data in dependency_graph.items():
            if module_name != "analysis" and count < 5:
                deps = len(module_data.get("dependencies", []))
                sample_modules.append(f"- {module_name}: {deps} dependencies")
                count += 1

        return f"""
You are a senior Verilog/SystemVerilog hardware design architect analyzing module hierarchy and dependency relationships in an RTL codebase.

Dependency metrics summary:
- Total modules: {total_nodes}
- Internal modules: {internal_nodes}
- External IP blocks: {external_nodes}
- Circular instantiation: {"Yes" if has_cycles else "No"} ({cycle_count} cycles)
- Maximum fan-out: {max_fan_out}

Sample modules:
{chr(10).join(sample_modules)}

Objectives:
- Provide a highly professional, in-depth analysis of the module hierarchy and dependency structure.
- Produce output suitable for:
  - Architects and technical leads, and
  - Automated coding agents that will use the structured output to drive refactorings.

Please provide analysis for:

1) Dependency Health
   - Assess overall module hierarchy health for this Verilog/SystemVerilog design.
   - Comment on potential "god modules" (monolithic RTL blocks), heavily shared include/package files, and deep include chains.

2) Coupling Analysis
   - Interpret fan-out and fan-in to identify tight coupling and high-risk hotspots.
   - Highlight problematic coupling patterns (e.g., cross-domain dependencies, circular module instantiation, cyclic includes).

3) Circular Dependencies
   - {"Analyze the impact of circular instantiation on elaboration, timing analysis, and verification. Propose strategies to systematically eliminate these cycles." if has_cycles else "Explain the benefits of having no circular instantiation, and provide guidance on how to preserve this property as the design evolves."}

4) Architecture Implications
   - Describe what the module hierarchy suggests about layering, boundaries, and modular design.
   - Identify clear violations of standard architectural principles (e.g., low-level logic depending on high-level controls).

5) Refactoring Opportunities
   - Propose specific, high-value refactoring opportunities:
     - e.g., "extract interface/modport", "introduce adapter module", "split monolithic RTL block", "move shared declarations into dedicated packages".

6) Alignment with SystemVerilog Best Practices
   - Evaluate how the current structure aligns with Reuse Methodology Manual (RMM) and SystemVerilog best practices such as:
     - package (.svh) vs. module (.sv) separation,
     - use of parameterized modules,
     - interface/modport patterns,
     - stable port lists.

7) Synthesis & Elaboration Impact
   - Explain likely implications for synthesis times, incremental elaboration, and CI stability.
   - Suggest concrete synthesis-oriented improvements (e.g., hierarchical synthesis partitions, file list organization, limiting widely included packages).

8) Verification Plan for Dependency Refactors
   - Outline a verification/validation approach for dependency-related refactoring:
     - elaboration checks,
     - unit/integration/regression simulations,
     - port signature compatibility checks.

Important response format requirements:
- You may provide a short narrative, but the primary output must be a single JSON object at the end.
- The JSON must strictly follow this schema, and nothing should follow it:

{{
  "architecture_assessment": {{
    "summary": "string",
    "observations": ["string"]
  }},
  "issues": [
    {{
      "id": "DEP-ISSUE-1",
      "title": "Short title",
      "description": "Detailed description of the dependency issue",
      "context": "Where/when this occurs (e.g., cross-layer dependency between core and UI)",
      "severity": "low|medium|high|critical",
      "impact": "build_times|maintainability|testability|runtime_risk|other"
    }}
  ],
  "proposed_solutions": [
    {{
      "issue_id": "DEP-ISSUE-1",
      "summary": "Short solution summary",
      "steps": [
        "Concrete step 1 (e.g., split package A into A_iface.svh and A_impl.svh)",
        "Concrete step 2"
      ],
      "implementation_hints": [
        "Specific hints for a coding agent (e.g., introduce interface/modport, move implementation-only includes to module .sv)"
      ]
    }}
  ],
  "risks_and_concerns": [
    {{
      "category": "architecture|build|regression|complexity",
      "description": "Risk description",
      "likelihood": "low|medium|high",
      "impact": "low|medium|high|critical",
      "mitigations": ["string"]
    }}
  ],
  "recommendations": [
    {{
      "id": "DEP-REC-1",
      "title": "Short recommendation title",
      "priority": "P0|P1|P2",
      "description": "What to do and why",
      "related_issues": ["DEP-ISSUE-1"]
    }}
  ],
  "verification_plan": {{
    "overview": "How to validate dependency refactors",
    "phases": [
      {{
        "name": "Phase 1 - Safe refactors & elaboration verification",
        "goals": ["string"],
        "verification_types": ["elaboration", "unit_sim", "integration_sim"],
        "actions": [
          "Run full elaboration on all supported synthesis tools",
          "Execute unit and integration simulations for impacted modules",
          "Add verification tests specifically around refactored interfaces"
        ]
      }}
    ]
  }}
}}
""".strip()

    @staticmethod
    def get_health_metrics_analysis_prompt(health_metrics: Dict[str, Any]) -> str:
        """Generate prompt for health metrics analysis (per-metric, structured)."""
        scores: Dict[str, Dict[str, Any]] = {}
        for metric_name, metric_data in health_metrics.items():
            if isinstance(metric_data, dict) and "score" in metric_data:
                scores[metric_name] = {
                    "score": metric_data.get("score", 0),
                    "grade": metric_data.get("grade", "F"),
                    "issues": len(metric_data.get("issues", [])),
                }

        overall_health = health_metrics.get("overall_health", {})
        overall_score = overall_health.get("score", 0)
        overall_grade = overall_health.get("grade", "F")

        scores_lines = [
            f"- {name}: {data['score']}/100 ({data['grade']}) - {data['issues']} issues"
            for name, data in scores.items()
        ]

        return f"""
You are an HDL code quality expert analyzing detailed design health metrics for a mature RTL codebase.

Overall health:
- Score: {overall_score}/100
- Grade: {overall_grade}

Individual metric scores:
{chr(10).join(scores_lines)}

Available metrics (if present in the input):
- dependency_score
- quality_score
- complexity_score
- maintainability_score
- documentation_score
- test_coverage_score
- security_score
- overall_health

Objectives:
- Provide a highly professional, per-metric analysis.
- For each metric present, your analysis must include:
  - Issue description.
  - Proposed solution (implementation-ready, stepwise).
  - Architectural / issue assessment.
  - Risks and concerns.
  - Recommendations (prioritized).
  - Actionable testing plan.

Guidance:
- Use HDL/RTL-specific terminology and best practices.
- Focus on changes that realistically improve the design over time.
- Distinguish between quick wins and structural efforts.

Important response format requirements:
- You may include a brief narrative, but the primary output must be a single JSON object at the end.
- The JSON must follow this schema (include only metrics that exist in health_metrics):

{{
  "metrics": {{
    "<metric_name>": {{
      "issues": [
        {{
          "id": "METRIC-ISSUE-1",
          "description": "string",
          "severity": "low|medium|high|critical",
          "impact": "string"
        }}
      ],
      "proposed_solutions": [
        {{
          "issue_id": "METRIC-ISSUE-1",
          "steps": ["string"],
          "implementation_hints": ["string"]
        }}
      ],
      "architecture_assessment": {{
        "summary": "string",
        "observations": ["string"]
      }},
      "risks_and_concerns": [
        {{
          "description": "string",
          "likelihood": "low|medium|high",
          "impact": "low|medium|high|critical"
        }}
      ],
      "recommendations": [
        {{
          "id": "METRIC-REC-1",
          "priority": "P0|P1|P2",
          "description": "string",
          "related_issues": ["METRIC-ISSUE-1"]
        }}
      ],
      "testing_plan": {{
        "overview": "string",
        "actions": ["string"]
      }}
    }}
  }},
  "overall": {{
    "priority_roadmap": [
      {{
        "step": 1,
        "focus_area": "security|complexity|maintainability|dependencies|tests|documentation",
        "description": "string"
      }}
    ]
  }}
}}
Notes:
- Replace <metric_name> with actual keys (e.g., "dependency_score", "security_score") that are present in health_metrics.
- Do not include metrics that are not present in the input.
""".strip()

    @staticmethod
    def get_documentation_recommendations_prompt(
        documentation_analysis: Dict[str, Any]
    ) -> str:
        """Generate prompt for documentation recommendations."""
        base_docs = documentation_analysis.get("base_documentation", {})
        overview = base_docs.get("overview", {})

        total_modules = overview.get("total_modules", 0)
        internal_modules = overview.get("internal_modules", 0)
        external_deps = overview.get("external_dependencies", 0)

        return f"""
You are a technical documentation specialist for professional Verilog/SystemVerilog RTL projects.

Project overview:
- Total modules: {total_modules}
- Internal modules: {internal_modules}
- External IP blocks: {external_deps}

Objectives:
- Provide a professional, comprehensive assessment of documentation gaps and opportunities.
- Produce recommendations that can be executed by both humans and coding agents.
- Include issue descriptions, proposals, risks, and a documentation-focused verification/validation plan.

Please cover:

1) Module Interface Documentation
   - Standards (e.g., natural-docs or structured comments) and conventions suitable for this design.
   - Minimum expectations for public and internal module interfaces.

2) Include File & Package Documentation
   - Documentation of clock domain, reset strategy, timing constraints, and synthesis attributes.
   - Guidance for documenting macros, generate constructs, and parameterized modules.

3) EDA Tooling & Flow Documentation
   - Documentation of synthesis flows, simulation tools, lint tools, and verification methodology.
   - Configuration, constraint files, and design variations across technology libraries.

4) Architecture Documentation
   - Recommended architectural views (block diagrams, dataflow, clock domain crossing model, hierarchy).
   - How to keep such documentation aligned with the RTL code.

5) Designer Onboarding
   - Documentation that accelerates onboarding for new RTL designers.

6) Maintenance & Operational Documentation
   - Documentation that supports long-term maintenance, debugging, failure analysis, and releases.

7) Integration & External IP
   - Documentation of IP cores, EDA tool versions, technology library compatibility, and interface expectations.

8) Examples & Reference Designs
   - Types of code samples and design patterns that should be provided and kept verified through simulation.

Important response format requirements:
- Optionally provide a brief narrative.
- Then output a single JSON object and nothing after it, matching this schema:

{{
  "issues": [
    {{
      "id": "DOC-ISSUE-1",
      "area": "api|headers|build|architecture|onboarding|maintenance|integration|examples",
      "description": "Documentation gap or weakness",
      "severity": "low|medium|high|critical",
      "impact": "onboarding|maintainability|operability|risk"
    }}
  ],
  "proposed_solutions": [
    {{
      "issue_id": "DOC-ISSUE-1",
      "summary": "Short solution summary",
      "steps": [
        "Concrete step 1 (e.g., add structured comments for all public ports in module X)",
        "Concrete step 2"
      ],
      "implementation_hints": [
        "Hints for a coding agent (e.g., for each module in rtl/, ensure comments describing ports, clock domains, and reset strategy are present)"
      ]
    }}
  ],
  "architecture_assessment": {{
    "summary": "How current documentation reflects or obscures the architecture",
    "observations": ["string"]
  }},
  "risks_and_concerns": [
    {{
      "description": "Risk from poor or missing documentation",
      "likelihood": "low|medium|high",
      "impact": "low|medium|high|critical",
      "mitigations": ["string"]
    }}
  ],
  "recommendations": [
    {{
      "id": "DOC-REC-1",
      "priority": "P0|P1|P2",
      "description": "Specific documentation initiative",
      "related_issues": ["DOC-ISSUE-1"]
    }}
  ],
  "verification_plan": {{
    "overview": "How to validate documentation improvements",
    "actions": [
      "Introduce documentation coverage checks in CI (e.g., percentage of public module ports documented)",
      "Regularly verify that design examples elaborate and simulate correctly as part of regression",
      "Perform periodic focused doc reviews for top-risk modules"
    ]
  }}
}}
""".strip()

    @staticmethod
    def get_modularization_plan_prompt(
        dependency_graph: Dict[str, Any],
        base_plan: Dict[str, Any],
    ) -> str:
        """Generate prompt for RTL modularization planning."""
        analysis = dependency_graph.get("analysis", {})

        action_counts: Dict[str, int] = {}
        for module_name, plan_data in base_plan.items():
            action = plan_data.get("action", "unknown")
            action_counts[action] = action_counts.get(action, 0) + 1

        actions_lines = [
            f"- {action}: {count} modules" for action, count in action_counts.items()
        ]

        return f"""
You are a senior Verilog/SystemVerilog hardware design architect refining an RTL modularization plan.

Current dependency metrics:
- Total modules: {analysis.get('total_nodes', 0)}
- Circular instantiation: {analysis.get('cycle_count', 0)}
- Max fan-out: {analysis.get('max_fan_out', 0)}

Summary of proposed actions:
{chr(10).join(actions_lines)}

Objectives:
- Enhance the modularization plan with precise, actionable guidance.
- Output must:
  - Reflect sound HDL design principles per the Reuse Methodology Manual.
  - Be directly usable by coding agents implementing the changes.

Please provide:

1) Target Module Boundaries
   - Describe the desired module boundaries and layering (e.g., interfaces, datapath, control, testbenches).
   - Clarify how to enforce cohesion and reduce coupling within clock domains and across domain crossings.

2) Package & Interface Strategy
   - Propose how to organize packages and modules:
     - shared packages vs module-local declarations,
     - stable interfaces vs implementation details.
   - Recommend interface patterns (interfaces, modports, parameterized wrappers) suitable for this design.

3) File Organization Strategy
   - Recommend how to structure synthesis units and file lists (e.g., per-module .sv files, hierarchy-aware organization).
   - Explain benefits in terms of synthesis times, elaboration speed, and verification isolation.

4) Migration Plan
   - Provide a phased, low-risk migration strategy from the current structure to the desired modular RTL architecture.
   - Each phase should have clear, implementable actions.

5) EDA Flow Adaptation
   - Recommend changes to synthesis scripts and Makefiles to reflect new modules.
   - Discuss hierarchical synthesis partitions, independent elaboration units, and elaboration-time considerations.

6) Verification Strategy for Modularization
   - Define how to validate modularization changes without regressions:
     - regression simulations,
     - module-level testbenches,
     - interface/contract verification tests.

7) Performance and Footprint Considerations
   - Call out performance and memory footprint risks and mitigations associated with the modularization.

Important response format requirements:
- Provide a brief, professional narrative only if needed.
- Then output a single JSON object matching the schema below, with nothing after it:

{{
  "architecture_assessment": {{
    "summary": "string",
    "target_structure": ["description of layers/modules"],
    "observations": ["string"]
  }},
  "issues": [
    {{
      "id": "MOD-ISSUE-1",
      "description": "Current structural/modular problem",
      "severity": "low|medium|high|critical",
      "impact": "maintainability|build|testability|performance"
    }}
  ],
  "proposed_solutions": [
    {{
      "issue_id": "MOD-ISSUE-1",
      "summary": "Short solution summary",
      "steps": [
        "Concrete step 1",
        "Concrete step 2"
      ],
      "implementation_hints": [
        "Hints for coding agent (e.g., create new library target 'libcore_net' and move files X,Y there)"
      ]
    }}
  ],
  "migration_plan": {{
    "phases": [
      {{
        "name": "Phase 1 - Identify and isolate core modules",
        "goals": ["string"],
        "actions": ["string"]
      }}
    ]
  }},
  "risks_and_concerns": [
    {{
      "description": "Risk from modularization",
      "likelihood": "low|medium|high",
      "impact": "low|medium|high|critical",
      "mitigations": ["string"]
    }}
  ],
  "recommendations": [
    {{
      "id": "MOD-REC-1",
      "priority": "P0|P1|P2",
      "description": "Strategic modularization recommendation",
      "related_issues": ["MOD-ISSUE-1"]
    }}
  ],
  "verification_plan": {{
    "overview": "How to verify modularization changes",
    "phases": [
      {{
        "name": "Phase 1 - Baseline regression",
        "goals": ["string"],
        "verification_types": ["unit_sim", "integration_sim", "system_sim"],
        "actions": [
          "Run full regression simulation after each major structural change",
          "Introduce contract verification for key module interfaces"
        ]
      }}
    ]
  }}
}}
""".strip()

    @staticmethod
    def get_validation_insights_prompt(
        validation_report: Dict[str, Any],
        modularization_plan: Dict[str, Any],
    ) -> str:
        """Generate prompt for validation insights."""
        base_validation = validation_report.get("base_validation", {})
        overall_score = base_validation.get("overall_score", 0)
        issues = base_validation.get("issues", [])

        issues_lines = [f"- {issue}" for issue in issues[:5]]

        return f"""
You are an HDL design quality and architecture expert validating an RTL modularization plan.

Validation results:
- Overall score: {overall_score}/100
- Issues identified: {len(issues)}

Sample key issues:
{chr(10).join(issues_lines)}

Objectives:
- Provide a professional, risk-focused validation assessment.
- Output must be clear enough for leadership and precise enough for coding agents.

Please address:

1) Risk Assessment
   - Identify the highest risks associated with the current modularization plan,
     including hidden dependencies, port signature breaks, regression risk, timing changes, and CDC violations.

2) Success Criteria
   - Define specific, measurable success criteria (e.g., reductions in synthesis time, defect rate, complexity).

3) Quality Gates
   - Propose quality gates during modularization:
     - verification coverage thresholds,
     - allowed number of circular instantiation cycles,
     - acceptable lint warning levels.

4) Rollback Strategy
   - Describe how to roll back modularization changes safely if significant issues are discovered.

5) Monitoring & Metrics
   - Identify key technical and process metrics to monitor during and after modularization.

6) Team & Process Impact
   - Highlight impacts on development workflows, code ownership, and review practices.

7) Tooling Requirements
   - Recommend tools (module hierarchy visualizers, lint analyzers, CDC checkers, formal verification, CI checks) necessary to support safe modularization.

8) Timeline Considerations
   - Provide guidance on realistic timelines and sequencing for Verilog/SystemVerilog RTL modularization efforts.

Important response format requirements:
- You may include a concise commentary.
- Then output a single JSON object with the following structure, and nothing after it:

{{
  "risk_assessment": [
    {{
      "id": "VAL-RISK-1",
      "description": "Risk description",
      "likelihood": "low|medium|high",
      "impact": "low|medium|high|critical",
      "mitigations": ["string"]
    }}
  ],
  "success_criteria": [
    {{
      "id": "VAL-SC-1",
      "description": "Success metric or criteria",
      "metric_type": "synthesis_time|defects|coverage|complexity|timing|cdc_safety|other",
      "target_value": "string"
    }}
  ],
  "quality_gates": [
    {{
      "id": "VAL-QG-1",
      "description": "Quality gate description",
      "metric": "coverage|lint_warnings|circular_instantiation|cdc_violations|other",
      "threshold": "string"
    }}
  ],
  "rollback_strategy": {{
    "overview": "How to roll back safely",
    "steps": ["string"]
  }},
  "monitoring_plan": {{
    "metrics": ["string"],
    "reporting_frequency": "string",
    "responsible_roles": ["string"]
  }},
  "recommendations": [
    {{
      "id": "VAL-REC-1",
      "priority": "P0|P1|P2",
      "description": "Validation-related recommendation"
    }}
  ],
  "verification_plan": {{
    "overview": "Validation verification plan",
    "phases": [
      {{
        "name": "Phase 1 - Pre-modularization baseline",
        "goals": ["string"],
        "actions": ["string"]
      }},
      {{
        "name": "Phase 2 - Incremental validation of modular changes",
        "goals": ["string"],
        "actions": ["string"]
      }}
    ]
  }}
}}
""".strip()

    @staticmethod
    def get_final_report_prompt(
        summary: Dict[str, Any],
        dependency_graph: Dict[str, Any],
        documentation: Dict[str, Any],
        modularization_plan: Dict[str, Any],
        validation_report: Dict[str, Any],
        health_metrics: Dict[str, Any],
    ) -> str:
        """Generate prompt for final comprehensive report."""
        file_stats = summary.get("file_stats", {})
        total_files = file_stats.get("total_files", 0)
        languages = file_stats.get("languages", {})

        overall_health = health_metrics.get("overall_health", {})
        health_score = overall_health.get("score", 0)
        health_grade = overall_health.get("grade", "F")

        dep_analysis = dependency_graph.get("analysis", {})
        has_cycles = dep_analysis.get("has_cycles", False)

        return f"""
You are an expert Verilog/SystemVerilog hardware design architect preparing an executive-level final report for stakeholders.

Design overview:
- Total files: {total_files}
- Languages: {dict(languages)}
- Overall Design Health: {health_score}/100 ({health_grade})
- Circular Module Instantiation: {"Yes" if has_cycles else "No"}

Objectives:
- Provide a professional, concise executive summary that is technically accurate.
- Output should drive decision-making at the leadership level and be structured enough for tooling and coding agents.

Your report should cover:

1) Executive Summary
   - High-level assessment of design health, major strengths, and major weaknesses.

2) Critical Issues
   - 5–10 critical issues across security, architecture, reliability, complexity, verification coverage, and maintainability.

3) Strategic Recommendations
   - Long-term improvement themes and strategic initiatives.

4) Implementation Roadmap
   - A phased roadmap (short-, medium-, and long-term).
   - Each phase should have clear goals and key actions.

5) Resource & Effort Estimates
   - High-level estimates of effort (small/medium/large) and skill requirements.

6) Risk Assessment
   - Key risks of not acting.
   - Key risks associated with major changes.

7) Success Metrics
   - How to measure success of the improvement program (technical and process metrics).

8) Design Methodology Modernization Opportunities
   - Opportunities for modernization (e.g., SystemVerilog features, UVM adoption, modern verification methodology, advanced synthesis techniques).

9) Team & Process Recommendations
   - Recommendations around practices, reviews, ownership, and skills.

10) Monitoring Strategy
    - How to continuously monitor health and prevent regressions.

Important response format requirements:
- You may write a concise executive narrative if desired.
- Then produce a single JSON object and nothing after it, conforming exactly to this schema:

{{
  "executive_summary": {{
    "overview": "string",
    "key_findings": ["string"]
  }},
  "critical_issues": [
    {{
      "id": "CRIT-1",
      "title": "Short issue name",
      "description": "string",
      "category": "security|architecture|quality|verification|performance|synthesis|timing|cdc|other",
      "severity": "low|medium|high|critical",
      "impact": "string"
    }}
  ],
  "strategic_recommendations": [
    {{
      "id": "STRAT-1",
      "title": "Recommendation name",
      "description": "string",
      "priority": "P0|P1|P2",
      "related_issues": ["CRIT-1"]
    }}
  ],
  "roadmap": [
    {{
      "phase": 1,
      "name": "Phase name",
      "time_horizon": "short|medium|long",
      "goals": ["string"],
      "key_actions": ["string"]
    }}
  ],
  "resource_requirements": [
    {{
      "id": "RES-1",
      "description": "Needed skills/resources",
      "size": "small|medium|large"
    }}
  ],
  "risks_and_concerns": [
    {{
      "description": "Risk description",
      "likelihood": "low|medium|high",
      "impact": "low|medium|high|critical",
      "mitigations": ["string"]
    }}
  ],
  "success_metrics": [
    {{
      "id": "MET-1",
      "description": "What to measure",
      "metric_type": "coverage|defects|synthesis_time|performance|security|timing|cdc_safety|other",
      "target_value": "string"
    }}
  ],
  "modernization_opportunities": [
    {{
      "id": "MOD-1",
      "description": "Modernization opportunity (e.g., adopt SystemVerilog assertions in module X, integrate UVM testbench)"
    }}
  ],
  "team_and_process_recommendations": [
    {{
      "id": "TEAM-1",
      "description": "Team/process improvement"
    }}
  ],
  "monitoring_strategy": {{
    "overview": "string",
    "metrics": ["string"],
    "cadence": "string"
  }}
}}
""".strip()

    @staticmethod
    def get_security_focus_prompt(security_metrics: Dict[str, Any]) -> str:
        """Generate prompt focused on HDL design security and reliability analysis."""
        security_score = security_metrics.get("score", 0)
        security_grade = security_metrics.get("grade", "F")
        issues = security_metrics.get("issues", [])

        issues_lines = [f"- {issue}" for issue in issues[:5]]

        return f"""
You are an HDL design security and reliability expert performing a focused assessment of design robustness.

Design reliability assessment summary:
- Reliability Score: {security_score}/100 ({security_grade})
- Issues identified: {len(issues)}

Sample top reliability concerns:
{chr(10).join(issues_lines)}

Objectives:
- Provide a highly professional, in-depth design reliability analysis.
- Output should support both RTL engineers and coding agents performing remediations.

Please address:

1) Signal Integrity & Initialization
   - Identify and characterize signal integrity risks (uninitialized signals, multiple drivers, floating nets, missing resets).

2) Width Management & Mismatches
   - Highlight risks from signal width handling (truncation, sign extension, bus width mismatches, parameterization issues).

3) Port Validation & CDC Safety
   - Comment on likely input validation weaknesses and clock domain crossing (CDC) synchronization issues.

4) Reliable Design Practices
   - Recommend relevant design standards and conventions for this codebase (e.g., synthesis best practices, Reuse Methodology Manual, CDC guidelines).

5) Static Lint & Formal Verification
   - Propose concrete static lint and formal verification tooling and how to integrate it into the design flow.

6) Design Review Focus
   - Define what reliability-focused design reviews should explicitly look for.

7) Violation Remediation Plan
   - Provide a prioritized remediation roadmap, ordered by risk and impact on functionality.

8) Reliability Verification Plan
   - Recommend constrained random verification, formal verification, lint analysis, and regression testing strategies.

Important response format requirements:
- You may include a brief professional commentary.
- Then output a single JSON object and nothing after it, respecting this schema:

{{
  "issues": [
    {{
      "id": "SEC-ISSUE-1",
      "category": "signal_integrity|width_mismatch|cdc_safety|reset_safety|synthesis|timing|other",
      "description": "Design reliability issue description",
      "severity": "low|medium|high|critical",
      "likelihood": "low|medium|high",
      "impact": "low|medium|high|critical"
    }}
  ],
  "proposed_solutions": [
    {{
      "issue_id": "SEC-ISSUE-1",
      "summary": "Short fix summary",
      "steps": [
        "Concrete remediation step 1 (e.g., add CDC synchronizer between clock domains, fix signal width mismatch)",
        "Concrete remediation step 2"
      ],
      "implementation_hints": [
        "Hints for a coding agent (e.g., search for uninitialized registers, add reset to all flops, parameterize all bus widths)"
      ]
    }}
  ],
  "architecture_assessment": {{
    "summary": "How the architecture affects the design reliability posture",
    "observations": ["string"]
  }},
  "risks_and_concerns": [
    {{
      "description": "Risk if issues remain unaddressed",
      "likelihood": "low|medium|high",
      "impact": "low|medium|high|critical",
      "mitigations": ["string"]
    }}
  ],
  "recommendations": [
    {{
      "id": "SEC-REC-1",
      "priority": "P0|P1|P2",
      "description": "Reliability program recommendation (e.g., adopt synthesis best practices, add CDC checker to flow)",
      "related_issues": ["SEC-ISSUE-1"]
    }}
  ],
  "verification_plan": {{
    "overview": "Reliability verification strategy",
    "actions": [
      "Integrate Verilator lint / Spyglass into continuous verification for key modules",
      "Introduce constrained random verification for corner cases and cross-domain interactions",
      "Add formal verification and CDC analysis for critical CDC paths and state machines"
    ]
  }}
}}
""".strip()

    @staticmethod
    def get_intent_prompt_for_metrics(user_input_prompt: str) -> str:
        """
        Generate the system prompt for intent extraction related to HDL design analysis
        and metrics queries (retrieve/compare/aggregate).

        This prompt remains focused on intent extraction and is unchanged in structure.
        """
        base_prompt = """
You are a senior HDL codebase analysis assistant focused on design health metrics and architecture.

Analyze the following user request and return a JSON object describing the user's intent
for design analysis and reporting.

You have access to detailed metrics such as:
- complexity (module/file-level metrics, parameterization)
- maintainability (hierarchy health, modularity hotspots)
- security (design reliability, CDC safety, signal integrity issues)
- quality (lint violations, synthesis feasibility, style, HDL hygiene)
- documentation (coverage, clarity, completeness)
- verification_coverage (simulation coverage, formal verification, lint integration)
- dependency (module hierarchy health, cycles, coupling)
- overall_health (combined scores and grades)

Supported intents:
- "retrieve": get metrics or summaries for specific modules/files/components.
- "compare": compare metrics between entities (modules, components, releases).
- "aggregate": summarize or aggregate metrics over the entire codebase.

Guidelines:
- For comparison, set "intent": "compare" and list "entities", each with its own criteria.
- For retrieval, set "intent": "retrieve" with a "criteria" object
  (e.g., {"module": "core.network", "metric": "security"}).
- For aggregation (e.g., "overall health score", "top 10 risky modules"),
  use "intent": "aggregate" and put the aggregation target in "criteria".

- If the request applies to ALL modules or the entire codebase, leave "criteria": {}.
- If a specific output format is requested (table, summary, JSON, list), include "output_format".
- If the user requests specific artifacts (e.g., "final report", "security summary",
  "modularization plan"), include them in "fields_to_extract".
- Always return a single JSON object, with no explanation or additional text.

Example 1:
User: "Compare security and complexity metrics for core.network and core.storage as a table."
Return:
{
  "intent": "compare",
  "entities": [
    {"module": "core.network"},
    {"module": "core.storage"}
  ],
  "fields_to_extract": ["security", "complexity"],
  "output_format": "table"
}

Example 2:
User: "Give me a summary of all health metrics for the networking module."
Return:
{
  "intent": "retrieve",
  "criteria": {"module": "network", "target": "health_metrics"},
  "fields_to_extract": ["overall_health", "critical_issues", "recommendations"],
  "output_format": "summary"
}

Example 3:
User: "Show the top 10 highest-risk modules based on combined scores."
Return:
{
  "intent": "aggregate",
  "criteria": {"target": "top_risk_modules", "limit": 10},
  "fields_to_extract": ["module", "overall_health", "security", "complexity"],
  "output_format": "table"
}

User request:
""".strip()

        return f"{base_prompt}\n{user_input_prompt}"