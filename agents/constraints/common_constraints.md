# Verilog/SystemVerilog Hardware Design Constraints

## 1. Issue Identification Rules
*Use these rules to filter out false positives during analysis.*

### A. Clock Domain Annotations
*   **Context**: Signals crossing clock domains should be annotated with their source and destination clock domains.
*   **Rule**: **IGNORE** CDC warnings if both source and destination clocks are the same (same clock_name in comments).
*   **Rule**: **FLAG** CDC hazards only when crossing different clock domains WITHOUT a synchronizer.
*   **Synchronizer Recognition**: 2-stage flip-flop chain, Gray-coded crossing, or async FIFO are valid synchronizers.

### B. Parameterized Generate Blocks
*   **Context**: Generate loops with parameter-defined iteration counts.
*   **Rule**: **IGNORE** warnings about unbounded generate loops if the generating parameter is bounded by constraint.
*   **Example**: `for (i = 0; i < DEPTH; i++)` where DEPTH is defined as localparam or parameter with explicit bound is SAFE.
*   **Rule**: **FLAG** only if generate loop variable is unconstrained or derived from user input.

### C. Memory Inference (RAM/ROM)
*   **Context**: Synchronous single-port or dual-port memory accessed via always @(posedge clk).
*   **Rule**: **IGNORE** false positives about undefined behavior in memory semantics.
*   **Pattern**: `always @(posedge clk) mem[addr] <= din;` is standard RAM inference and is SAFE.
*   **Rule**: **FLAG** only if memory addressing is out-of-bounds or width mismatch is real.

### D. Synthesis Pragmas & Attributes
*   **Context**: Pragmas like full_case, parallel_case, or synthesis attributes.
*   **Rule**: **IGNORE** warnings about "unreachable" case statements if full_case pragma is present.
*   **Rule**: **FLAG** only if the pragma contradicts actual code behavior.

### E. Interface & Modport Semantics
*   **Context**: SystemVerilog interfaces and modports provide port direction guarantees.
*   **Rule**: **IGNORE** "driver/reader mismatch" warnings on interface signals when proper modports are used.
*   **Rule**: **FLAG** only if a modport is misused (e.g., trying to drive an input modport signal).

### F. Asynchronous Reset Patterns
*   **Context**: Asynchronous reset is common in hardware; deassertion must be synchronized to avoid metastability.
*   **Rule**: **IGNORE** false positives about reset path detection if synchronizer is present in include context.
*   **Rule**: **FLAG** only if reset is applied asynchronously without proper synchronization on deassertion.

### G. Blocking vs Non-Blocking Assignments
*   **Context**: Sequential blocks (always @(posedge clk)) must use non-blocking (<=), combinational (@(*)) must use blocking (=).
*   **Rule**: **IGNORE** style warnings about assignment style if functionally correct (rare corner cases).
*   **Rule**: **FLAG** blocking in sequential (indicates race condition) and non-blocking in combinational (indicates timing issue).

### H. Coverage & Assertions
*   **Context**: Assertions and coverage directives for verification.
*   **Rule**: **IGNORE** warnings about missing assertions in testbench-only code or simulation pragmas.
*   **Rule**: **FLAG** only if a critical safety property is unverified in production code.

---

## 2. Issue Resolution Rules
*Use these rules when generating code fixes.*

### A. CDC (Clock Domain Crossing) Fixes
*   **Rule**: Always use 2-stage synchronizer minimum for single-bit CDC: `reg [1:0] sync_ff; always @(posedge dest_clk) sync_ff <= {sync_ff[0], cdc_signal};`
*   **Rule**: For multi-bit CDC, use Gray encoding or async FIFO. Do NOT synchronize each bit independently.
*   **Rule**: Document CDC paths in comments: `// CDC: signal crosses from clk_a to clk_b via sync_ff2`

### B. Reset Strategy Consistency
*   **Rule**: Within a clock domain, reset strategy must be consistent (all async or all sync, not mixed).
*   **Rule**: Asynchronous reset should be synchronized on deassertion: `always @(posedge clk or negedge arst_n) q <= arst_n ? 1'b0 : d;`
*   **Rule**: Synchronous reset is preferred for clocked logic: `always @(posedge clk) if (reset) q <= 1'b0; else q <= d;`

### C. Blocking vs Non-Blocking (Golden Rule)
*   **Rule**: Always block (always @(posedge clk)) → Use non-blocking (<=) assignments ONLY.
*   **Rule**: Combinational block (always @(*)) → Use blocking (=) assignments ONLY.
*   **Rule**: Continuous assignment → Use assign for combinational logic.
*   **Exception**: Do NOT change blocking to non-blocking or vice versa without understanding functional impact.

### D. Module Port Lists & Signatures
*   **Rule**: DO NOT modify module port names, directions, or widths in public interfaces.
*   **Rule**: Fixes must preserve module signature compatibility with instantiation sites.
*   **Rule**: If a port width is parameterized, ensure parameter is visible to all instantiators.

### E. Parameter & Localparam Usage
*   **Rule**: Use `parameter` for values overridable at instantiation time.
*   **Rule**: Use `localparam` for values local to a module, not overridable.
*   **Rule**: When adding parameters for bounds checking, ensure they have sensible defaults.

### F. Sensitivity List Completeness
*   **Rule**: For combinational always blocks, list all read signals or use @(*) (preferred in modern SystemVerilog).
*   **Rule**: For sequential blocks, sensitivity list is typically @(posedge clk) or @(posedge clk or negedge arst_n).
*   **Rule**: When fixing, prefer @(*) for combinational to avoid maintenance burden.

### G. Generate Block Parameterization
*   **Rule**: When using generate loops, explicitly document the loop bounds.
*   **Rule**: Ensure loop variable is constrained by a parameter with documented maximum value.
*   **Rule**: Add comments: `// Generate DEPTH instances, where DEPTH = parameter from module declaration`

### H. Testbench vs RTL Code
*   **Rule**: DO NOT flag testbench constructs (initial blocks, $display, $finish) as production issues.
*   **Rule**: Testbench code is allowed to use blocking assignments, $strobe, deassert timing, etc.
*   **Rule**: Separate testbench checks from RTL checks if the file is marked as testbench (_tb.sv).

### I. Code Integrity Rules (LLM Fix Generation — Compilation Safety)
*These rules prevent the most common compilation-breaking errors in LLM-generated fixes.*

*   **I.1 — Blank Line Preservation**:
    - Blank lines around `end`, `endmodule`, `endfunction`, `endtask` are **CRITICAL** for readability.
    - If the original code has `end\n\nalways @(posedge clk)`, the fixed code **MUST** maintain this spacing.
    - **DO NOT** merge lines by removing newlines between blocks or after closing keywords.

*   **I.2 — Function/Task Signature Consistency**:
    - When modifying a function CALL (e.g., adding an argument), the function DEFINITION **MUST** also be updated.
    - Argument counts in calls **MUST** match argument counts in definitions.
    - **DO NOT** add extra parameters to function calls without changing the corresponding definition.
    - If the function definition is not visible in the current chunk, **DO NOT** change the call arguments.

*   **I.3 — Instantiation Parameter Matching**:
    - When an instance uses parameters (e.g., `#(.WIDTH(8)) my_module inst(...)`), ensure the module definition accepts those parameters.
    - **DO NOT** add parameters to instantiation if the module definition doesn't declare them.
    - If modifying parameter values, ensure they are within the module's valid range (if documented).

*   **I.4 — Type Declaration Preservation**:
    - Loop variable declarations **MUST** retain their type: `for (int i = 0; i < N; i++)`
    - **DO NOT** remove type declarations from variable initializers.
    - **DO NOT** introduce narrower types that cause truncation (e.g., changing int to bit).

*   **I.5 — Width Specification Preservation**:
    - When declaring buses or arrays, preserve width specifications: `logic [DATA_WIDTH-1:0]`
    - **DO NOT** change `[N-1:0]` to `[N]` or similar without understanding the shift.
    - **DO NOT** assume default widths (1 bit) when explicit widths are needed.

*   **I.6 — Clock Domain Annotation Consistency**:
    - If a signal is marked as belonging to a clock domain in comments, all synchronizers must respect that domain.
    - **DO NOT** change clock domain annotations without updating synchronizer instantiations.

*   **I.7 — Macro & Generate Constraints**:
    - Use only macros defined or visible in the current file or included .svh files.
    - Do NOT introduce undefined `define or parameter references.
    - When using generate, ensure generating parameters are defined with valid bounds.

### J. Performance & Timing Considerations
*   **Linting Suppression Override**: Do NOT introduce timing overhead when fixing CDC, reset, or synchronization issues. Synchronizers add minimal latency (2-3 clock cycles) which is expected.
*   **Pipeline Insertion**: When fixing long combinational paths, insert registers strategically; prefer slicing data path over control path.
*   **Hot Path Avoidance**: Do NOT add overhead (extra comparisons, locks, etc.) in packet data paths or high-frequency control paths without justification.

### K. Documentation & Comments
*   **CDC Documentation**: Every cross-domain signal should have a comment indicating source and destination clock domains.
*   **Reset Documentation**: Async reset paths should document synchronization strategy.
*   **Parameter Bounds**: Document maximum values for parameterized elements (generate loop counts, FIFO depths, etc.).

---

## 3. False Positive Patterns (Common)

### A. Inferred Latch from Case Statements
*   **Pattern**: `case (sel) 2'b00: out = a; 2'b01: out = b; 2'b10: out = c; 2'b11: out = d; endcase` in combinational block.
*   **Reality**: If all cases are covered (2'b00, 2'b01, 2'b10, 2'b11), NO latch is inferred.
*   **Rule**: Do NOT flag as latch inference if all cases of the selector width are covered.

### B. "Missing Reset" on Always @(*)
*   **Pattern**: Combinational always @(*) block without reset clause.
*   **Reality**: Combinational logic does NOT need reset; it's computed continuously.
*   **Rule**: Do NOT flag missing reset on combinational blocks.

### C. Unregistered Output from Combinational Logic
*   **Pattern**: Output driven by combinational always @(*) block.
*   **Reality**: Combinational output is valid and synthesizes to gates, not registers.
*   **Rule**: Do NOT flag unregistered outputs unless there's an explicit timing requirement.

### D. Parameter Width Mismatch (False Positive)
*   **Pattern**: `logic [WIDTH-1:0]` assigned to `logic [8:0]` in a parameterized design.
*   **Reality**: If WIDTH > 9 in actual instantiation, mismatch is real. If WIDTH <= 9, it's safe.
*   **Rule**: Do NOT flag width mismatch if the mismatch is contingent on parameter values not visible in this chunk.
*   **Rule**: Only FLAG if WIDTH is statically known and too small.

### E. Clock Domain Crossing Without Context
*   **Pattern**: Signal assigned in always @(posedge clk_a) block, read in always @(posedge clk_b) block.
*   **Reality**: May be intentional CDC with synchronizer, but synchronizer might be in another file or module.
*   **Rule**: Do NOT flag as CDC hazard if a synchronizer is documented in include context or visible on the receiving side.

### F. "Blocking Assignment in Sequential Block"
*   **Pattern**: `always @(posedge clk) out = in;` (rare, but valid in some contexts like combinational output).
*   **Reality**: This assigns combinational logic to be driven on clock edge; NOT a sequential register.
*   **Rule**: FLAG only if the intent is sequential storage. Do NOT flag if it's intentional combinational assignment on clock event (valid but rare).

---

## 4. Module & Interface Declaration Rules

### A. Module Instantiation Validation
*   **Check**: Instance port count matches module port count.
*   **Check**: Instance port directions are compatible (output from module to input of instance user).
*   **Check**: Port widths match (or are parameterized consistently).
*   **Rule**: Do NOT flag port mismatches if parameters make them compatible at elaboration time.

### B. Parameter Override Validation
*   **Check**: Parameter overrides at instantiation are within module's declared type range.
*   **Check**: Dependencies between parameters are satisfied.
*   **Rule**: Do NOT flag parameter overrides if they are within valid range in the module definition (in include context).

### C. Interface Modport Compliance
*   **Check**: Tasks/functions using modports access only the signals available in that modport.
*   **Rule**: Do NOT flag modport violations if the accessing task is not defined in this chunk (it may have correct modports visible elsewhere).

---

