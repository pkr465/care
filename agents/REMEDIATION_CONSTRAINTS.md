# REMEDIATION_CONSTRAINTS.md

## 1. Overview
This document defines strict constraints and overrides for the **Automated Remediation Agent**. These rules take precedence over standard security linter suggestions and generic best practices. The codebase contains performance-critical paths where standard "safe" APIs introduce unacceptable latency.

## 2. Critical Performance Constraints

### 2.1 Memory Copy Operations
**Constraint:** Do **NOT** replace `std::memcpy`, `memcpy`, or `memmove` with `memscpy`, `memcpy_s`, or other "secure" variants in the data path.  
**Reasoning:** The overhead of the secure API wrapper introduces latency that violates our real-time processing requirements (Budget: < 5μs per packet).

**Required Action:**  
Instead of changing the API, you must ensure bounds safety via **explicit pre-validation** using standard C++ logic.

*   ❌ **Prohibited Fix:**
    ```cpp
    // DO NOT DO THIS
    memscpy(out, out_size, in.data(), copy_len);
    ```

*   ✅ **Allowed Fix:**
    ```cpp
    // DO THIS INSTEAD
    if (copy_len > out_size) {
        // Handle error (log, truncate, or return error code)
        return ERR_BUFFER_OVERFLOW;
    }
    std::memcpy(out, in.data(), copy_len);
    ```

## 3. Linter Rule Overrides

The following static analysis rules must be handled differently than their standard remediation descriptions suggest.

| Rule ID    | Standard Description              | Standard Remediation           | **LLM Action / Constraint**                                                                                                                                         |
| :--------- | :-------------------------------- | :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **QCT001** | `memcpy` used; prefer `memscpy()` | Replace with `memscpy`         | **IGNORE** the standard remediation. **RETAIN** `memcpy`. Add an explicit `if` check for buffer boundaries immediately before the call if one is missing.           |
| **QCT055** | Raw pointer usage                 | Use `std::shared_ptr`          | **RESTRICTED**. Do not introduce smart pointers in `framework/platform/bpl` or kernel-adjacent code. Use `std::unique_ptr` only if zero-overhead can be guaranteed. |
| **SYS009** | Uninitialized variable            | Initialize to `0` or `nullptr` | **CONTEXT AWARE**. Ensure initialization does not occur inside a hot loop unless necessary.                                                                         |

## 4. General Code Integrity

1.  **No External Dependencies:** Do not import new libraries (e.g., Boost, Abseil) to solve a syntax issue. Use only the Standard Template Library (STL) or existing project utilities.
2.  **Preserve Signatures:** Do not change function signatures in public headers (`.h`) unless explicitly requested. Fixes must be contained within the implementation (`.cpp`) bodies.
3.  **Comment Requirement:** If a security rule is suppressed or solved via manual checks (like the `memcpy` case), add a comment explaining why:
    ```cpp
    // INTENTIONAL: Using memcpy for performance. Bounds checked above.
    std::memcpy(out, in, len);
    ```

## 5. Example Scenario (QCT001)

**Input Violation:**
```json
{
  "rule": "QCT001",
  "severity": "high",
  "description": "memcpy used; prefer memscpy()",
  "snippet": "std::memcpy(out, in.data(), copy_len);",
  "remediation": "Use memscpy(dst, dst_size, src, src_size)"
}
```

**Correct LLM Generated Patch:**
```diff
- std::memcpy(out, in.data(), copy_len);
+ // SAFETY: Manual bounds check to avoid memscpy latency overhead
+ if (copy_len > sizeof(out_buffer_size)) {
+     LOG_ERROR("Buffer overflow detected");
+     return;
+ }
+ std::memcpy(out, in.data(), copy_len);
```
