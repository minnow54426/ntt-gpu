# Run and debug the NTT in VS Code

Use the N=8 CPU example to follow an eight-coefficient polynomial. The forward
transform computes its values at eight special points; the inverse recovers
the coefficients. No CUDA hardware is needed. The terminal commands work
on supported CPU platforms; the supplied VS Code debugger setup targets macOS.

## One-time setup

1. Open the repository folder in VS Code: `code .`
2. If VS Code asks, install the recommended extensions:
   - CMake Tools
   - C/C++
   - LLDB DAP
3. Wait for CMake Tools to finish configuring `build/debug`.

The workspace uses Xcode's `lldb-dap`, the program that lets VS Code control
the debugger. Install the Xcode command-line tools before using this configuration.

## Run the small example

From a terminal:

```sh
cmake -S . -B build -DNTT_ENABLE_CUDA=OFF
cmake --build build --target ntt_debug_example
./build/ntt_debug_example
```


Open the command palette with `Cmd+Shift+P`, choose **Tasks: Run Task**, then
choose **Run: NTT debug example**.

The expected output is:

```text
natural-order coefficients: 1 2 3 4 5 6 7 8
bit-reversed evaluation values: 5 0 13 8 9 11 5 8
recovered coefficients: 1 2 3 4 5 6 7 8
roundtrip passed
```

## Debug it step by step

1. Open `examples/step_by_step.cpp`.
2. Click left of the `transform.forward(values)` line to create a breakpoint.
3. Open **Run and Debug** with `Cmd+Shift+D`.
4. Select **Debug: NTT example (step by step)**.
5. Press `F5`. The program first stops at `main` because `stopOnEntry` is on.
6. Press `F5` again to reach the breakpoint.
7. Press `F11` to step into `NegacyclicNtt::forward` in `include/ntt/ntt.hpp`.
8. Use `F10` to advance one source line at a time.

Start by watching `values` (the array), `upper_index` and `lower_index` (the
two positions being combined), and `twiddle_power` (their modular multiplier).
Then inspect `stage_pair_distance`, `group_start_index` and `pair_offset`
to understand how the loops choose that pair.

For `N=8`, the forward transform has three stages with pair distances
`4`, `2`, and `1`. Expand `values` after each butterfly to see the array
change. The code overwrites the same array rather than creating a new one.

## Other shared commands

- `Cmd+Shift+B`: configure and build the debug example.
- **Test: All CPU checks**: build and run the complete CPU test suite.
- **Debug: CPU unit tests**: generate the `N=1024` fixture and debug the tests.

The debugger deliberately targets the CPU reference. CUDA kernel debugging
will be added on the RTX machine using `cuda-gdb` or Nsight after the native
CUDA build is available.

## CodeLLDB crash on current macOS/Xcode

<details markdown="1">
<summary>Recorded workaround for CodeLLDB 1.11.4</summary>


The recorded CodeLLDB 1.11.4 / LLDB 19.1.7 setup crashed with `SIGBUS` on
AppleClang symbols. Xcode `lldb-dap` stopped correctly at `main`. If an old
`lldb` configuration appears, run **Developer: Reload Window** and select
the `lldb-dap` configuration.

</details>
