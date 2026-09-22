# Example programs for the decimate library.
#
# These are simple, runnable C programs that link against libdecimate so users
# can see the public ABI in action and benchmark it. The Python example lives
# at examples/example_decimate.py and is run directly (no build step).

add_executable(example_c example_c.c)
target_link_libraries(example_c PRIVATE decimate::decimate)

# Register as a test-free ctest entry that merely runs the example and prints
# timing; it doubles as a smoke test of the public ABI end-to-end.
add_test(NAME example_c COMMAND example_c)
