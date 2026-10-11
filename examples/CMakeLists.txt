# Example programs for the decimate library.
#
# These are simple, runnable C programs that link against libdecimate so users
# can see the public ABI in action and benchmark it. The Python example lives
# at examples/example_decimate.py and is run directly (no build step).

add_executable(example_c example_c.c)
target_link_libraries(example_c PRIVATE decimate::decimate)

# example_c.c calls libm functions (acos, sincosf) directly; on Linux these
# are not pulled in transitively via decimate::decimate, so link explicitly.
if(UNIX AND NOT APPLE)
    target_link_libraries(example_c PRIVATE m)
endif()

# Register as a test-free ctest entry that merely runs the example and prints
# timing; it doubles as a smoke test of the public ABI end-to-end.
add_test(NAME example_c COMMAND example_c)

# On Windows, the DLL lives in a different output directory
# ($<TARGET_FILE_DIR:decimate>) than the example executable, so the loader
# cannot find it via the default search path. Without this, the process
# hits a native "dependent DLL not found" hard-error dialog that hangs
# forever on a headless CI runner (no one to dismiss it).
if(WIN32)
    set_tests_properties(example_c PROPERTIES
        ENVIRONMENT "PATH=$<TARGET_FILE_DIR:decimate>;$ENV{PATH}")
endif()
