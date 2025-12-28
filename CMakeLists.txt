cmake_minimum_required(VERSION 3.5)
project(procyon-lib)

if (UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

# PROJECT-WIDE TARGET SETTINGS
add_library(procyon-lib-settings INTERFACE)
if(MSVC)
    target_compile_options(procyon-lib-settings INTERFACE
        -W3 -wd4201 -wd4100 -wd4189 -wd4505 -wd4127
    )
endif()

# MATH LIBRARY
add_library(procyon-lib-math INTERFACE)
target_include_directories(procyon-lib-math INTERFACE src)

# CORE LIBRARY:
add_library(procyon-lib-core STATIC
    src/p_core/p_core.c
)
target_link_libraries(procyon-lib-core PRIVATE procyon-lib-math)
target_include_directories(procyon-lib-core PUBLIC src)

# UTILITY LIBRARY
add_library(procyon-lib-util INTERFACE)
target_include_directories(procyon-lib-util INTERFACE src)

# TESTS
add_executable(procyon-lib-tests tests/test_main.c)
target_link_libraries(procyon-lib-tests PRIVATE
    procyon-lib-settings
    procyon-lib-core
    procyon-lib-util
)
set_target_properties(procyon-lib-tests PROPERTIES EXCLUDE_FROM_ALL TRUE)