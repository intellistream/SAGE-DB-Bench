# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# Parameters:
#
# BOOST_ROOT:
#   Specify root of the Boost library if Boost cannot be auto-detected. On Windows, a fallback to a
#   downloaded nuget version will be used if Boost cannot be found.
#
# DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS:
#   This is a work-in-progress feature, not completed yet. The core DiskANN library will be split into
#   build-related and search-related functionality. In build-related functionality, when using tcmalloc,
#   it's possible to release memory that's free but reserved by tcmalloc. Setting this to true enables
#   such behavior.
#   Contact for this feature: gopalrs.

# Some variables like MSVC are defined only after project(), so put that first.
cmake_minimum_required(VERSION 3.15)
project(diskann)

set(CMAKE_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT MSVC)
	set(CMAKE_CXX_COMPILER g++)
endif()

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

# Install nuget packages for dependencies.
if (MSVC)
    find_program(NUGET_EXE NAMES nuget)

    if (NOT NUGET_EXE)
        message(FATAL_ERROR "Cannot find nuget command line tool.\nPlease install it from e.g. https://www.nuget.org/downloads")
    endif()

    set(DISKANN_MSVC_PACKAGES_CONFIG ${CMAKE_BINARY_DIR}/packages.config)
    set(DISKANN_MSVC_PACKAGES ${CMAKE_BINARY_DIR}/packages)

    message(STATUS "Invoking nuget to download Boost, OpenMP and MKL dependencies...")
    configure_file(${PROJECT_SOURCE_DIR}/windows/packages.config.in ${DISKANN_MSVC_PACKAGES_CONFIG})
    exec_program(${NUGET_EXE} ARGS install \"${DISKANN_MSVC_PACKAGES_CONFIG}\" -ExcludeVersion -OutputDirectory \"${DISKANN_MSVC_PACKAGES}\")
    if (RESTAPI)
	    set(DISKANN_MSVC_RESTAPI_PACKAGES_CONFIG ${CMAKE_BINARY_DIR}/restapi/packages.config)
	    configure_file(${PROJECT_SOURCE_DIR}/windows/packages_restapi.config.in ${DISKANN_MSVC_RESTAPI_PACKAGES_CONFIG})
        exec_program(${NUGET_EXE} ARGS install \"${DISKANN_MSVC_RESTAPI_PACKAGES_CONFIG}\" -ExcludeVersion -OutputDirectory \"${DISKANN_MSVC_PACKAGES}\")
    endif()
    message(STATUS "Finished setting up nuget dependencies")
endif()




include_directories(${PROJECT_SOURCE_DIR}/include)

# It's necessary to include tcmalloc headers only if calling into MallocExtension interface.
# For using tcmalloc in DiskANN tools, it's enough to just link with tcmalloc.
if (DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS)
    include_directories(${PROJECT_SOURCE_DIR}/gperftools/src)

    if (MSVC)
        include_directories(${PROJECT_SOURCE_DIR}/gperftools/src/windows)
    endif()
endif()

#OpenMP
if (MSVC)
    # Do not use find_package here since it would use VisualStudio's built-in OpenMP, but MKL libraries
    # refer to Intel's OpenMP.
    #
    # No extra settings are needed for compilation: it only needs /openmp flag which is set further below,
    # in the common MSVC compiler options block.
    include_directories(BEFORE "${DISKANN_MSVC_PACKAGES}/intelopenmp.devel.win/lib/native/include")
    link_libraries("${DISKANN_MSVC_PACKAGES}/intelopenmp.devel.win/lib/native/win-x64/libiomp5md.lib")

    set(OPENMP_WINDOWS_RUNTIME_FILES
        "${DISKANN_MSVC_PACKAGES}/intelopenmp.redist.win/runtimes/win-x64/native/libiomp5md.dll"
        "${DISKANN_MSVC_PACKAGES}/intelopenmp.redist.win/runtimes/win-x64/native/libiomp5md.pdb")
else()
    find_package(OpenMP)

    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    else()
        message(FATAL_ERROR "No OpenMP support")
    endif()
endif()

# DiskANN core uses header-only libraries. Only DiskANN tools need program_options which has a linker library,
# but its size is small. Reduce number of dependent DLLs by linking statically.
if (MSVC)
    set(Boost_USE_STATIC_LIBS ON)
endif()

find_package(Boost COMPONENTS program_options)

# For Windows, fall back to nuget version if find_package didn't find it.
if (MSVC AND NOT Boost_FOUND)
    set(DISKANN_BOOST_INCLUDE "${DISKANN_MSVC_PACKAGES}/boost/lib/native/include")
    # Multi-threaded static library.
    set(PROGRAM_OPTIONS_LIB_PATTERN "${DISKANN_MSVC_PACKAGES}/boost_program_options-vc${MSVC_TOOLSET_VERSION}/lib/native/libboost_program_options-vc${MSVC_TOOLSET_VERSION}-mt-x64-*.lib")
    file(GLOB DISKANN_BOOST_PROGRAM_OPTIONS_LIB ${PROGRAM_OPTIONS_LIB_PATTERN})

    set(PROGRAM_OPTIONS_DLIB_PATTERN "${DISKANN_MSVC_PACKAGES}/boost_program_options-vc${MSVC_TOOLSET_VERSION}/lib/native/libboost_program_options-vc${MSVC_TOOLSET_VERSION}-mt-gd-x64-*.lib")
    file(GLOB DISKANN_BOOST_PROGRAM_OPTIONS_DLIB ${PROGRAM_OPTIONS_DLIB_PATTERN})

    if (EXISTS ${DISKANN_BOOST_INCLUDE} AND EXISTS ${DISKANN_BOOST_PROGRAM_OPTIONS_LIB} AND EXISTS ${DISKANN_BOOST_PROGRAM_OPTIONS_DLIB})
        set(Boost_FOUND ON)
        set(Boost_INCLUDE_DIR ${DISKANN_BOOST_INCLUDE})
        add_library(Boost::program_options STATIC IMPORTED)
        set_target_properties(Boost::program_options PROPERTIES IMPORTED_LOCATION_RELEASE "${DISKANN_BOOST_PROGRAM_OPTIONS_LIB}")
        set_target_properties(Boost::program_options PROPERTIES IMPORTED_LOCATION_DEBUG "${DISKANN_BOOST_PROGRAM_OPTIONS_DLIB}")
        message(STATUS "Falling back to using Boost from the nuget package")
    else()
        message(WARNING "Couldn't find Boost. Was looking for ${DISKANN_BOOST_INCLUDE} and ${PROGRAM_OPTIONS_LIB_PATTERN}")
    endif()
endif()

if (NOT Boost_FOUND)
    message(FATAL_ERROR "Couldn't find Boost dependency")
endif()

include_directories(${Boost_INCLUDE_DIR})

#MKL Config
if (MSVC)
    # Only the DiskANN DLL and one of the tools need MKL libraries. Additionally, only a small part of MKL is used.
    # Given that and given that MKL DLLs are huge, use static linking to end up with no MKL DLL dependencies and with
    # significantly smaller disk footprint.
    #
    # The compile options are not modified as there's already an unconditional -DMKL_ILP64 define below
    # for all architectures, which is all that's needed.
    set(DISKANN_MKL_INCLUDE_DIRECTORIES "${DISKANN_MSVC_PACKAGES}/intelmkl.static.win-x64/lib/native/include")
    set(DISKANN_MKL_LIB_PATH "${DISKANN_MSVC_PACKAGES}/intelmkl.static.win-x64/lib/native/win-x64")

    set(DISKANN_MKL_LINK_LIBRARIES
        "${DISKANN_MKL_LIB_PATH}/mkl_intel_ilp64.lib"
        "${DISKANN_MKL_LIB_PATH}/mkl_core.lib"
        "${DISKANN_MKL_LIB_PATH}/mkl_intel_thread.lib")
else()
    # expected path for manual intel mkl installs
    set(POSSIBLE_OMP_PATHS "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.so;/usr/lib/x86_64-linux-gnu/libiomp5.so;/opt/intel/lib/intel64_lin/libiomp5.so")
    foreach(POSSIBLE_OMP_PATH ${POSSIBLE_OMP_PATHS})
        if (EXISTS ${POSSIBLE_OMP_PATH})
            get_filename_component(OMP_PATH ${POSSIBLE_OMP_PATH} DIRECTORY)
        endif()
    endforeach()

    if(NOT OMP_PATH)
        message(FATAL_ERROR "Could not find Intel OMP in standard locations; use -DOMP_PATH to specify the install location for your environment")
    endif()
    link_directories(${OMP_PATH})

    set(POSSIBLE_MKL_LIB_PATHS "/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so;/usr/lib/x86_64-linux-gnu/libmkl_core.so;/opt/intel/mkl/lib/intel64/libmkl_core.so")
    foreach(POSSIBLE_MKL_LIB_PATH ${POSSIBLE_MKL_LIB_PATHS})
        if (EXISTS ${POSSIBLE_MKL_LIB_PATH})
            get_filename_component(MKL_PATH ${POSSIBLE_MKL_LIB_PATH} DIRECTORY)
        endif()
    endforeach()

    set(POSSIBLE_MKL_INCLUDE_PATHS "/opt/intel/oneapi/mkl/latest/include;/usr/include/mkl;/opt/intel/mkl/include/;")
    foreach(POSSIBLE_MKL_INCLUDE_PATH ${POSSIBLE_MKL_INCLUDE_PATHS})
        if (EXISTS ${POSSIBLE_MKL_INCLUDE_PATH})
            set(MKL_INCLUDE_PATH ${POSSIBLE_MKL_INCLUDE_PATH})
        endif()
    endforeach()
    if(NOT MKL_PATH)
        message(FATAL_ERROR "Could not find Intel MKL in standard locations; use -DMKL_PATH to specify the install location for your environment")
    elseif(NOT MKL_INCLUDE_PATH)
        message(FATAL_ERROR "Could not find Intel MKL in standard locations; use -DMKL_INCLUDE_PATH to specify the install location for headers for your environment")
    endif()
    if (EXISTS ${MKL_PATH}/libmkl_def.so.2)
        set(MKL_DEF_SO ${MKL_PATH}/libmkl_def.so.2)
    elseif(EXISTS ${MKL_PATH}/libmkl_def.so)
        set(MKL_DEF_SO ${MKL_PATH}/libmkl_def.so)
    else()
        message(FATAL_ERROR "Despite finding MKL, libmkl_def.so was not found in expected locations.")
    endif()
    link_directories(${MKL_PATH})
    include_directories(${MKL_INCLUDE_PATH})

    # compile flags and link libraries
    add_compile_options(-m64 -Wl,--no-as-needed)
    if (NOT PYBIND)
        link_libraries(mkl_intel_ilp64 mkl_intel_thread mkl_core iomp5 pthread m dl)
    else()
        # static linking for python so as to minimize customer dependency issues
        link_libraries(
                ${MKL_PATH}/libmkl_intel_ilp64.a
                ${MKL_PATH}/libmkl_intel_thread.a
                ${MKL_PATH}/libmkl_core.a
                ${MKL_DEF_SO}
                iomp5
                pthread
                m
                dl
        )
    endif()
endif()

add_definitions(-DMKL_ILP64)

# Section for tcmalloc. The DiskANN tools are always linked to tcmalloc. For Windows, they also need to
# force-include the _tcmalloc symbol for enabling tcmalloc.
#
# The DLL itself needs to be linked to tcmalloc only if DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS
# is enabled.
if (MSVC)
    if (NOT EXISTS "${PROJECT_SOURCE_DIR}/gperftools/gperftools.sln")
        message(FATAL_ERROR "The gperftools submodule was not found. "
                "Please check-out git submodules by doing 'git submodule init' followed by 'git submodule update'")
    endif()

    set(TCMALLOC_LINK_LIBRARY "${PROJECT_SOURCE_DIR}/gperftools/x64/Release-Patch/libtcmalloc_minimal.lib")
    set(TCMALLOC_WINDOWS_RUNTIME_FILES
        "${PROJECT_SOURCE_DIR}/gperftools/x64/Release-Patch/libtcmalloc_minimal.dll"
        "${PROJECT_SOURCE_DIR}/gperftools/x64/Release-Patch/libtcmalloc_minimal.pdb")

    # Tell CMake how to build the tcmalloc linker library from the submodule.
    add_custom_target(build_libtcmalloc_minimal DEPENDS ${TCMALLOC_LINK_LIBRARY})
    add_custom_command(OUTPUT ${TCMALLOC_LINK_LIBRARY}
                       COMMAND ${CMAKE_VS_MSBUILD_COMMAND} gperftools.sln /m /nologo
                           /t:libtcmalloc_minimal /p:Configuration="Release-Patch"
                           /property:Platform="x64"
                           /p:PlatformToolset=v${MSVC_TOOLSET_VERSION}
                           /p:WindowsTargetPlatformVersion=${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}
                       WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/gperftools)

    add_library(libtcmalloc_minimal_for_exe STATIC IMPORTED)
    add_library(libtcmalloc_minimal_for_dll STATIC IMPORTED)

    set_target_properties(libtcmalloc_minimal_for_dll PROPERTIES
                          IMPORTED_LOCATION "${TCMALLOC_LINK_LIBRARY}")

    set_target_properties(libtcmalloc_minimal_for_exe PROPERTIES
                          IMPORTED_LOCATION "${TCMALLOC_LINK_LIBRARY}"
                          INTERFACE_LINK_OPTIONS /INCLUDE:_tcmalloc)

    # Ensure libtcmalloc_minimal is built before it's being used.
    add_dependencies(libtcmalloc_minimal_for_dll build_libtcmalloc_minimal)
    add_dependencies(libtcmalloc_minimal_for_exe build_libtcmalloc_minimal)

    set(DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS libtcmalloc_minimal_for_exe)
elseif(NOT PYBIND)
    set(DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS "-ltcmalloc")
endif()

if (DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS)
    add_definitions(-DRELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS)

    if (MSVC)
        set(DISKANN_DLL_TCMALLOC_LINK_OPTIONS libtcmalloc_minimal_for_dll)
    endif()
endif()

if (NOT MSVC)
    set(DISKANN_ASYNC_LIB aio)
endif()

#Main compiler/linker settings 
if(MSVC)
	#language options
	add_compile_options(/permissive- /openmp:experimental /Zc:twoPhase- /Zc:inline /WX- /std:c++17 /Gd /W3 /MP /Zi /FC /nologo)
	#code generation options
	add_compile_options(/arch:AVX2 /fp:fast /fp:except- /EHsc /GS- /Gy)
	#optimization options
	add_compile_options(/Ot /Oy /Oi)
	#path options
	add_definitions(-DUSE_AVX2 -DUSE_ACCELERATED_PQ -D_WINDOWS -DNOMINMAX -DUNICODE)
    # Linker options. Exclude VCOMP/VCOMPD.LIB which contain VisualStudio's version of OpenMP.
    # MKL was linked against Intel's OpenMP and depends on the corresponding DLL.
    add_link_options(/NODEFAULTLIB:VCOMP.LIB /NODEFAULTLIB:VCOMPD.LIB /DEBUG:FULL /OPT:REF /OPT:ICF)
	
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/x64/Debug)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/x64/Debug)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/x64/Debug)

	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/x64/Release)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/x64/Release)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/x64/Release)
else()
	set(ENV{TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD} 500000000000)
    #	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG -O0 -fsanitize=address -fsanitize=leak -fsanitize=undefined")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG -Wall -Wextra")
    if (NOT PYBIND)
	    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -DNDEBUG -march=native -mtune=native -ftree-vectorize")
    else()
        #-Ofast is super problematic for python.  see: https://moyix.blogspot.com/2022/09/someones-been-messing-with-my-subnormals.html
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -march=native -mtune=native -ftree-vectorize")
        add_compile_options(-fPIC)
    endif()
	add_compile_options(-march=native -Wall -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_AVX2)
endif()

add_subdirectory(src)
if (NOT PYBIND)
    add_subdirectory(apps)
    add_subdirectory(apps/utils)
endif()

if (UNIT_TEST)
    enable_testing()
    add_subdirectory(tests)
endif()

if (MSVC)
    message(STATUS "The ${PROJECT_NAME}.sln has been created, opened it from VisualStudio to build Release or Debug configurations.\n"
                   "Alternatively, use MSBuild to build:\n\n"
                   "msbuild.exe ${PROJECT_NAME}.sln /m /nologo /t:Build /p:Configuration=\"Release\" /property:Platform=\"x64\"\n")
endif()

if (RESTAPI)
    if (MSVC)
        set(DISKANN_CPPRESTSDK "${DISKANN_MSVC_PACKAGES}/cpprestsdk.v142/build/native")
        	# expected path for apt packaged intel mkl installs
        link_libraries("${DISKANN_CPPRESTSDK}/x64/lib/cpprest142_2_10.lib")
        include_directories("${DISKANN_CPPRESTSDK}/include")
    endif()
    add_subdirectory(apps/restapi)
endif()

include(clang-format.cmake)

if(PYBIND)
    add_subdirectory(python)
else()
    message(STATUS "Not building python bindings")
endif()
