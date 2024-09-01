cmake_minimum_required(VERSION 3.5)

project( iyCode )

find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
    file(GLOB_RECURSE COMP_METHOD
        "./gallo/*.cpp"
        "./gallo/*.h"
        "./soros/*.cpp"
        "./soros/*.h"
        "./yun/*.cpp"
        "./yun/*.h"
    )
    
    add_executable( iyBarcode main.cpp ${COMP_METHOD})
    
    target_link_libraries( iyBarcode ${OpenCV_LIBS})
    
endif()





