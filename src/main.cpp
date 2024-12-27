#include "kokkos.h"
#include "io/output/output.h"

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);


    Kokkos::finalize();
}