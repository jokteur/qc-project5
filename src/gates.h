#pragma once
#include "kokkos.h"
#include "complex.h"

/**
 * Could extend to support more gates
 */
enum class GateType {
    X,
    Y,
    Z,
    H,
    T,
    SqrtX,
    SqrtY,
    CX,
    CZ
};


struct Gate {
    GateType type;
    int target;
    int control = -1; // In case of single qubit gate, control is -1 and not used
    int cycle;
};

std::string gate_to_text(GateType gate) {
    switch (gate) {
    case GateType::X:
        return "X";
    case GateType::Y:
        return "Y";
    case GateType::Z:
        return "Z";
    case GateType::H:
        return "H";
    case GateType::T:
        return "T";
    case GateType::SqrtX:
        return "SqrtX";
    case GateType::SqrtY:
        return "SqrtY";
    case GateType::CX:
        return "CX";
    case GateType::CZ:
        return "CZ";
    }
    return "";
}

GateType text_to_gate(const std::string& text) {
    if (text == "X" || text == "x")
        return GateType::X;
    if (text == "Y" || text == "y")
        return GateType::Y;
    if (text == "Z" || text == "z")
        return GateType::Z;
    if (text == "H" || text == "h")
        return GateType::H;
    if (text == "T" || text == "t")
        return GateType::T;
    if (text == "SqrtX" || text == "x_1_2")
        return GateType::SqrtX;
    if (text == "SqrtY" || text == "y_1_2")
        return GateType::SqrtY;
    if (text == "CX" || text == "cx")
        return GateType::CX;
    if (text == "CZ" || text == "cz")
        return GateType::CZ;

    throw std::runtime_error("Unknown gate: " + text);
}

KOKKOS_INLINE_FUNCTION void z_gate(cmplx a0, cmplx a1, cmplx* new_w) {
    new_w[0] = a0;
    new_w[1] = -a1;
}

KOKKOS_INLINE_FUNCTION void y_gate(cmplx a0, cmplx a1, cmplx* new_w) {
    cmplx j = cmplx(0, 1);
    new_w[0] = -a1 * j;
    new_w[1] = a0 * j;
}

KOKKOS_INLINE_FUNCTION void t_gate(cmplx a0, cmplx a1, cmplx* new_w) {
    cmplx j = cmplx(0, 1);
    /**
     * We can rewrite T gate as:
     * T = diag(1, exp(-j * pi / 4)) = diag(1, 1/sqrt(2)*(1-i))
     *   = 1/sqrt(2) * diag(sqrt(2), (1-i))
     *
     * Now we can take 1/sqrt(2) in the sqrt_counter
     */
     new_w[0] = Kokkos::sqrt(2) * a0 /* * 1.0 / Kokkos::sqrt(2) */;
     new_w[1] = a1 * (1 + j) /* * 1.0 / Kokkos::sqrt(2) */;
}

KOKKOS_INLINE_FUNCTION void p0_gate(cmplx a0, cmplx a1, cmplx* new_w) {
    new_w[0] = a0;
    new_w[1] = a1;
}

KOKKOS_INLINE_FUNCTION void x_gate(cmplx a0, cmplx a1, cmplx* new_w) {
    new_w[0] = a1;
    new_w[1] = a0;
}
KOKKOS_INLINE_FUNCTION void h_gate(cmplx a0, cmplx a1, cmplx* new_w) {
    // 1/2 is taken into account in sqrt_counter
    new_w[0] = (a0 + a1) /* * 1.0 / Kokkos::sqrt(2) */;
    new_w[1] = (a0 - a1) /* * 1.0 / Kokkos::sqrt(2) */;
}

KOKKOS_INLINE_FUNCTION void sqrt_x_gate(cmplx a0, cmplx a1, cmplx* new_w) {
    cmplx j = cmplx(0, 1);
    // 1/2 is taken into account in sqrt_counter
    new_w[0] = ((1 + j)*a0 + (1 - j)*a1) /* *0.5 */;
    new_w[1] = ((1 - j)*a0 + (1 + j)*a1) /* *0.5 */;
}

KOKKOS_INLINE_FUNCTION void sqrt_y_gate(cmplx a0, cmplx a1, cmplx* new_w) {
    cmplx j = cmplx(0, 1);
    // 1/2 is taken into account in sqrt_counter
    new_w[0] = ((1 + j)*a0 - (1 + j)*a1) /* *0.5 */;
    new_w[1] = ((1 + j)*a0 + (1 + j)*a1) /* *0.5 */;
}