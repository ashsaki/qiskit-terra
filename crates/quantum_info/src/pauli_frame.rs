// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::clifford::Pauli1q;

/// A Clifford operator `C`, stored as the images `C† X_q C` and `C† Z_q C` of the Pauli
/// generators.
///
/// This type gives the same results as [`crate::clifford::Clifford`] for the methods that both
/// types have. Use this type when you evolve many Paulis by a Clifford that changes gate by gate.
/// The evolution of a single-qubit Pauli reads one row. The evolution of a Pauli of weight `w`
/// multiplies at most `2 w` rows.
///
/// Row `q` holds the image of `X_q` and row `num_qubits + q` holds the image of `Z_q`. Each row
/// is the Pauli `i^phase X^x Z^z`. On each qubit, the `X` factor is on the left of the `Z` factor.
/// A row has `words` 64-bit words for the `x` bits, followed by `words` 64-bit words for the `z`
/// bits.
#[derive(Clone, Debug)]
pub struct PauliFrame {
    num_qubits: usize,
    /// The number of 64-bit words that hold the `x` bits (or the `z` bits) of one row.
    words: usize,
    /// The bits of all rows, one row after the other.
    xz: Vec<u64>,
    /// The power of `i` in each row, modulo 4.
    phase: Vec<u8>,
}

impl PauliFrame {
    /// Creates the frame of the identity Clifford on `num_qubits` qubits.
    pub fn identity(num_qubits: usize) -> Self {
        let words = num_qubits.div_ceil(64);
        let stride = 2 * words;
        let mut xz = vec![0; 2 * num_qubits * stride];
        for q in 0..num_qubits {
            xz[q * stride + q / 64] |= 1 << (q % 64);
            xz[(num_qubits + q) * stride + words + q / 64] |= 1 << (q % 64);
        }
        Self {
            num_qubits,
            words,
            xz,
            phase: vec![0; 2 * num_qubits],
        }
    }

    #[inline]
    fn stride(&self) -> usize {
        2 * self.words
    }

    #[inline]
    fn row(&self, row: usize) -> &[u64] {
        let stride = self.stride();
        &self.xz[row * stride..(row + 1) * stride]
    }

    /// Sets the row `target` to `i^extra * target * source`.
    fn multiply_rows(&mut self, target: usize, source: usize, extra: u8) {
        let stride = self.stride();
        let (target_row, source_row) = if target < source {
            let (left, right) = self.xz.split_at_mut(source * stride);
            (
                &mut left[target * stride..(target + 1) * stride],
                &right[..stride],
            )
        } else {
            let (left, right) = self.xz.split_at_mut(target * stride);
            (
                &mut right[..stride],
                &left[source * stride..(source + 1) * stride],
            )
        };
        let sign = multiply(target_row, source_row, self.words);
        self.phase[target] = (self.phase[target] + self.phase[source] + sign + extra) & 3;
    }

    fn swap_rows(&mut self, a: usize, b: usize) {
        let stride = self.stride();
        let (low, high) = (a.min(b), a.max(b));
        let (left, right) = self.xz.split_at_mut(high * stride);
        left[low * stride..(low + 1) * stride].swap_with_slice(&mut right[..stride]);
        self.phase.swap(a, b);
    }

    /// Changes the sign of a row.
    #[inline]
    fn negate_row(&mut self, row: usize) {
        self.phase[row] = (self.phase[row] + 2) & 3;
    }

    /// Modifies the frame in-place by appending an S-gate.
    pub fn append_s(&mut self, qubit: usize) {
        self.multiply_rows(qubit, self.num_qubits + qubit, 3);
    }

    /// Modifies the frame in-place by appending an Sdg-gate.
    pub fn append_sdg(&mut self, qubit: usize) {
        self.multiply_rows(qubit, self.num_qubits + qubit, 1);
    }

    /// Modifies the frame in-place by appending an SX-gate.
    pub fn append_sx(&mut self, qubit: usize) {
        self.multiply_rows(self.num_qubits + qubit, qubit, 3);
    }

    /// Modifies the frame in-place by appending an SXdg-gate.
    pub fn append_sxdg(&mut self, qubit: usize) {
        self.multiply_rows(self.num_qubits + qubit, qubit, 1);
    }

    /// Modifies the frame in-place by appending an H-gate.
    pub fn append_h(&mut self, qubit: usize) {
        self.swap_rows(qubit, self.num_qubits + qubit);
    }

    /// Modifies the frame in-place by appending a SWAP-gate.
    pub fn append_swap(&mut self, qubit0: usize, qubit1: usize) {
        self.swap_rows(qubit0, qubit1);
        self.swap_rows(self.num_qubits + qubit0, self.num_qubits + qubit1);
    }

    /// Modifies the frame in-place by appending a CX-gate.
    pub fn append_cx(&mut self, qubit0: usize, qubit1: usize) {
        self.multiply_rows(qubit0, qubit1, 0);
        self.multiply_rows(self.num_qubits + qubit1, self.num_qubits + qubit0, 0);
    }

    /// Modifies the frame in-place by appending a CZ-gate.
    pub fn append_cz(&mut self, qubit0: usize, qubit1: usize) {
        self.multiply_rows(qubit0, self.num_qubits + qubit1, 0);
        self.multiply_rows(qubit1, self.num_qubits + qubit0, 0);
    }

    /// Modifies the frame in-place by appending a CY-gate.
    pub fn append_cy(&mut self, qubit0: usize, qubit1: usize) {
        self.append_sdg(qubit1);
        self.append_cx(qubit0, qubit1);
        self.append_s(qubit1);
    }

    /// Modifies the frame in-place by appending an X-gate.
    pub fn append_x(&mut self, qubit: usize) {
        self.negate_row(self.num_qubits + qubit);
    }

    /// Modifies the frame in-place by appending a Z-gate.
    pub fn append_z(&mut self, qubit: usize) {
        self.negate_row(qubit);
    }

    /// Modifies the frame in-place by appending a Y-gate.
    pub fn append_y(&mut self, qubit: usize) {
        self.append_x(qubit);
        self.append_z(qubit);
    }

    /// Modifies the frame in-place by appending an iSWAP-gate.
    pub fn append_iswap(&mut self, qubit0: usize, qubit1: usize) {
        self.append_s(qubit0);
        self.append_s(qubit1);
        self.append_h(qubit0);
        self.append_cx(qubit0, qubit1);
        self.append_cx(qubit1, qubit0);
        self.append_h(qubit1);
    }

    /// Modifies the frame in-place by appending an ECR-gate.
    pub fn append_ecr(&mut self, qubit0: usize, qubit1: usize) {
        self.append_s(qubit0);
        self.append_sx(qubit1);
        self.append_cx(qubit0, qubit1);
        self.append_x(qubit0);
    }

    /// Modifies the frame in-place by appending a DCX-gate.
    pub fn append_dcx(&mut self, qubit0: usize, qubit1: usize) {
        self.append_cx(qubit0, qubit1);
        self.append_cx(qubit1, qubit0);
    }

    /// Modifies the frame in-place by appending an RZ-gate. The angle of the gate is
    /// `multiple * pi/2`.
    pub fn append_rz(&mut self, qubit: usize, multiple: usize) {
        match multiple % 4 {
            0 => {}
            1 => self.append_s(qubit),
            2 => self.append_z(qubit),
            3 => self.append_sdg(qubit),
            _ => unreachable!("The value is smaller than 4."),
        }
    }

    /// Modifies the frame in-place by appending an RX-gate. The angle of the gate is
    /// `multiple * pi/2`.
    pub fn append_rx(&mut self, qubit: usize, multiple: usize) {
        match multiple % 4 {
            0 => {}
            1 => self.append_sx(qubit),
            2 => self.append_x(qubit),
            3 => self.append_sxdg(qubit),
            _ => unreachable!("The value is smaller than 4."),
        }
    }

    /// Modifies the frame in-place by appending an RY-gate. The angle of the gate is
    /// `multiple * pi/2`.
    pub fn append_ry(&mut self, qubit: usize, multiple: usize) {
        match multiple % 4 {
            0 => {}
            1 => {
                self.append_z(qubit);
                self.append_h(qubit);
            }
            2 => self.append_y(qubit),
            3 => {
                self.append_h(qubit);
                self.append_z(qubit);
            }
            _ => unreachable!("The value is smaller than 4."),
        }
    }

    /// Modifies the frame in-place by appending a Pauli product rotation. The angle of the
    /// rotation is `multiple * pi/2`. The Pauli has the bits `pauli_z[j]` and `pauli_x[j]` on the
    /// qubit `indices[j]`.
    pub fn append_ppr(
        &mut self,
        pauli_z: &[bool],
        pauli_x: &[bool],
        indices: &[u32],
        multiple: usize,
    ) {
        let multiple = multiple % 4;
        if multiple == 0 {
            return;
        }
        // The rotation changes only the generators that anticommute with the Pauli.
        // A generator X_q anticommutes with the Pauli if the Pauli has a Z bit on qubit q.
        // A generator Z_q anticommutes with the Pauli if the Pauli has an X bit on qubit q.
        let rows = pauli_z
            .iter()
            .zip(pauli_x)
            .zip(indices)
            .flat_map(|((&z, &x), &qubit)| {
                let qubit = qubit as usize;
                [(z, qubit), (x, self.num_qubits + qubit)]
            })
            .filter_map(|(anticommutes, row)| anticommutes.then_some(row))
            .collect::<Vec<_>>();
        if multiple == 2 {
            for row in rows {
                self.negate_row(row);
            }
            return;
        }
        let (image, image_phase) = self.image(pauli_z, pauli_x, indices);
        let extra = if multiple == 1 { 3 } else { 1 };
        for row in rows {
            let stride = self.stride();
            let sign = multiply(
                &mut self.xz[row * stride..(row + 1) * stride],
                &image,
                self.words,
            );
            self.phase[row] = (self.phase[row] + image_phase + sign + extra) & 3;
        }
    }

    /// Returns the image `C† P C` of a Pauli `P` as the bits of one row and the power of `i`.
    /// The Pauli has the bits `pauli_z[j]` and `pauli_x[j]` on the qubit `indices[j]`.
    fn image(&self, pauli_z: &[bool], pauli_x: &[bool], indices: &[u32]) -> (Vec<u64>, u8) {
        let mut image = vec![0; self.stride()];
        let mut phase = 0;
        for ((&z, &x), &qubit) in pauli_z.iter().zip(pauli_x).zip(indices) {
            let qubit = qubit as usize;
            if x {
                phase += self.phase[qubit] + multiply(&mut image, self.row(qubit), self.words);
            }
            if z {
                let row = self.num_qubits + qubit;
                phase += self.phase[row] + multiply(&mut image, self.row(row), self.words);
            }
            if x && z {
                // Y = i X Z.
                phase += 1;
            }
            phase &= 3;
        }
        (image, phase)
    }

    /// Evolves a Pauli by the Clifford. The Pauli has the bits `in_z[j]` and `in_x[j]` on the
    /// qubit `indices_in[j]`.
    ///
    /// Returns the evolved Pauli in the sparse ZX format `(sign, z, x, indices)`, as
    /// [`crate::clifford::Clifford::evolve_pauli`] does.
    pub fn evolve_pauli(
        &self,
        in_z: &[bool],
        in_x: &[bool],
        indices_in: &[u32],
    ) -> (bool, Vec<bool>, Vec<bool>, Vec<u32>) {
        if in_z.iter().zip(in_x).all(|(&z, &x)| !z && !x) {
            return (false, in_z.to_vec(), in_x.to_vec(), indices_in.to_vec());
        }
        let (image, phase) = self.image(in_z, in_x, indices_in);
        to_sparse(&image, phase, self.words)
    }

    /// Evolves a single-qubit Pauli on the qubit `qbit` by the Clifford.
    ///
    /// Returns the evolved Pauli in the sparse ZX format `(sign, z, x, indices)`, as
    /// [`crate::clifford::Clifford::evolve_single_qubit_pauli`] does.
    pub fn evolve_single_qubit_pauli(
        &self,
        pauli: Pauli1q,
        qbit: usize,
    ) -> (bool, Vec<bool>, Vec<bool>, Vec<u32>) {
        match pauli {
            Pauli1q::X => to_sparse(self.row(qbit), self.phase[qbit], self.words),
            Pauli1q::Z => {
                let row = self.num_qubits + qbit;
                to_sparse(self.row(row), self.phase[row], self.words)
            }
            Pauli1q::Y => self.evolve_pauli(&[true], &[true], &[qbit as u32]),
        }
    }
}

/// Sets `left` to the product `left * right` of two rows, and returns the power of `i` that the
/// product adds, modulo 4.
#[inline]
fn multiply(left: &mut [u64], right: &[u64], words: usize) -> u8 {
    let (left_x, left_z) = left.split_at_mut(words);
    let (right_x, right_z) = right.split_at(words);
    // Each Z bit of `left` must move to the right of an X bit of `right` on the same qubit.
    // Each such move changes the sign.
    let mut parity = 0;
    for (left_z, right_x) in left_z.iter().zip(right_x) {
        parity ^= left_z & right_x;
    }
    for (left_x, right_x) in left_x.iter_mut().zip(right_x) {
        *left_x ^= right_x;
    }
    for (left_z, right_z) in left_z.iter_mut().zip(right_z) {
        *left_z ^= right_z;
    }
    2 * (parity.count_ones() & 1) as u8
}

/// Converts a Hermitian row to the sparse ZX format `(sign, z, x, indices)`. The indices are in
/// ascending order.
fn to_sparse(row: &[u64], phase: u8, words: usize) -> (bool, Vec<bool>, Vec<bool>, Vec<u32>) {
    let (row_x, row_z) = row.split_at(words);
    let mut num_y = 0;
    let mut num_terms = 0;
    for (x, z) in row_x.iter().zip(row_z) {
        num_y += (x & z).count_ones();
        num_terms += (x | z).count_ones() as usize;
    }
    // The row is `i^phase X^x Z^z` and each Y is `i X Z`. Thus the sign is `i^(phase - num_y)`.
    let sign = (phase as u32 + 4 - (num_y & 3)) & 3;
    debug_assert!(sign == 0 || sign == 2, "The row is not Hermitian.");
    let mut z = Vec::with_capacity(num_terms);
    let mut x = Vec::with_capacity(num_terms);
    let mut indices = Vec::with_capacity(num_terms);
    for (word, (&word_x, &word_z)) in row_x.iter().zip(row_z).enumerate() {
        let mut bits = word_x | word_z;
        while bits != 0 {
            let bit = bits.trailing_zeros();
            bits &= bits - 1;
            z.push((word_z >> bit) & 1 == 1);
            x.push((word_x >> bit) & 1 == 1);
            indices.push(64 * word as u32 + bit);
        }
    }
    (sign == 2, z, x, indices)
}

#[cfg(test)]
mod tests {
    use super::PauliFrame;
    use crate::clifford::{Clifford, Pauli1q};
    use rand::prelude::*;
    use rand_pcg::Pcg64Mcg;

    /// Returns a random Pauli on `weight` different qubits. The Pauli can have identity terms.
    fn random_pauli(
        rng: &mut Pcg64Mcg,
        num_qubits: usize,
        weight: usize,
    ) -> (Vec<bool>, Vec<bool>, Vec<u32>) {
        let mut indices: Vec<u32> = (0..num_qubits as u32).collect();
        indices.shuffle(rng);
        indices.truncate(weight);
        let (z, x) = (0..weight)
            .map(|_| {
                [(false, false), (true, false), (false, true), (true, true)][rng.random_range(0..4)]
            })
            .unzip();
        (z, x, indices)
    }

    /// Appends the same random Clifford gate to the Clifford and to the frame.
    fn append_random_gate(rng: &mut Pcg64Mcg, clifford: &mut Clifford, frame: &mut PauliFrame) {
        let num_qubits = clifford.num_qubits();
        let q0 = rng.random_range(0..num_qubits);
        let q1 = (q0 + rng.random_range(1..num_qubits.max(2))) % num_qubits;
        let multiple = rng.random_range(0..8);
        let num_kinds = if num_qubits == 1 { 12 } else { 19 };
        match rng.random_range(0..num_kinds) {
            0 => {
                clifford.append_x(q0);
                frame.append_x(q0);
            }
            1 => {
                clifford.append_y(q0);
                frame.append_y(q0);
            }
            2 => {
                clifford.append_z(q0);
                frame.append_z(q0);
            }
            3 => {
                clifford.append_h(q0);
                frame.append_h(q0);
            }
            4 => {
                clifford.append_s(q0);
                frame.append_s(q0);
            }
            5 => {
                clifford.append_sdg(q0);
                frame.append_sdg(q0);
            }
            6 => {
                clifford.append_sx(q0);
                frame.append_sx(q0);
            }
            7 => {
                clifford.append_sxdg(q0);
                frame.append_sxdg(q0);
            }
            8 => {
                clifford.append_rz(q0, multiple);
                frame.append_rz(q0, multiple);
            }
            9 => {
                clifford.append_rx(q0, multiple);
                frame.append_rx(q0, multiple);
            }
            10 => {
                clifford.append_ry(q0, multiple);
                frame.append_ry(q0, multiple);
            }
            11 => {
                let weight = rng.random_range(1..=num_qubits.min(6));
                let (z, x, indices) = random_pauli(rng, num_qubits, weight);
                clifford.append_ppr(&z, &x, &indices, multiple);
                frame.append_ppr(&z, &x, &indices, multiple);
            }
            12 => {
                clifford.append_cx(q0, q1);
                frame.append_cx(q0, q1);
            }
            13 => {
                clifford.append_cz(q0, q1);
                frame.append_cz(q0, q1);
            }
            14 => {
                clifford.append_cy(q0, q1);
                frame.append_cy(q0, q1);
            }
            15 => {
                clifford.append_swap(q0, q1);
                frame.append_swap(q0, q1);
            }
            16 => {
                clifford.append_iswap(q0, q1);
                frame.append_iswap(q0, q1);
            }
            17 => {
                clifford.append_ecr(q0, q1);
                frame.append_ecr(q0, q1);
            }
            18 => {
                clifford.append_dcx(q0, q1);
                frame.append_dcx(q0, q1);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_evolve_matches_clifford() {
        for num_qubits in [1, 2, 5, 64, 65, 130] {
            let mut rng = Pcg64Mcg::seed_from_u64(num_qubits as u64);
            let mut clifford = Clifford::identity(num_qubits);
            let mut frame = PauliFrame::identity(num_qubits);
            for _ in 0..30 {
                for _ in 0..10 {
                    append_random_gate(&mut rng, &mut clifford, &mut frame);
                }
                for qubit in [0, num_qubits / 2, num_qubits - 1] {
                    for pauli in [Pauli1q::X, Pauli1q::Y, Pauli1q::Z] {
                        assert_eq!(
                            frame.evolve_single_qubit_pauli(pauli, qubit),
                            clifford.evolve_single_qubit_pauli(pauli, qubit)
                        );
                    }
                }
                let weight = rng.random_range(1..=num_qubits.min(6));
                let (z, x, indices) = random_pauli(&mut rng, num_qubits, weight);
                assert_eq!(
                    frame.evolve_pauli(&z, &x, &indices),
                    clifford.evolve_pauli(&z, &x, &indices)
                );
            }
        }
    }

    #[test]
    fn test_evolve_identity_pauli() {
        let mut frame = PauliFrame::identity(3);
        frame.append_h(0);
        frame.append_cx(0, 2);
        let (z, x, indices) = (vec![false, false], vec![false, false], vec![2, 0]);
        assert_eq!(
            frame.evolve_pauli(&z, &x, &indices),
            (false, z.clone(), x.clone(), indices.clone())
        );
    }
}
