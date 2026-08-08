from typing import Optional, Tuple

import torch
import torch.nn as nn


class CausalPhaseDisentangler(nn.Module):
    """
    Causal-Phase Disentangler.

    Core idea:
    - Treat phase as causal structural factor.
    - Treat amplitude as non-causal / subject-sensitive factor.
    - During training, mix amplitude with another sample while preserving phase.

    Given latent map Z:
        FFT(Z) = A(Z) * exp(j * P(Z))

    Perturb:
        A_noise = omega * A(Z_k) + (1 - omega) * A(Z_i)

    Reconstruct:
        Z_tilde = IFFT(A_noise * exp(j * P(Z_i)))

    This module applies 2D FFT over the spatial dimensions of the learned
    latent feature map, not raw voxel sequence and not temporal BOLD.
    """

    def __init__(
        self,
        perturb_prob: float = 0.75,
        omega_min: float = 0.1,
        omega_max: float = 0.6,
        use_batch_shuffle: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.perturb_prob = perturb_prob
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.use_batch_shuffle = use_batch_shuffle
        self.eps = eps

    def spectral_decompose(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Tensor [B, C, H, W]

        Returns:
            amplitude: Tensor [B, C, H, W]
            phase: Tensor [B, C, H, W]
        """
        spectrum = torch.fft.fft2(z, dim=(-2, -1), norm="ortho")
        amplitude = torch.abs(spectrum)
        phase = torch.angle(spectrum)
        return amplitude, phase

    def reconstruct_from_amp_phase(
        self,
        amplitude: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        real = amplitude * torch.cos(phase)
        imag = amplitude * torch.sin(phase)
        complex_spectrum = torch.complex(real, imag)
        z_rec = torch.fft.ifft2(complex_spectrum, dim=(-2, -1), norm="ortho").real
        return z_rec

    def make_counterfactual_amplitude(
        self,
        amplitude: torch.Tensor,
        perm: Optional[torch.Tensor] = None,
        omega: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = amplitude.shape[0]
        device = amplitude.device

        if perm is None:
            if self.use_batch_shuffle:
                perm = torch.randperm(b, device=device)
            else:
                perm = torch.roll(torch.arange(b, device=device), shifts=1)

        if omega is None:
            omega = torch.empty(b, 1, 1, 1, device=device).uniform_(
                self.omega_min,
                self.omega_max,
            )

        amp_k = amplitude[perm]
        amp_noise = omega * amp_k + (1.0 - omega) * amplitude

        return amp_noise, perm, omega

    def forward(
        self,
        z: torch.Tensor,
        force_perturb: bool = False,
    ):
        """
        Returns:
            z_clean: original latent map
            z_perturbed: phase-preserving amplitude-perturbed latent map
            info: diagnostic dictionary
        """
        if z.ndim != 4:
            raise ValueError(f"Expected z shape [B, C, H, W], got {tuple(z.shape)}")

        amplitude, phase = self.spectral_decompose(z)

        do_perturb = force_perturb or (
            self.training and torch.rand(()) < self.perturb_prob
        )

        if do_perturb:
            amp_noise, perm, omega = self.make_counterfactual_amplitude(amplitude)
            z_tilde = self.reconstruct_from_amp_phase(amp_noise, phase)
        else:
            perm = torch.arange(z.shape[0], device=z.device)
            omega = torch.zeros(z.shape[0], 1, 1, 1, device=z.device)
            z_tilde = z

        info = {
            "amplitude": amplitude,
            "phase": phase,
            "perm": perm,
            "omega": omega,
            "perturbed": do_perturb,
        }

        return z, z_tilde, info


def amplitude_phase_swap(z_a: torch.Tensor, z_b: torch.Tensor):
    """
    Utility for spectral swapping analysis.

    Returns:
        amp_swap: phase from A, amplitude from B
        phase_swap: amplitude from A, phase from B
    """
    spec_a = torch.fft.fft2(z_a, dim=(-2, -1), norm="ortho")
    spec_b = torch.fft.fft2(z_b, dim=(-2, -1), norm="ortho")

    amp_a, phase_a = torch.abs(spec_a), torch.angle(spec_a)
    amp_b, phase_b = torch.abs(spec_b), torch.angle(spec_b)

    def rec(amp, phase):
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        return torch.fft.ifft2(torch.complex(real, imag), dim=(-2, -1), norm="ortho").real

    amp_swap = rec(amp_b, phase_a)
    phase_swap = rec(amp_a, phase_b)

    return amp_swap, phase_swap
