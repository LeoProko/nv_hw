import torch


def mel_loss(output_spec: torch.Tensor, target_spec: torch.Tensor):
    output_spec.squeeze()
    if len(output_spec.shape) == 2:
        output_spec = output_spec.unsqueeze(0)

    spec_len = min(output_spec.size(-1), target_spec.size(-1))
    output_spec = output_spec[:, :, :spec_len]
    target_spec = target_spec[:, :, :spec_len]

    return torch.nn.functional.l1_loss(output_spec, target_spec) * 45


def generator_loss(dgses: torch.Tensor):
    loss = 0
    for dgs in dgses:
        loss += torch.mean((dgs - 1) ** 2)

    return loss


def feature_loss(target_fms: torch.Tensor, gen_fms: torch.Tensor):
    loss = 0
    for target_fm, gen_fm in zip(target_fms, gen_fms):
        for tfm, gfm in zip(target_fm, gen_fm):
            if len(tfm.shape) == 2:
                tfm = tfm.unsqueeze(0)

            if tfm.size(1) != gfm.size(1):
                size = min(tfm.size(1), gfm.size(1))
                tfm = tfm[:, :size, :]
                gfm = gfm[:, :size, :]
            if tfm.size(2) != gfm.size(2):
                size = min(tfm.size(2), gfm.size(2))
                tfm = tfm[:, :, :size]
                gfm = gfm[:, :, :size]

            loss += torch.mean(torch.abs(tfm - gfm))

    return loss * 2


def discriminator_loss(target_outputs: torch.Tensor, gen_outputs: torch.Tensor):
    loss = 0
    for target_output, gen_output in zip(target_outputs, gen_outputs):
        target_loss = torch.mean((target_output - 1) ** 2)
        generator_loss = torch.mean(gen_output**2)
        loss += target_loss + generator_loss

    return loss
